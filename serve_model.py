"""Inference for the from-scratch transformer (model.py).

Loads a per-language-pair artifact directory produced by train.py:

    models/<pair>/
        weights.pth        # state_dict for build_transformer(...)
        config.json        # architecture + special-token ids + tokenizer filenames
        src_tokenizer.json # HF tokenizers WordLevel/BPE for the source language
        tgt_tokenizer.json # HF tokenizers for the target language (English)

and serves greedy / beam-search translation.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tokenizers import Tokenizer

from model import build_transformer


def select_device(force: str = "") -> str:
    force = (force or "").strip().lower()
    if force in {"cpu", "cuda", "mps"}:
        return force
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _token_id(tok: Tokenizer, config_id: Optional[int], *candidates: str, default: int) -> int:
    if config_id is not None:
        return int(config_id)
    for name in candidates:
        found = tok.token_to_id(name)
        if found is not None:
            return found
    return default


class TranslatorModel:
    """One trained source→English model plus its tokenizers."""

    def __init__(self, model_dir: Path, device: str):
        self.model_dir = Path(model_dir)
        self.device = device

        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise RuntimeError(f"Missing config.json in {self.model_dir}")
        self.config = json.loads(config_path.read_text())

        weights_path = self.model_dir / self.config.get("weights", "weights.pth")
        if not weights_path.exists():
            raise RuntimeError(f"Missing weights file: {weights_path}")

        src_tok_path = self.model_dir / self.config.get("src_tokenizer", "src_tokenizer.json")
        tgt_tok_path = self.model_dir / self.config.get("tgt_tokenizer", "tgt_tokenizer.json")
        for p in (src_tok_path, tgt_tok_path):
            if not p.exists():
                raise RuntimeError(f"Missing tokenizer file: {p}")

        self.src_tok = Tokenizer.from_file(str(src_tok_path))
        self.tgt_tok = Tokenizer.from_file(str(tgt_tok_path))

        self.seq_len = int(self.config["seq_len"])
        self.pad_id = _token_id(self.tgt_tok, self.config.get("pad_id"), "[PAD]", "<pad>", default=1)
        self.sos_id = _token_id(self.tgt_tok, self.config.get("sos_id"), "[SOS]", "[CLS]", "<sos>", default=2)
        self.eos_id = _token_id(self.tgt_tok, self.config.get("eos_id"), "[EOS]", "[SEP]", "<eos>", default=3)

        self.model = build_transformer(
            src_vocab_size=int(self.config["src_vocab_size"]),
            tgt_vocab_size=int(self.config["tgt_vocab_size"]),
            src_seq_len=self.seq_len,
            tgt_seq_len=self.seq_len,
            d_model=int(self.config.get("d_model", 256)),
            N=int(self.config.get("N", 4)),
            h=int(self.config.get("h", 8)),
            dropout=float(self.config.get("dropout", 0.1)),
            d_ff=int(self.config.get("d_ff", 512)),
        )
        state = torch.load(str(weights_path), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def _encode_source(self, text: str) -> torch.Tensor:
        ids = self.src_tok.encode(text).ids[: self.seq_len - 2]
        ids = [self.sos_id] + ids + [self.eos_id]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _decode_target(self, ids: List[int]) -> str:
        clean = [t for t in ids if t not in (self.sos_id, self.eos_id, self.pad_id)]
        return self.tgt_tok.decode(clean).strip()

    @torch.inference_mode()
    def translate(self, text: str, num_beams: int = 1, max_new_tokens: Optional[int] = None) -> str:
        max_new_tokens = max_new_tokens or self.seq_len
        # decoder sequence (incl. leading SOS) must fit the positional-encoding buffer
        max_new_tokens = min(max_new_tokens, self.seq_len - 1)
        src = self._encode_source(text)
        src_mask = (src != self.pad_id).unsqueeze(1).unsqueeze(1).int()
        encoder_output = self.model.encode(src, src_mask)
        if num_beams <= 1:
            ids = self._greedy(encoder_output, src_mask, max_new_tokens)
        else:
            ids = self._beam(encoder_output, src_mask, num_beams, max_new_tokens)
        return self._decode_target(ids)

    def _causal_mask(self, size: int) -> torch.Tensor:
        return torch.tril(torch.ones(1, size, size, device=self.device)).int()

    def _greedy(self, encoder_output, src_mask, max_new_tokens) -> List[int]:
        ys = torch.tensor([[self.sos_id]], dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens):
            out = self.model.decode(encoder_output, src_mask, ys, self._causal_mask(ys.size(1)))
            logits = self.model.project(out[:, -1])
            next_id = int(logits.argmax(-1).item())
            ys = torch.cat([ys, torch.tensor([[next_id]], device=self.device)], dim=1)
            if next_id == self.eos_id:
                break
        return ys[0].tolist()

    def _beam(self, encoder_output, src_mask, num_beams, max_new_tokens) -> List[int]:
        length_penalty = float(self.config.get("length_penalty", 1.0))
        beams = [([self.sos_id], 0.0)]
        completed: List[tuple] = []
        for _ in range(max_new_tokens):
            candidates: List[tuple] = []
            for seq, score in beams:
                if seq[-1] == self.eos_id:
                    completed.append((seq, score))
                    continue
                ys = torch.tensor([seq], dtype=torch.long, device=self.device)
                out = self.model.decode(encoder_output, src_mask, ys, self._causal_mask(ys.size(1)))
                log_probs = torch.log_softmax(self.model.project(out[:, -1]), dim=-1)[0]
                top_lp, top_ids = log_probs.topk(num_beams)
                for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                    candidates.append((seq + [tid], score + lp))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)
            beams = candidates[:num_beams]
            if all(seq[-1] == self.eos_id for seq, _ in beams):
                completed.extend(beams)
                break
        pool = completed or beams
        best = max(pool, key=lambda x: x[1] / (len(x[0]) ** length_penalty))
        return best[0]


class TranslatorRegistry:
    """Lazily loads per-src-lang models from a root models directory."""

    def __init__(self, models_dir: Path, device: str):
        self.models_dir = Path(models_dir)
        self.device = device
        self._cache: Dict[str, TranslatorModel] = {}
        self._pair_dirs = self._discover()

    def _discover(self) -> Dict[str, Path]:
        pairs: Dict[str, Path] = {}
        if not self.models_dir.exists():
            return pairs
        for child in sorted(self.models_dir.iterdir()):
            config = child / "config.json"
            if child.is_dir() and config.exists():
                try:
                    src_lang = json.loads(config.read_text()).get("src_lang")
                except json.JSONDecodeError:
                    continue
                if src_lang:
                    pairs[src_lang] = child
        return pairs

    @property
    def supported_langs(self) -> List[str]:
        return sorted(self._pair_dirs)

    def available(self) -> bool:
        return bool(self._pair_dirs)

    def get(self, src_lang: str) -> TranslatorModel:
        if src_lang not in self._pair_dirs:
            raise ValueError(f"Unsupported src_lang: {src_lang}")
        if src_lang not in self._cache:
            self._cache[src_lang] = TranslatorModel(self._pair_dirs[src_lang], self.device)
        return self._cache[src_lang]

    def warm(self) -> None:
        for lang in self._pair_dirs:
            self.get(lang)


@lru_cache(maxsize=1)
def build_registry(models_dir: str, device: str) -> TranslatorRegistry:
    return TranslatorRegistry(Path(models_dir), device)
