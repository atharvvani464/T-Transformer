"""Train the from-scratch transformer (model.py) as a seq2seq translator.

One model per language pair (e.g. ca-en, de-en) on the opus_books corpus.
Artifacts are written to models/<src>_<tgt>/ ready to be served by serve_model.py:

    models/ca_en/{weights.pth, config.json, src_tokenizer.json, tgt_tokenizer.json}

Usage:
    python train.py                       # all default pairs, full run
    QUICK_RUN=1 python train.py           # tiny smoke run (few samples / 1 epoch)
    python train.py --pairs ca-en         # a single pair
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, Dataset

from model import build_transformer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.manual_seed(42)

SPECIALS = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

# ---- config ----
QUICK_RUN = os.getenv("QUICK_RUN", "0") == "1"
SEQ_LEN = int(os.getenv("SEQ_LEN", "64"))
D_MODEL = int(os.getenv("D_MODEL", "256"))
N_LAYERS = int(os.getenv("N_LAYERS", "4"))
N_HEADS = int(os.getenv("N_HEADS", "8"))
D_FF = int(os.getenv("D_FF", "512"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "16000"))
MIN_FREQ = int(os.getenv("MIN_FREQ", "2"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
LR = float(os.getenv("LR", "1e-3"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "2000"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.1"))
MAX_GRAD_NORM = 1.0

if QUICK_RUN:
    BATCH_SIZE = min(BATCH_SIZE, 16)

DEFAULT_PAIRS = ["ca-en", "de-en"]
OPUS_KEY = {"ca": "ca", "de": "de", "en": "en"}

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def build_tokenizer(texts: List[str]) -> Tokenizer:
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(vocab_size=VOCAB_SIZE, min_frequency=MIN_FREQ, special_tokens=SPECIALS)
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def encode(tok: Tokenizer, text: str) -> List[int]:
    ids = tok.encode(text).ids[: SEQ_LEN - 2]
    return [SOS_ID] + ids + [EOS_ID]


def pad_to_seq(ids: List[int]) -> List[int]:
    ids = ids[:SEQ_LEN]
    return ids + [PAD_ID] * (SEQ_LEN - len(ids))


class PairDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], src_tok: Tokenizer, tgt_tok: Tokenizer):
        self.rows = rows
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        src = pad_to_seq(encode(self.src_tok, row["src"]))
        tgt_full = encode(self.tgt_tok, row["tgt"])[:SEQ_LEN]
        decoder_input = pad_to_seq(tgt_full[:-1])
        label = pad_to_seq(tgt_full[1:])
        return {
            "encoder_input": torch.tensor(src, dtype=torch.long),
            "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def causal_mask(size: int) -> torch.Tensor:
    return torch.tril(torch.ones(1, size, size)).int()


def collate(batch):
    enc = torch.stack([b["encoder_input"] for b in batch])
    dec = torch.stack([b["decoder_input"] for b in batch])
    label = torch.stack([b["label"] for b in batch])
    src_mask = (enc != PAD_ID).unsqueeze(1).unsqueeze(1).int()
    tgt_pad = (dec != PAD_ID).unsqueeze(1).unsqueeze(1).int()
    tgt_mask = tgt_pad & causal_mask(dec.size(1))
    return enc, dec, label, src_mask, tgt_mask


def load_pair_rows(pair: str) -> List[Dict[str, str]]:
    src_lang, tgt_lang = pair.split("-")
    ds = load_dataset("opus_books", pair)["train"]
    rows = []
    for ex in ds:
        s = ex["translation"].get(OPUS_KEY[src_lang], "")
        t = ex["translation"].get(OPUS_KEY[tgt_lang], "")
        if s and t:
            rows.append({"src": s, "tgt": t})
    return rows


def split_rows(rows: List[Dict[str, str]]) -> Tuple[list, list, list]:
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(rows), generator=g).tolist()
    rows = [rows[i] for i in perm]
    n = len(rows)
    n_test = max(1, int(n * 0.05))
    n_val = max(1, int(n * 0.05))
    return rows[n_test + n_val:], rows[:n_val], rows[n_val:n_val + n_test]


def make_lr_lambda(warmup: int):
    def lr_lambda(step: int) -> float:
        step = max(1, step)
        return min(step / warmup, (warmup / step) ** 0.5)
    return lr_lambda


@torch.inference_mode()
def greedy_translate(model, src_tok, tgt_tok, text: str) -> str:
    model.eval()
    ids = torch.tensor([pad_to_seq(encode(src_tok, text))], device=DEVICE)
    src_mask = (ids != PAD_ID).unsqueeze(1).unsqueeze(1).int()
    enc = model.encode(ids, src_mask)
    ys = torch.tensor([[SOS_ID]], device=DEVICE)
    for _ in range(SEQ_LEN):
        out = model.decode(enc, src_mask, ys, causal_mask(ys.size(1)).to(DEVICE))
        nxt = int(model.project(out[:, -1]).argmax(-1).item())
        ys = torch.cat([ys, torch.tensor([[nxt]], device=DEVICE)], dim=1)
        if nxt == EOS_ID:
            break
    toks = [t for t in ys[0].tolist() if t not in (SOS_ID, EOS_ID, PAD_ID)]
    return tgt_tok.decode(toks).strip()


def evaluate_bleu(model, rows, src_tok, tgt_tok, limit: int) -> Dict[str, float]:
    try:
        import evaluate
        bleu = evaluate.load("sacrebleu")
        chrf = evaluate.load("chrf")
    except Exception as exc:  # metrics optional; don't fail training
        print(f"[metrics] skipped ({exc})")
        return {}
    preds, refs = [], []
    for row in rows[:limit]:
        preds.append(greedy_translate(model, src_tok, tgt_tok, row["src"]))
        refs.append([row["tgt"]])
    scores = {
        "bleu": round(bleu.compute(predictions=preds, references=refs)["score"], 2),
        "chrf": round(chrf.compute(predictions=preds, references=[r[0] for r in refs])["score"], 2),
    }
    return scores


def train_pair(pair: str, out_root: Path) -> Dict[str, object]:
    src_lang, tgt_lang = pair.split("-")
    print(f"\n=== Training {pair} on {DEVICE} (QUICK_RUN={QUICK_RUN}) ===")

    rows = load_pair_rows(pair)
    if QUICK_RUN:
        rows = rows[: int(os.getenv("QUICK_SAMPLES", "1500"))]
    train_rows, val_rows, test_rows = split_rows(rows)
    print(f"rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    src_tok = build_tokenizer([r["src"] for r in train_rows])
    tgt_tok = build_tokenizer([r["tgt"] for r in train_rows])
    src_vocab, tgt_vocab = src_tok.get_vocab_size(), tgt_tok.get_vocab_size()
    print(f"vocab: src={src_vocab} tgt={tgt_vocab}")

    loader = DataLoader(
        PairDataset(train_rows, src_tok, tgt_tok),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,
        num_workers=0, drop_last=False,
    )

    model = build_transformer(src_vocab, tgt_vocab, SEQ_LEN, SEQ_LEN,
                              d_model=D_MODEL, N=N_LAYERS, h=N_HEADS,
                              dropout=DROPOUT, d_ff=D_FF).to(DEVICE)
    steps_per_epoch = max(1, math.ceil(len(train_rows) / BATCH_SIZE))
    total_steps = EPOCHS * steps_per_epoch
    warmup = max(1, min(WARMUP_STEPS, total_steps // 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(warmup))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)

    step = 0
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for enc, dec, label, src_mask, tgt_mask in loader:
            enc, dec, label = enc.to(DEVICE), dec.to(DEVICE), label.to(DEVICE)
            src_mask, tgt_mask = src_mask.to(DEVICE), tgt_mask.to(DEVICE)

            encoder_output = model.encode(enc, src_mask)
            decoder_output = model.decode(encoder_output, src_mask, dec, tgt_mask)
            logits = model.project(decoder_output)
            loss = loss_fn(logits.view(-1, tgt_vocab), label.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            step += 1
            running += loss.item()
        avg = running / max(1, len(loader))
        print(f"epoch {epoch + 1}/{EPOCHS}  loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    out_dir = out_root / f"{src_lang}_{tgt_lang}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "weights.pth")
    src_tok.save(str(out_dir / "src_tokenizer.json"))
    tgt_tok.save(str(out_dir / "tgt_tokenizer.json"))

    eval_limit = 50 if QUICK_RUN else 500
    metrics = evaluate_bleu(model, test_rows, src_tok, tgt_tok, eval_limit)

    config = {
        "src_lang": src_lang, "tgt_lang": tgt_lang,
        "d_model": D_MODEL, "N": N_LAYERS, "h": N_HEADS, "d_ff": D_FF, "dropout": DROPOUT,
        "seq_len": SEQ_LEN, "src_vocab_size": src_vocab, "tgt_vocab_size": tgt_vocab,
        "pad_id": PAD_ID, "sos_id": SOS_ID, "eos_id": EOS_ID,
        "weights": "weights.pth",
        "src_tokenizer": "src_tokenizer.json", "tgt_tokenizer": "tgt_tokenizer.json",
        "metrics": metrics,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"saved -> {out_dir}  metrics={metrics}")

    sample = test_rows[0]["src"] if test_rows else "El teu text aquí"
    print(f"sample {src_lang}->{tgt_lang}: {sample!r} => {greedy_translate(model, src_tok, tgt_tok, sample)!r}")
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--out", default=os.getenv("MODELS_DIR", "models"))
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for pair in args.pairs:
        train_pair(pair, out_root)


if __name__ == "__main__":
    main()
