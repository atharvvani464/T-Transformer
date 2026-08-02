import importlib
import os
import sys
from pathlib import Path

import torch
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _reload_infer_with_models_dir(models_dir: Path):
    os.environ["MODELS_DIR"] = str(models_dir)
    os.environ.setdefault("DEFAULT_NUM_BEAMS", "1")

    # serve_model caches registries by (dir, device); clear so each test is isolated.
    if "serve_model" in sys.modules:
        sys.modules["serve_model"].build_registry.cache_clear()
    for name in ("infer", "serve_model"):
        sys.modules.pop(name, None)

    return importlib.import_module("infer")


def test_health_degraded_without_model(tmp_path: Path):
    module = _reload_infer_with_models_dir(tmp_path / "empty_models")
    with TestClient(module.app) as client:
        response = client.get("/health")
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "degraded"
        assert "No trained artifacts" in (body.get("startup_error") or "")
        assert body["supported_langs"] == []


def test_translate_rejects_unsupported_lang_before_model_load(tmp_path: Path):
    module = _reload_infer_with_models_dir(tmp_path / "empty_models")
    with TestClient(module.app) as client:
        response = client.post(
            "/translate",
            json={"text": "bonjour", "src_lang": "fr", "num_beams": 1},
        )
        assert response.status_code == 400
        assert "Unsupported src_lang" in response.json()["detail"]


def test_known_lang_without_artifact_returns_503(tmp_path: Path):
    module = _reload_infer_with_models_dir(tmp_path / "empty_models")
    with TestClient(module.app) as client:
        response = client.post(
            "/translate",
            json={"text": "Das ist ein Test.", "src_lang": "de", "num_beams": 1},
        )
        assert response.status_code == 503
        assert "No trained artifact" in response.json()["detail"]


def _build_tiny_artifact(models_dir: Path) -> None:
    """Create a minimal ca_en artifact so the serve path can be exercised offline."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer

    import json
    from model import build_transformer

    specials = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
    src_texts = ["hola mon", "el gat menja", "bon dia amic"]
    tgt_texts = ["hello world", "the cat eats", "good day friend"]

    def train_tok(texts):
        tok = Tokenizer(WordLevel(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        tok.train_from_iterator(texts, WordLevelTrainer(special_tokens=specials, min_frequency=1))
        return tok

    src_tok, tgt_tok = train_tok(src_texts), train_tok(tgt_texts)
    out = models_dir / "ca_en"
    out.mkdir(parents=True, exist_ok=True)

    model = build_transformer(src_tok.get_vocab_size(), tgt_tok.get_vocab_size(),
                              16, 16, d_model=32, N=2, h=2, dropout=0.0, d_ff=64)
    torch.save(model.state_dict(), out / "weights.pth")
    src_tok.save(str(out / "src_tokenizer.json"))
    tgt_tok.save(str(out / "tgt_tokenizer.json"))
    (out / "config.json").write_text(json.dumps({
        "src_lang": "ca", "tgt_lang": "en",
        "d_model": 32, "N": 2, "h": 2, "d_ff": 64, "dropout": 0.0, "seq_len": 16,
        "src_vocab_size": src_tok.get_vocab_size(), "tgt_vocab_size": tgt_tok.get_vocab_size(),
        "pad_id": 0, "sos_id": 2, "eos_id": 3,
        "weights": "weights.pth",
        "src_tokenizer": "src_tokenizer.json", "tgt_tokenizer": "tgt_tokenizer.json",
    }))


def test_serve_path_with_tiny_artifact(tmp_path: Path):
    models_dir = tmp_path / "models"
    _build_tiny_artifact(models_dir)

    module = _reload_infer_with_models_dir(models_dir)
    with TestClient(module.app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert "ca" in health["supported_langs"]

        response = client.post(
            "/translate",
            json={"text": "hola mon", "src_lang": "ca", "num_beams": 2},
        )
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body.get("translation"), str)
        assert float(body["latency_ms"]) >= 0

        batch = client.post(
            "/translate/batch",
            json={"items": [
                {"text": "hola mon", "src_lang": "ca", "num_beams": 1},
                {"text": "bon dia amic", "src_lang": "ca", "num_beams": 1},
            ]},
        )
        assert batch.status_code == 200
        assert batch.json()["batch_size"] == 2


def test_model_smoke_if_real_artifact_exists():
    artifact_dir = ROOT / "models" / "ca_en"
    if not (artifact_dir / "config.json").exists():
        import pytest

        pytest.skip("Skipping: models/ca_en artifact not present locally")

    module = _reload_infer_with_models_dir(ROOT / "models")
    with TestClient(module.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post(
            "/translate",
            json={"text": "El perill era desesperat.", "src_lang": "ca", "num_beams": 1},
        )
        assert response.status_code == 200
        translation = response.json()["translation"].strip()
        assert len(translation) > 0
        # non-degenerate: not a single token repeated forever
        words = translation.split()
        assert len(set(words)) > 1 or len(words) <= 1
