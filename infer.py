import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from serve_model import build_registry, select_device

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models")).expanduser().resolve()
MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", "128"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "64"))
DEFAULT_NUM_BEAMS = int(os.getenv("DEFAULT_NUM_BEAMS", "1"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
DEVICE = select_device(os.getenv("FORCE_DEVICE", ""))

# Source languages this service knows how to route, independent of which
# artifacts are present. Used to reject unsupported langs before any model load.
KNOWN_SRC_LANGS = {"ca", "de"}

registry = None
startup_error = None


def normalize_lang(lang: str) -> str:
    lang = (lang or "").strip()
    return {"ca_XX": "ca", "de_DE": "de", "en_XX": "en"}.get(lang, lang)


def _get_registry():
    global registry
    if registry is None:
        registry = build_registry(str(MODELS_DIR), DEVICE)
    return registry


def _validate_src_lang(src_lang: str) -> str:
    src_lang = normalize_lang(src_lang)
    if src_lang not in KNOWN_SRC_LANGS:
        raise ValueError(f"Unsupported src_lang: {src_lang}")
    return src_lang


def translate(text: str, src_lang: str, num_beams: int = DEFAULT_NUM_BEAMS) -> str:
    src_lang = _validate_src_lang(src_lang)
    reg = _get_registry()
    if src_lang not in reg.supported_langs:
        raise RuntimeError(
            f"No trained artifact for '{src_lang}' in {MODELS_DIR}. "
            f"Available: {reg.supported_langs or 'none'}."
        )
    return reg.get(src_lang).translate(text, num_beams=num_beams, max_new_tokens=MAX_NEW_TOKENS)


class TranslateIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    src_lang: str
    num_beams: int = Field(default=DEFAULT_NUM_BEAMS, ge=1, le=8)


class BatchTranslateIn(BaseModel):
    items: List[TranslateIn] = Field(..., min_length=1, max_length=128)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global startup_error
    try:
        reg = _get_registry()
        if not reg.available():
            raise RuntimeError(
                f"No trained artifacts found under {MODELS_DIR}. "
                "Train with train.py or sync the Colab export."
            )
        reg.warm()
        startup_error = None
    except Exception as exc:
        startup_error = str(exc)
    yield


app = FastAPI(title="T-Transformer Translator API", version=MODEL_VERSION, lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, object]:
    reg = _get_registry()
    loaded = reg.available() and startup_error is None
    return {
        "status": "ok" if loaded else "degraded",
        "device": DEVICE,
        "models_dir": str(MODELS_DIR),
        "model_version": MODEL_VERSION,
        "supported_langs": reg.supported_langs,
        "startup_error": startup_error,
    }


@app.post("/translate")
def translate_one(req: TranslateIn) -> Dict[str, object]:
    started = time.perf_counter()
    try:
        output = translate(req.text, req.src_lang, req.num_beams)
        return {
            "translation": output,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/translate/batch")
def translate_batch(req: BatchTranslateIn) -> Dict[str, object]:
    started = time.perf_counter()
    try:
        for item in req.items:
            _validate_src_lang(item.src_lang)
        translations = [
            translate(item.text, item.src_lang, item.num_beams) for item in req.items
        ]
        return {
            "translations": translations,
            "batch_size": len(req.items),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
