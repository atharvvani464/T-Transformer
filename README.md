# T-Transformer — a from-scratch transformer translation service

A custom implementation of the Transformer ("Attention Is All You Need") trained as a
multilingual translator (Catalan→English, German→English) and served over FastAPI. The
architecture in [`model.py`](model.py) is built from first principles — embeddings,
sinusoidal positional encoding, multi-head attention, encoder/decoder stacks — and the same
module is used for both training and inference.

> **New to the project?** [`ARCHITECTURE.md`](ARCHITECTURE.md) explains the design, its
> significance, and how the transformer actually makes translation possible, step by step.

```
model.py        # the transformer architecture (build_transformer)
train.py        # trains one model per language pair on opus_books
serve_model.py  # loads a trained artifact, greedy + beam-search decoding
infer.py        # FastAPI service (/health, /translate, /translate/batch)
01_train.ipynb  # Colab (GPU) training notebook
tests/          # API + serving tests
```

## Architecture

- `d_model=256`, `N=4` encoder/decoder layers, `h=8` heads, `d_ff=512`, `seq_len=64`
- One self-contained model **per language pair**, each with its own WordLevel tokenizers.

Trained artifacts live under `models/<src>_<tgt>/`:

```
models/ca_en/
    weights.pth         # state_dict for build_transformer(...)
    config.json         # architecture + special-token ids + tokenizer filenames
    src_tokenizer.json  # source-language tokenizer
    tgt_tokenizer.json  # English tokenizer
```

## Train

Training pulls `opus_books` and builds tokenizers automatically. Run on a GPU for real
quality; use `QUICK_RUN=1` for a fast CPU/MPS smoke test.

```bash
pip install -r requirements.txt

# full run, both pairs
python train.py

# single pair
python train.py --pairs ca-en

# quick smoke (few samples, tiny)
QUICK_RUN=1 QUICK_SAMPLES=1500 EPOCHS=12 python train.py --pairs ca-en
```

Key env knobs: `EPOCHS`, `BATCH_SIZE`, `LR`, `D_MODEL`, `N_LAYERS`, `N_HEADS`, `D_FF`,
`SEQ_LEN`, `VOCAB_SIZE`, `WARMUP_STEPS`. BLEU/chrF on a held-out test split are written into
each pair's `config.json`.

### Training on Colab

Open [`01_train.ipynb`](01_train.ipynb) in Colab with a GPU runtime. It mounts Drive,
installs deps, runs `train.py` for both pairs on the T4, writes versioned snapshots under
`models/<pair>/versions/<timestamp>` with a `LATEST` pointer + `metrics.json`, and produces a
downloadable `models_artifact.zip`. Unzip it into local `models/`.

## Serve

```bash
MODELS_DIR=models DEFAULT_NUM_BEAMS=1 uvicorn infer:app --host 0.0.0.0 --port 8000
```

The service auto-discovers every `models/<pair>/` with a `config.json`. With no artifacts
present it starts in a **degraded** state (health reports it) rather than crashing.

### Endpoints

- `GET /health` → status, device, `supported_langs`, model version
- `POST /translate` → `{ "text", "src_lang", "num_beams" }` → `{ "translation", "latency_ms" }`
- `POST /translate/batch` → `{ "items": [ ... ] }`

`src_lang` accepts `ca`/`de` (also the `ca_XX`/`de_DE` aliases).

```bash
curl -X POST http://127.0.0.1:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"El perill era desesperat.","src_lang":"ca","num_beams":1}'
```

## Docker

The image installs only the serving deps (`requirements-serve.txt`) and runs on CPU.

```bash
docker build -t t-transformer:latest .
docker run --rm -p 8000:8000 t-transformer:latest
```

Bake artifacts into the image by having `models/<pair>/` present at build time, or mount at
runtime: `docker run -p 8000:8000 -v "$PWD/models:/app/models" t-transformer:latest`.

## Test

```bash
pytest -q
```

API-validation, degraded-health, and a tiny end-to-end serving test always run. The
real-artifact smoke test runs automatically when `models/ca_en/` exists locally.

## Latency notes

- `DEFAULT_NUM_BEAMS=1` for the low-latency path; raise `num_beams` per request for quality.
- Pin `MODELS_DIR` to local SSD; set `FORCE_DEVICE=cuda|mps|cpu` to override device selection.

## Roadmap

- **Next PR:** deploy a live public endpoint (e.g. Hugging Face Spaces / Render / Fly).
