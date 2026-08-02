# T-Transformer — Architecture, Significance, and Impact

This document explains **what** this project is, **why** it matters, and **how** a transformer
turns a sentence in one language into a sentence in another. Every mechanism described here is
implemented from first principles in [`model.py`](model.py) — no `nn.Transformer`, no
pretrained backbone. The same module trains the weights ([`train.py`](train.py)) and serves
them ([`serve_model.py`](serve_model.py), [`infer.py`](infer.py)).

---

## 1. What this project is

A **custom implementation of the Transformer** ("Attention Is All You Need", Vaswani et al.,
2017) trained as a neural machine translation (NMT) system and deployed as a low-latency HTTP
service. It translates **Catalan → English** and **German → English**, with one self-contained
model per language pair.

- **Architecture** — hand-written encoder–decoder transformer: token embeddings, sinusoidal
  positional encoding, scaled dot-product multi-head attention, position-wise feed-forward
  networks, residual connections, and layer normalization.
- **Training** — supervised sequence-to-sequence learning on the `opus_books` parallel corpus,
  with teacher forcing, label smoothing, a warmup→decay learning-rate schedule, and
  gradient clipping.
- **Serving** — a FastAPI service exposing `/translate`, `/translate/batch`, and `/health`,
  with greedy and beam-search decoding, runnable locally and as a Docker container.

Model shape: `d_model=256`, `N=4` encoder & decoder layers, `h=8` attention heads, `d_ff=512`,
max sequence length `64`.

---

## 2. Why it matters (significance)

**It demystifies the architecture behind modern AI.** The transformer is the foundation of
essentially every large language model in use today. Translation is the original task the
transformer was designed for, which makes it the cleanest possible setting to show the
mechanism end to end. Building it from scratch — rather than calling a library — demonstrates a
*mechanistic* understanding of attention, not just API familiarity.

**It is a complete system, not a notebook.** The same architecture is trained on GPU (Colab
T4), exported as a versioned artifact, loaded by a production-shaped inference service,
covered by tests, and containerized. That spans the full lifecycle a real ML system requires:
data → training → evaluation → artifact management → serving → deployment.

**It is honest about scope.** Trained on a small literary corpus with a small model, the
translations are coherent but not state-of-the-art — and the project *measures* that with
BLEU/chrF rather than claiming otherwise. The value is the demonstrated command of the
architecture and the delivery pipeline.

---

## 3. Impact — what it demonstrates

| Capability | Where it shows up |
|---|---|
| Transformer internals from first principles | [`model.py`](model.py) — attention, positional encoding, encoder/decoder stacks |
| Sequence-to-sequence training | [`train.py`](train.py) — teacher forcing, masking, label smoothing, LR schedule |
| Autoregressive decoding | [`serve_model.py`](serve_model.py) — greedy + beam search with length penalty |
| API & service design | [`infer.py`](infer.py) — validation, degraded-health, latency reporting, batching |
| Reproducible training on cloud GPU | [`01_train.ipynb`](01_train.ipynb) — Colab T4, versioned artifacts |
| Testing & packaging | [`tests/`](tests), [`Dockerfile`](Dockerfile), CI |

---

## 4. How the transformer makes translation possible

Translation is a **sequence-to-sequence** problem: map a variable-length source sentence to a
variable-length target sentence. The transformer solves it with an **encoder** that builds a
context-aware representation of the source, and a **decoder** that generates the target one
token at a time while attending back to that representation. Here is the full path a sentence
takes, in the order the data actually flows.

### 4.1 Tokenization — words become integers

A WordLevel tokenizer maps each word to an integer id. Two special tokens frame every
sequence: `[SOS]` (start) and `[EOS]` (end); `[PAD]` fills batches to equal length. Source and
target each have their own tokenizer/vocabulary because the two languages share few tokens.

```
"El perill era desesperat."  →  [SOS] el perill era desesperat . [EOS]
```

### 4.2 Input embeddings — integers become vectors

`InputEmbeddings` looks each id up in a learned `(vocab_size × d_model)` matrix, producing a
`d_model=256`-dimensional vector per token. These vectors are *learned* so that words used in
similar ways end up near each other in space. (The embeddings are scaled by `√d_model` to keep
their magnitude comparable to the positional signal added next.)

### 4.3 Positional encoding — restoring word order

Attention, by itself, is **order-agnostic**: it treats the input as a *set*, so "dog bites man"
and "man bites dog" would look identical. To fix this, `PositionalEncoding` adds a fixed
sinusoidal signal — a unique pattern of sines and cosines at different frequencies — to each
position. The model can now tell position 1 from position 5, and can generalize to relative
distances between words.

```
token vector[pos] = embedding[pos] + positional_signal[pos]
```

### 4.4 Self-attention — every word looks at every other word

This is the core idea. For each token, the model asks: *which other tokens in this sentence
should I pay attention to in order to understand myself?* In "the animal didn't cross the street
because **it** was tired," resolving "it" requires looking back at "animal." Attention learns to
do exactly that.

Mechanically, each token's vector is projected into three roles via learned matrices
`W_q, W_k, W_v`:

- **Query (Q)** — what this token is looking for
- **Key (K)** — what each token offers as a match
- **Value (V)** — the information each token will contribute if attended to

The attention weight from one token to another is the dot product of its query with the other's
key, scaled and softmaxed into a probability distribution; the output is the weighted sum of
values:

```
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d_k ) · V
```

The `√d_k` scaling keeps the dot products from growing large enough to saturate the softmax.
The result: every token's representation is rewritten as a blend of the tokens it found
relevant — it is now **contextualized**.

### 4.5 Multi-head attention — several relationships at once

One attention computation captures one kind of relationship. Language has many simultaneously
(subject–verb agreement, pronoun reference, adjective–noun, long-range clauses). **Multi-head
attention** (`h=8`) runs eight attention computations in parallel, each in its own subspace
(`d_k = d_model / h = 32` dims), then concatenates and mixes them through `W_o`. Each head is
free to specialize in a different linguistic pattern.

### 4.6 Feed-forward, residuals, and normalization — the rest of a block

Each attention sub-layer is followed by a position-wise **feed-forward network**
(`Linear → ReLU → Linear`, widening to `d_ff=512` and back) that transforms each token
independently, adding non-linear representational power.

Two mechanisms keep a deep stack trainable:

- **Residual connections** — each sub-layer's input is added to its output
  (`x + Sublayer(x)`), giving gradients a short path and letting layers refine rather than
  replace the representation.
- **Layer normalization** — re-centers and re-scales each token vector, stabilizing training.

> Implementation note: this project fixes a real bug where `LayerNormalization` received the
> feature count as its `eps` argument, collapsing normalization and preventing the model from
> learning. Correcting it to per-feature scale/bias parameters is what makes training converge.

An **encoder block** = self-attention + feed-forward, each wrapped in residual + norm. The
**encoder** stacks `N=4` of these. Its output is a sequence of vectors, one per source token,
each aware of the entire source sentence.

### 4.7 The decoder — generating the translation

The decoder produces the English sentence one token at a time. Each decoder block has **three**
sub-layers:

1. **Masked self-attention** — attends to the target tokens generated *so far*. A causal mask
   hides future positions, because at generation time the model cannot see words it hasn't
   produced yet. This preserves the autoregressive property.
2. **Cross-attention** — the bridge between languages. Here the **queries come from the target**
   (what the decoder is trying to say next) while the **keys and values come from the encoder
   output** (the source sentence). This is the step where the model looks at the Catalan/German
   sentence to decide the next English word — the literal act of translation.
3. **Feed-forward** — same per-token transformation as in the encoder.

The decoder also stacks `N=4` blocks.

### 4.8 Projection and decoding — vectors become words

A final `ProjectionLayer` (`Linear` to `tgt_vocab_size`) turns each decoder output vector into
a score over the entire target vocabulary; softmax makes it a probability distribution over the
next word. Generation is then autoregressive:

- **Greedy decoding** — take the highest-probability token, append it, feed the sequence back
  in, repeat until `[EOS]`. Fast; the low-latency default.
- **Beam search** — keep the `k` most promising partial translations at each step and expand
  them, scoring by total log-probability with a length penalty. Slower but higher quality;
  selectable per request via `num_beams`.

```
[SOS] → "The"  → "The danger" → "The danger was" → "The danger was desperate" → [EOS]
```

Each step consumes the whole source (via cross-attention) and the whole partial target (via
masked self-attention), so every generated word is conditioned on both the full input and
everything already written.

### 4.9 How it learns (training)

During training the model is shown source→target pairs and uses **teacher forcing**: the decoder
is fed the *correct* target prefix and asked to predict the next token at every position at once.
The loss is cross-entropy between predicted and actual next tokens, with:

- **padding ignored** so the model isn't graded on filler,
- **label smoothing** so it doesn't become overconfident,
- a **warmup→inverse-sqrt learning-rate schedule** and **gradient clipping** for stable
  optimization.

Over many epochs, the embedding matrices, attention projections, and feed-forward weights all
adjust so that the cross-attention learns to align target words with the right source words —
which is precisely what makes translation emerge.

---

## 5. From architecture to service

The trained weights for a pair are saved as `models/<pair>/weights.pth` alongside a
`config.json` (architecture + special-token ids) and the two tokenizers. `serve_model.py`
rebuilds the exact architecture from the config, loads the weights, and runs the decoding loop;
`infer.py` wraps it in a FastAPI app that routes each request to the right per-language model,
validates input, reports per-request latency, and degrades gracefully when no artifact is
present. The result is a direct line from *the mathematics of attention* to *a running
translation endpoint*.

## 6. Further reading

- Vaswani et al., "Attention Is All You Need" (2017) — the original architecture.
- [`model.py`](model.py) — the implementation each section above maps onto.
- [`README.md`](README.md) — how to train, serve, test, and containerize.
