# Transformer from Scratch — NumPy

A decoder-only Transformer implemented **from scratch in NumPy**, including
tokenization, embeddings, causal self-attention, feed-forward layers,
backpropagation, cross-entropy, and parameter updates.

The implementation is deliberately explicit: the mathematical operations map
directly onto the code, with no PyTorch, autograd, or neural-network libraries.

## Models

Three progressively more expressive implementations are included:

* `singlehead/` — one Transformer block, one attention head
* `multihead/` — one block with multiple attention heads
* `multiblock/` — stacked multi-head blocks with pre-norm LayerNorm

The model is a character-level language model trained on:

> *The quick brown fox jumps over the lazy dog.*

At this scale, the model is primarily overfitting the training data
rather than demonstrating meaningful language understanding.

## Architecture

```text
Text
 │
 ▼
Tokenizer
 │
 ▼
Token IDs
 │
 ▼
Token + Positional Embeddings
 │
 ▼
Transformer Block(s)
 │
 ├── Causal Self-Attention
 │     ├── Q, K, V projections
 │     ├── Scaled dot-product attention
 │     ├── Causal masking
 │     └── Output projection
 │
 ├── Residual
 │
 ├── Feed-Forward Network
 │
 └── Residual
 │
 ▼
Language Model Head
 │
 ▼
Logits
 │
 ▼
Cross-Entropy Loss
 │
 ▼
Backpropagation
 │
 ▼
Parameter Updates
```

`multihead/` computes attention independently for each head, concatenates the
results, and applies the output projection.

`multiblock/` stacks pre-norm Transformer blocks:

```text
LayerNorm → Attention → Residual
LayerNorm → MLP       → Residual
```

A final LayerNorm feeds the vocabulary projection.

## Verification

The implementation includes checks for:

* tokenizer round trips
* positional encodings
* causal masking
* softmax
* tensor dimensions

Typical single-head shapes:

```text
X          (d_model, L)
Q          (d_k, L)
K          (d_k, L)
V          (d_v, L)
Scores     (L, L)
Attention  (L, L)
Y          (d_v, L)
Output     (d_model, L)
```

## Running

```bash
python singlehead/pretrain.py
python multihead/pretrain.py
python multiblock/pretrain.py
```

`print_probs = True` inspects the next-character probability
distribution during generation.

## Purpose

This is a **learning implementation**, not an optimized Transformer library.

The goal is to make the computational structure of the model explicit enough
to inspect, modify, and experiment with directly.
