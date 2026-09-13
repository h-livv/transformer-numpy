# Transformer from Scratch — NumPy

A small decoder-only Transformer language model implemented from scratch in
Python and NumPy, including the forward pass, training objective, and
backpropagation.

The project is built to understand language models through mathematical
derivation rather than framework abstractions. Each component is derived from
the underlying operation and implemented directly using NumPy, with matrix
dimensions and information flow kept explicit.

The model is character-level. `single_head/` is one block and one attention
head. `multihead/` is one block with four heads. `multiblock/` stacks several
multi-head blocks. All three train on a short repeated sentence and can
complete a prefix of that sentence. That is a closed toy task, not general
language modeling.

## Architecture

The high-level pipeline is:

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
Transformer Block
 │
 ├── Causal Self-Attention
 │     ├── Q, K, V projections
 │     ├── Scaled Dot-Product Attention
 │     ├── Causal Masking
 │     └── Output Projection
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
Loss
 │
 ▼
Backpropagation
 │
 ▼
Parameter Updates
```

Training uses random windows of length `context_length`. Generation scores
only the last window so the positional encodings match those seen in training.

## From Scratch

The implementation uses only Python and NumPy for the model and training
machinery.

There are no deep-learning frameworks, automatic-differentiation systems, or
prebuilt neural-network layers. Forward propagation, attention,
backpropagation, gradient computation, and parameter updates are implemented
manually from their underlying mathematics.

The purpose is to make the correspondence between the mathematics and the
implementation explicit.

## Verification

Individual components are checked numerically, including:

- tokenizer round trips
- positional encoding equivalence
- causal masking
- softmax normalization
- attention dimensions
- forward-pass dimensions
- gradient calculations

Typical attention shapes:

```text
X          (d_model, L)
Q          (d_k, L)
K          (d_k, L)
V          (d_v, L)
Scores     (L, L)
Attention  (L, L)
Y          (d_v, L)
Output     (d_model, L)
Final      (d_model, L)
```

Run a variant with:

```text
python single_head/train.py
python multihead/train.py
python multiblock/pretrain.py
```

## Philosophy

This is a learning implementation, not an optimized Transformer library. The
focus is on deriving the computation from first principles and translating
that derivation directly into NumPy.
