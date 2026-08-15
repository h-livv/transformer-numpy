# Transformer from Scratch — NumPy

A from-scratch Transformer implementation in Python and NumPy, built to understand the architecture through mathematical derivation rather than abstraction.

Each component is derived from the operation it needs to perform, with matrix dimensions and information flow kept explicit.

## Architecture

The pipeline is:

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
Token Embeddings
 │
 ▼
Positional Encoding
 │
 ▼
Q, K, V projections
 │
 ▼
Scaled Dot-Product Attention
 │
 ▼
Causal Masking
 │
 ▼
Softmax Attention Weights
 │
 ▼
Weighted Value Aggregation
 │
 ▼
Output Projection
 │
 ▼
Residual Connection
```

## Verification

Each stage is checked numerically, including tokenizer round trips, positional encoding equivalence, causal masking, softmax normalization, and attention dimensions.

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

## Philosophy

This is a learning implementation, not an optimized Transformer library. The focus is on deriving the computation from first principles and translating that derivation directly into NumPy.
