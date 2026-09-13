# Transformer from Scratch — NumPy

A decoder-only transformer written completely from scratch in NumPy.
Tokenization, embedding, self-attention, forward-pass, backpropagation,
cross-entropy, all written out.

It is a character-level language model. Three versions:

- `singlehead/` — one block, one attention head
- `multihead/` — one block, multi-head attention
- `multiblock/` — several multi-head blocks, plus pre-norm LayerNorm

The model trains on *The quick brown fox jumps over the lazy dog.* 
<br>
Prompt it with *The* and it finishes the sentence. That is memorization, not understanding.
<br>
Quite literally a stochastic parrot.

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

`multihead/` runs the same attention function on each head’s own Q, K, and
V, concatenates the outputs, then mixes them with one `Wo`.

`multiblock/` stacks this block. Each block is:
<br>
LayerNorm → attention →
residual → LayerNorm → MLP → residual. 
<br>
One more LayerNorm is implemented after the last
block, then a single vocab head. Residuals live in the block.

Training samples random windows of length `context_length`, then generation only
scores the last window so positions match training.

## From Scratch

This was build using Python and NumPy only. No PyTorch, autograd, or `nn.Linear`.

The point is to make the math line up with the code.

## Verification

The scripts check tokenizer round trips, positional encodings, causal
masking, softmax, and dimensions.

Typical attention shapes (one head):

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

Reproduce the pipeline:

```text
python singlehead/pretrain.py
python multihead/pretrain.py
python multiblock/pretrain.py
```

Set `print_probs = True` prints the next-character probability distribution at each step.

## Philosophy

This is a learning implementation, not an optimized Transformer library. The focus is on deriving the computation from first principles and translating that derivation directly into NumPy.