# Multi-Head Clone

This directory is a copy of the single-head model at the repo root, with
causal self-attention made multi-head. The root `src/` and `train.py` are
unchanged.

Each head runs the same single-head attention function on its own Q, K, and V
weights. The head outputs are concatenated along the feature axis, then mixed
by one output matrix `Wo`.

The model is still one Transformer block. It trains on the same short
repeated sentence.

Run from the repo root:

```text
python multihead/train.py
```

or from this directory:

```text
python train.py
```

Set `print_probs = True` in `train.py` to print next-character probabilities
during generation.
