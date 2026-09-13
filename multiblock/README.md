# Multi-Block Clone

This directory stacks several Transformer blocks, each with multi-head
attention. `single_head/` and `multihead/` are left unchanged.

A block is pre-norm: LayerNorm, then attention, residual, LayerNorm, then
the feed-forward network, residual. A final LayerNorm sits after the last
block, then the language-model head once. Backprop runs backward through
that chain.

`single_head/` and `multihead/` have no LayerNorm.

The entrypoint is `pretrain.py`, not `train.py`. The script fits the
next-character model and then samples from it; pretraining is the usual name
for that stage.

Run from the repo root:

```text
python multiblock/pretrain.py
```
