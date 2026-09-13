import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tokenizer import Tokenizer
from src.embedder import Embedder
from src.block import Block
from src.head import LanguageModelHead
from src.layernorm import LayerNorm
from src.backpropagation import Backpropagation

np.random.seed(0)

train_text = "The quick brown fox jumps over the lazy dog. " * 20

tokenizer = Tokenizer(train_text)
tokenizer.build_vocab()

vocab_size = tokenizer.vocab_size
embed_dim = 16
reduced_dim = 4
n_heads = 4
n_blocks = 5
hidden_dim = 64
print_probs = False

embedder = Embedder(embed_dim=embed_dim, vocab_size=vocab_size)
blocks = [
    Block(embed_dim, reduced_dim, n_heads, hidden_dim)
    for _ in range(n_blocks)
]
lm_head = LanguageModelHead(vocab_size, embed_dim)
ln_f = LayerNorm(embed_dim)
backprop = Backpropagation()


def forward(token_ids):
    X = embedder.embedding(token_ids)
    for block in blocks:
        X = block.forward(X)
    X = ln_f.forward(X)
    probabilities = lm_head.forward(X)
    return lm_head.Z, probabilities


def print_next_token_distribution(prefix, probs, selected_id):
    selected_char = tokenizer.decode([selected_id])
    print(f"After {prefix!r}")
    print(f"Selected: {selected_char!r}  (p={float(probs[selected_id]):.6f})")
    print("Next-token probabilities:")
    shown = 0
    for idx in np.argsort(probs)[::-1]:
        p = float(probs[idx])
        if p < 1e-6 and idx != selected_id:
            continue
        char = tokenizer.decode([int(idx)])
        mark = "  <- selected" if idx == selected_id else ""
        print(f"  {char!r:8}  {p:.6f}{mark}")
        shown += 1
    hidden = len(probs) - shown
    if hidden > 0:
        print(f"  (the remaining {hidden} tokens have zero probability)")
    print("")


def generate(prompt, max_new_tokens):
    ids = tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        window = ids[-context_length:]
        _, probs = forward(window)
        next_probs = probs[:, -1]
        next_id = int(np.argmax(next_probs))
        if print_probs:
            print_next_token_distribution(tokenizer.decode(ids), next_probs, next_id)
        ids = np.append(ids, next_id)
    return tokenizer.decode(ids)


def sgd_step(lr):
    embedder.step(lr)
    for block in blocks:
        block.step(lr)
    ln_f.step(lr)
    lm_head.step(lr)


prompt = "The "

token_ids = tokenizer.encode(prompt)
reconstructed = tokenizer.decode(token_ids)

logits, probabilities = forward(token_ids)
predicted_id = lm_head.next_token_id()
predicted_char = tokenizer.decode([predicted_id])

print("--- TOKENIZATION OUTPUT ---")
print("Original:     ", repr(prompt))
print("Token IDs:    ", token_ids)
print("Reconstructed:", repr(reconstructed))
print("")
assert reconstructed == prompt

print("--- EMBEDDER OUTPUT ---")
embed_shape = embedder.verification()
print("Embedded Matrix Shape: ", embed_shape)
print(f"Interpretation: {embed_shape[0]} rows of vector math across {embed_shape[1]} token columns.")
print("")

print("--- POSITIONAL EMBEDDER OUTPUT ---")
embedder.verify_pos(sequence_length=len(token_ids))
print("")

for i, block in enumerate(blocks):
    block.verification(i)

print("--- FINAL LAYERNORM ---")
ln_f.verification()
print("")

print("--- LANGUAGE MODEL HEAD ---")
lm_head.verification()

print("--- NEXT TOKEN (untrained) ---")
print("Prompt:          ", repr(prompt))
print("Predicted id:    ", predicted_id)
print("Predicted char:  ", repr(predicted_char))
print("P(predicted):    ", float(lm_head.next_token_probs()[predicted_id]))
print("")

train_ids = tokenizer.encode(train_text)
context_length = 48
lr = 0.05
steps = 8000

print("--- TRAINING ---")
for step in range(steps):
    start = np.random.randint(0, len(train_ids) - context_length)
    batch = train_ids[start : start + context_length]
    _, probs = forward(batch)
    loss = backprop.loss_function(batch, probs)
    dZ = backprop.dl_dz()
    dX = lm_head.backward(dZ)
    dX = ln_f.backward(dX)
    for block in reversed(blocks):
        dX = block.backward(dX)
    embedder.backward(dX)
    sgd_step(lr)

print("Training successful.")
print(f"Final loss: {loss:.4f}")
print("")

print("--- GENERATION ---")
eval_prompt = "The "
print(f"Prompt: {eval_prompt!r}")
print("")
generated = generate(eval_prompt, 100)
print(f"Generated: {generated!r}")
