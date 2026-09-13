import numpy as np

from src.tokenizer import Tokenizer
from src.embedder import Embedder
from src.attention import Attention
from src.mlp import MLP
from src.backpropagation import Backpropagation

np.random.seed(0)

# Closed toy corpus — small enough that this 1-block model can finish a sentence.
train_text = "The quick brown fox jumps over the lazy dog. " * 20

tokenizer = Tokenizer(train_text)
tokenizer.build_vocab()

vocab_size = tokenizer.vocab_size
embed_dim = 16
reduced_dim = 4
hidden_dim = 64
print_probs = True

#Initialize layers once so weights are reused across steps.
embedder = Embedder(embed_dim=embed_dim, vocab_size=vocab_size)
attention = Attention(embed_dim, reduced_dim)
mlp = MLP(vocab_size, embed_dim, hidden_dim)
backprop = Backpropagation()


def forward(token_ids):
    embedded_matrix = embedder.embedding(token_ids)
    X_prime = attention.forward(embedded_matrix)
    probabilities = mlp.forward(X_prime)
    return mlp.Z, probabilities


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
    attention.step(lr)
    mlp.step(lr)


#The prompt.
prompt = "The quick brown fox jumps "

#Encode the prompt to get the token IDs.
token_ids = tokenizer.encode(prompt)

#Decode for verification
reconstructed = tokenizer.decode(token_ids)

#One forward pass for shape checks and next-token wiring.
logits, probabilities = forward(token_ids)
predicted_id = mlp.next_token_id()
predicted_char = tokenizer.decode([predicted_id])

#Verifications

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

print("--- ATTENTION OUTPUT ---")
attention.verification()
print("")

print("--- MLP OUTPUT ---")
mlp.verification()

print("--- NEXT TOKEN (untrained) ---")
print("Prompt:          ", repr(prompt))
print("Predicted id:    ", predicted_id)
print("Predicted char:  ", repr(predicted_char))
print("P(predicted):    ", float(mlp.next_token_probs()[predicted_id]))
print("")

train_ids = tokenizer.encode(train_text)
context_length = 48
lr = 0.05
steps = 3000

print("--- TRAINING ---")
for step in range(steps):
    start = np.random.randint(0, len(train_ids) - context_length)
    batch = train_ids[start : start + context_length]
    _, probs = forward(batch)
    loss = backprop.loss_function(batch, probs)
    dZ = backprop.dl_dz()
    dX_prime = mlp.backward(dZ)
    dX = attention.backward(dX_prime)
    embedder.backward(dX)
    sgd_step(lr)

print("Training successful.")
print(f"Final loss: {loss:.4f}")
print("")

print("--- GENERATION ---")
eval_prompt = "The quick brown fox jumps "
print(f"Prompt: {eval_prompt!r}")
print("")
generated = generate(eval_prompt, 18)
print(f"Generated: {generated!r}")
