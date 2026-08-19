from src.tokenizer import Tokenizer
from src.embedder import Embedder
from src.attention import Attention
from src.mlp import MLP

#Raw training data.
text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!? "

#Build the vocabulary.
tokenizer = Tokenizer(text)
tokenizer.build_vocab()

vocab_size = tokenizer.vocab_size
embed_dim = 16
reduced_dim = 4

#Initialize the embedder.
embedder = Embedder(embed_dim=embed_dim, vocab_size=vocab_size)

#The prompt.
prompt = "Hello!"

#Encode the prompt to get the token IDs.
token_ids = tokenizer.encode(prompt)

#Decode for verification
reconstructed = tokenizer.decode(token_ids)

#Embed the tokens using token_ids.
embedded_matrix = embedder.embedding(token_ids)

#Self-attention.
attention = Attention(embedded_matrix, embed_dim, reduced_dim, vocab_size)

attention.create_matrices()
attention.create_vectors()
attention.dot_product()
softmax_matrix = attention.softmax()
A_matrix = attention.aggregate_values()
Y_matrix = attention.project_output()
X_prime = attention.add_residual()

#MLP.
mlp = MLP(X_prime, vocab_size, embed_dim)

mlp.create_matrices()
mlp.layer1()
mlp.layer2()
mlp.output()
mlp.out_vocab()
mlp.probabilities()

#Verifications

print("--- TOKENIZATION OUTPUT ---")
print("Original:     ", repr(prompt))
print("Token IDs:    ", token_ids)
print("Reconstructed:", repr(reconstructed))
print("")
assert reconstructed == prompt

print("--- EMBEDDER OUTPUT ---")
embedder.verification()
print("Embedded Matrix Shape: ", embedder.verification())
print(f"Interpretation: {embedder.verification()[0]} rows of vector math across {embedder.verification()[1]} token columns.")
print("")

print("--- POSITIONAL EMBEDDER OUTPUT ---")
embedder.verify_pos(sequence_length=len(token_ids))
print("")

print("--- ATTENTION OUTPUT ---")
attention.verification()
print("")

print("--- MLP OUTPUT ---")
mlp.verification()
