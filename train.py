from src.tokenizer import Tokenizer
from src.embedder import Embedder
from src.attention import Attention

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
prompt = "Hello my name is John"

#Encode the prompt to get the token IDs.
token_ids = tokenizer.encode(prompt)

#Decode for verification
reconstructed = tokenizer.decode(token_ids)

#Embed the tokens using token_ids.
embedded_matrix = embedder.embedding(token_ids)

attention = Attention(embedded_matrix, embed_dim, reduced_dim)

attention.create_matrices()
attention.create_vectors()
attention.dot_product()
softmax_matrix = attention.softmax()
value_multiplication = attention.value_multi()
output = attention.output()
final_matrix = attention.final_update()

#Verifications

print("--- TOKENIZATION OUTPUT ---")
print("Original:     ", repr(prompt))
print("Token IDs:    ", token_ids)
print("Reconstructed:", repr(reconstructed))
print("")
assert reconstructed == prompt

print("--- EMBEDDER OUTPUT ---")
print(f"Output Matrix Shape: {embedded_matrix.shape}")
print(f"Interpretation: {embedded_matrix.shape[0]} rows of vector math across {embedded_matrix.shape[1]} token columns.")
print("")

print("--- POSITIONAL EMBEDDER OUTPUT ---")
embedder.verify_pos(sequence_length=len(token_ids))
print("")

print("--- ATTENTION OUTPUT ---")
attention.verification()

