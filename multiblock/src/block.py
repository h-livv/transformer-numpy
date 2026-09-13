from src.attention import Attention
from src.mlp import MLP
from src.layernorm import LayerNorm

class Block:

    def __init__(self, embed_dim, reduced_dim, n_heads, hidden_dim):
        self.ln1 = LayerNorm(embed_dim)
        self.attention = Attention(embed_dim, reduced_dim, n_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, hidden_dim)

    def forward(self, X):
        self.X = X
        attn_out = self.attention.forward(self.ln1.forward(X))
        self.X_prime = X + attn_out
        ffn_out = self.mlp.forward(self.ln2.forward(self.X_prime))
        self.X_out = self.X_prime + ffn_out
        return self.X_out

    def backward(self, dX_out):
        dX_prime = dX_out.copy()
        d_ln2 = self.mlp.backward(dX_out)
        dX_prime = dX_prime + self.ln2.backward(d_ln2)

        dX = dX_prime.copy()
        d_ln1 = self.attention.backward(dX_prime)
        dX = dX + self.ln1.backward(d_ln1)
        return dX

    def step(self, lr):
        self.ln1.step(lr)
        self.attention.step(lr)
        self.ln2.step(lr)
        self.mlp.step(lr)

    def verification(self, index):
        print(f"--- BLOCK {index} LN1 ---")
        self.ln1.verification()
        print(f"--- BLOCK {index} ATTENTION ---")
        self.attention.verification()
        print("")
        print(f"--- BLOCK {index} LN2 ---")
        self.ln2.verification()
        print(f"--- BLOCK {index} MLP ---")
        self.mlp.verification()
