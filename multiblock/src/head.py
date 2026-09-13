import numpy as np

class LanguageModelHead:

    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.Wout = np.random.randn(self.vocab_size, self.embed_dim) * np.sqrt(1.0 / self.embed_dim)

    def forward(self, X):
        self.X = X
        self.Z = self.Wout @ self.X
        self.exp_arr = np.exp(self.Z - np.max(self.Z, axis=0, keepdims=True))
        self.softmax_out = self.exp_arr / np.sum(self.exp_arr, axis=0, keepdims=True)
        return self.softmax_out

    def next_token_id(self):
        return int(np.argmax(self.Z[:, -1]))

    def next_token_probs(self):
        return self.softmax_out[:, -1]

    def backward(self, dZ):
        self.dWout = dZ @ self.X.T
        return self.Wout.T @ dZ

    def step(self, lr):
        self.Wout -= lr * self.dWout

    def verification(self):
        print(f"Input X:         {self.X.shape}")
        print(f"Logits Z:        {self.Z.shape}")
        print(f"Probabilities P: {self.softmax_out.shape}")
        print("")
