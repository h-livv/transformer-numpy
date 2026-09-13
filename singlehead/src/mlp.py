import numpy as np

class MLP:

    def __init__(self, vocab_size, embed_dim, hidden_dim=64):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.create_matrices()

    def create_matrices(self):
        self.w1 = np.random.randn(self.hidden_dim, self.embed_dim) * np.sqrt(1.0 / self.embed_dim)
        self.b1 = np.zeros((self.hidden_dim, 1))

        self.w2 = np.random.randn(self.embed_dim, self.hidden_dim) * np.sqrt(1.0 / self.hidden_dim)
        self.b2 = np.zeros((self.embed_dim, 1))

        self.Wout = np.random.randn(self.vocab_size, self.embed_dim) * np.sqrt(1.0 / self.embed_dim)

    def relu(self, x):
        return np.maximum(0,x)

    def layer1(self):
        self.H_pre = (self.w1 @ self.X_prime) + self.b1
        self.H = self.relu(self.H_pre)
        return self.H

    def layer2(self):
        self.F = (self.w2 @ self.H) + self.b2
        return self.F

    def output(self):
        self.X_doublep = self.X_prime + self.F
        return self.X_doublep

    def out_vocab(self):

        self.Z = self.Wout @ self.X_doublep

        return self.Z
    
    def probabilities(self):

        self.exp_arr = np.exp(self.Z - np.max(self.Z, axis=0, keepdims=True))

        self.softmax_out = self.exp_arr / np.sum(self.exp_arr, axis=0, keepdims=True)

        return self.softmax_out

    def next_token_id(self):
        return int(np.argmax(self.Z[:, -1]))

    def next_token_probs(self):
        return self.softmax_out[:, -1]

    def forward(self, X_prime):
        self.X_prime = X_prime
        self.layer1()
        self.layer2()
        self.output()
        self.out_vocab()
        self.probabilities()
        return self.softmax_out

    def backward(self, dZ):
        self.dWout = dZ @ self.X_doublep.T
        dX_doublep = self.Wout.T @ dZ

        dF = dX_doublep
        dX_prime = dX_doublep.copy()

        self.dw2 = dF @ self.H.T
        self.db2 = np.sum(dF, axis=1, keepdims=True)
        dH = self.w2.T @ dF

        dH_pre = dH * (self.H_pre > 0)
        self.dw1 = dH_pre @ self.X_prime.T
        self.db1 = np.sum(dH_pre, axis=1, keepdims=True)
        dX_prime += self.w1.T @ dH_pre

        return dX_prime

    def step(self, lr):
        self.w1 -= lr * self.dw1
        self.b1 -= lr * self.db1
        self.w2 -= lr * self.dw2
        self.b2 -= lr * self.db2
        self.Wout -= lr * self.dWout

    def verification(self):
        print(f"Input X':        {self.X_prime.shape}")
        print(f"Hidden H:        {self.H.shape}")
        print(f"FFN output F:    {self.F.shape}")
        print(f"Output X'':      {self.X_doublep.shape}")
        print(f"Logits Z:        {self.Z.shape}")
        print(f"Probabilities P: {self.softmax_out.shape}")
        print("")
