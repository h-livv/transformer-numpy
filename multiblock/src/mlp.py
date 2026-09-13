import numpy as np

class MLP:

    def __init__(self, embed_dim, hidden_dim=64):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.create_matrices()

    def create_matrices(self):
        self.w1 = np.random.randn(self.hidden_dim, self.embed_dim) * np.sqrt(1.0 / self.embed_dim)
        self.b1 = np.zeros((self.hidden_dim, 1))

        self.w2 = np.random.randn(self.embed_dim, self.hidden_dim) * np.sqrt(1.0 / self.hidden_dim)
        self.b2 = np.zeros((self.embed_dim, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def layer1(self):
        self.H_pre = (self.w1 @ self.X_prime) + self.b1
        self.H = self.relu(self.H_pre)
        return self.H

    def layer2(self):
        self.F = (self.w2 @ self.H) + self.b2
        return self.F

    def forward(self, X_prime):
        self.X_prime = X_prime
        self.layer1()
        return self.layer2()

    def backward(self, dF):
        dX_prime = np.zeros_like(self.X_prime)

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

    def verification(self):
        print(f"Input X':        {self.X_prime.shape}")
        print(f"Hidden H:        {self.H.shape}")
        print(f"FFN output F:    {self.F.shape}")
        print("")
