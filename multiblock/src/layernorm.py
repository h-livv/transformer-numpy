import numpy as np

class LayerNorm:

    def __init__(self, embed_dim, eps=1e-5):
        self.embed_dim = embed_dim
        self.eps = eps
        self.gamma = np.ones((embed_dim, 1))
        self.beta = np.zeros((embed_dim, 1))

    def forward(self, X):
        self.X = X
        self.mean = np.mean(X, axis=0, keepdims=True)
        self.xc = X - self.mean
        self.var = np.mean(self.xc ** 2, axis=0, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.X_hat = self.xc / self.std
        self.Y = self.gamma * self.X_hat + self.beta
        return self.Y

    def backward(self, dY):
        D = self.X.shape[0]
        self.dgamma = np.sum(dY * self.X_hat, axis=1, keepdims=True)
        self.dbeta = np.sum(dY, axis=1, keepdims=True)

        dX_hat = dY * self.gamma
        dX = (1.0 / D) / self.std * (
            D * dX_hat
            - np.sum(dX_hat, axis=0, keepdims=True)
            - self.X_hat * np.sum(dX_hat * self.X_hat, axis=0, keepdims=True)
        )
        return dX

    def step(self, lr):
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta

    def verification(self):
        print(f"LayerNorm Y:     {self.Y.shape}")
