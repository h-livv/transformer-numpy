import numpy as np

class Attention:

    def __init__(self, embed_dim, reduced_dim):
        self.embed_dim = embed_dim
        self.reduced_dim = reduced_dim
        self.create_matrices()

    def create_matrices(self):
        scale = np.sqrt(1.0 / self.embed_dim)
        self.Wq = np.random.randn(self.reduced_dim, self.embed_dim) * scale
        self.Wk = np.random.randn(self.reduced_dim, self.embed_dim) * scale
        self.Wv = np.random.randn(self.reduced_dim, self.embed_dim) * scale
        self.Wo = np.random.randn(self.embed_dim, self.reduced_dim) * np.sqrt(1.0 / self.reduced_dim)

    def create_vectors(self):
        self.Q = self.Wq@self.X
        self.K = self.Wk@self.X
        self.V = self.Wv@self.X
    
    def dot_product(self):
        self.dot = (self.Q.T @ self.K) / np.sqrt(self.reduced_dim)

    def softmax(self):

        sequence_length = self.dot.shape[0]
        self.upper_indices = np.triu_indices(sequence_length, k=1)

        masked_dot = self.dot.copy()
        masked_dot[self.upper_indices] = -np.inf
        
        self.exp_arr = np.exp(masked_dot - np.max(masked_dot, axis=1, keepdims=True))

        self.softmax_arr = self.exp_arr / np.sum(self.exp_arr, axis=1, keepdims=True)

        return self.softmax_arr

    def aggregate_values(self):
        
        self.Y = self.V@((self.softmax_arr).T)

        return self.Y

    def project_output(self):
        self.output_matrix = self.Wo @ self.Y

        return self.output_matrix

    def add_residual(self):

        self.X_prime = self.X + self.output_matrix

        return self.X_prime

    def forward(self, X):
        self.X = X
        self.create_vectors()
        self.dot_product()
        self.softmax()
        self.aggregate_values()
        self.project_output()
        return self.add_residual()

    def backward(self, dX_prime):
        dO = dX_prime
        dX = dX_prime.copy()

        self.dWo = dO @ self.Y.T
        dY = self.Wo.T @ dO

        dV = dY @ self.softmax_arr
        dA = dY.T @ self.V

        sum_ad = np.sum(dA * self.softmax_arr, axis=1, keepdims=True)
        dS = self.softmax_arr * (dA - sum_ad)
        dS[self.upper_indices] = 0

        dG = dS / np.sqrt(self.reduced_dim)
        dQ = self.K @ dG.T
        dK = self.Q @ dG

        self.dWq = dQ @ self.X.T
        self.dWk = dK @ self.X.T
        self.dWv = dV @ self.X.T

        dX += self.Wq.T @ dQ
        dX += self.Wk.T @ dK
        dX += self.Wv.T @ dV

        return dX

    def step(self, lr):
        self.Wq -= lr * self.dWq
        self.Wk -= lr * self.dWk
        self.Wv -= lr * self.dWv
        self.Wo -= lr * self.dWo

    def verification(self):
        print("X:", self.X.shape)
        print("Q:", self.Q.shape)
        print("K:", self.K.shape)
        print("V:", self.V.shape)
        print("Scores:", self.dot.shape)
        print("Attention:", self.softmax_arr.shape)
        print("Y:", self.Y.shape)
        print("O:", self.output_matrix.shape)
        print("X':", self.X_prime.shape)
