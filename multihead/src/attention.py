import numpy as np

class Attention:

    def __init__(self, embed_dim, reduced_dim, n_heads):
        self.embed_dim = embed_dim
        self.reduced_dim = reduced_dim
        self.n_heads = n_heads
        self.concat_dim = n_heads * reduced_dim
        self.create_matrices()

    def create_matrices(self):
        scale = np.sqrt(1.0 / self.embed_dim)
        self.Wq = []
        self.Wk = []
        self.Wv = []
        for _ in range(self.n_heads):
            self.Wq.append(np.random.randn(self.reduced_dim, self.embed_dim) * scale)
            self.Wk.append(np.random.randn(self.reduced_dim, self.embed_dim) * scale)
            self.Wv.append(np.random.randn(self.reduced_dim, self.embed_dim) * scale)
        self.Wo = np.random.randn(self.embed_dim, self.concat_dim) * np.sqrt(1.0 / self.concat_dim)

    def attend_one_head(self, Wq, Wk, Wv):
        Q = Wq @ self.X
        K = Wk @ self.X
        V = Wv @ self.X

        dot = (Q.T @ K) / np.sqrt(self.reduced_dim)

        masked_dot = dot.copy()
        masked_dot[self.upper_indices] = -np.inf

        exp_arr = np.exp(masked_dot - np.max(masked_dot, axis=1, keepdims=True))
        softmax_arr = exp_arr / np.sum(exp_arr, axis=1, keepdims=True)

        Y = V @ softmax_arr.T

        return Q, K, V, dot, softmax_arr, Y

    def create_vectors(self):
        self.Q = []
        self.K = []
        self.V = []
        self.dot = []
        self.softmax_arr = []
        self.Y = []

        for h in range(self.n_heads):
            Q, K, V, dot, softmax_arr, Y = self.attend_one_head(
                self.Wq[h],
                self.Wk[h],
                self.Wv[h],
            )
            self.Q.append(Q)
            self.K.append(K)
            self.V.append(V)
            self.dot.append(dot)
            self.softmax_arr.append(softmax_arr)
            self.Y.append(Y)

    def concatenate_heads(self):
        self.Y_concat = np.concatenate(self.Y, axis=0)
        return self.Y_concat

    def project_output(self):
        self.output_matrix = self.Wo @ self.Y_concat
        return self.output_matrix

    def add_residual(self):
        self.X_prime = self.X + self.output_matrix
        return self.X_prime

    def forward(self, X):
        self.X = X
        sequence_length = self.X.shape[1]
        self.upper_indices = np.triu_indices(sequence_length, k=1)
        self.create_vectors()
        self.concatenate_heads()
        self.project_output()
        return self.add_residual()

    def backward_one_head(self, dY, Q, K, V, softmax_arr, Wq, Wk, Wv):
        dV = dY @ softmax_arr
        dA = dY.T @ V

        sum_ad = np.sum(dA * softmax_arr, axis=1, keepdims=True)
        dS = softmax_arr * (dA - sum_ad)
        dS[self.upper_indices] = 0

        dG = dS / np.sqrt(self.reduced_dim)
        dQ = K @ dG.T
        dK = Q @ dG

        dWq = dQ @ self.X.T
        dWk = dK @ self.X.T
        dWv = dV @ self.X.T

        dX = Wq.T @ dQ
        dX = dX + Wk.T @ dK
        dX = dX + Wv.T @ dV

        return dWq, dWk, dWv, dX

    def backward(self, dX_prime):
        dO = dX_prime
        dX = dX_prime.copy()

        self.dWo = dO @ self.Y_concat.T
        dY_concat = self.Wo.T @ dO

        self.dWq = []
        self.dWk = []
        self.dWv = []

        for h in range(self.n_heads):
            start = h * self.reduced_dim
            end = start + self.reduced_dim
            dY = dY_concat[start:end, :]

            dWq, dWk, dWv, dX_head = self.backward_one_head(
                dY,
                self.Q[h],
                self.K[h],
                self.V[h],
                self.softmax_arr[h],
                self.Wq[h],
                self.Wk[h],
                self.Wv[h],
            )
            self.dWq.append(dWq)
            self.dWk.append(dWk)
            self.dWv.append(dWv)
            dX = dX + dX_head

        return dX

    def step(self, lr):
        for h in range(self.n_heads):
            self.Wq[h] -= lr * self.dWq[h]
            self.Wk[h] -= lr * self.dWk[h]
            self.Wv[h] -= lr * self.dWv[h]
        self.Wo -= lr * self.dWo

    def verification(self):
        print("Heads:", self.n_heads)
        print("X:", self.X.shape)
        for h in range(self.n_heads):
            print(f"Head {h} Q:", self.Q[h].shape)
            print(f"Head {h} K:", self.K[h].shape)
            print(f"Head {h} V:", self.V[h].shape)
            print(f"Head {h} Scores:", self.dot[h].shape)
            print(f"Head {h} Attention:", self.softmax_arr[h].shape)
            print(f"Head {h} Y:", self.Y[h].shape)
        print("Y concat:", self.Y_concat.shape)
        print("O:", self.output_matrix.shape)
        print("X':", self.X_prime.shape)
