import numpy as np

class Attention:

    def __init__(self, embedded_matrix, embed_dim, reduced_dim, vocab_size):
        self.X = embedded_matrix
        self.embed_dim = embed_dim
        self.reduced_dim = reduced_dim
        self.vocab_size = vocab_size

    def create_matrices(self):
        self.Wq = np.random.randn(self.reduced_dim, self.embed_dim)
        self.Wk = np.random.randn(self.reduced_dim, self.embed_dim)
        self.Wv = np.random.randn(self.reduced_dim, self.embed_dim)
        self.Wo = np.random.randn(self.embed_dim, self.reduced_dim)


    def create_vectors(self):
        self.Q = self.Wq@self.X
        self.K = self.Wk@self.X
        self.V = self.Wv@self.X
    
    def dot_product(self):
        self.dot = (self.Q.T @ self.K) / np.sqrt(self.reduced_dim)

    def softmax(self):

        sequence_length = self.dot.shape[0]
        upper_indices = np.triu_indices(sequence_length, k=1)

        masked_dot = self.dot.copy()
        masked_dot[upper_indices] = -np.inf
        
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
