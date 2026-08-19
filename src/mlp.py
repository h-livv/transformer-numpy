import numpy as np

class MLP:

    def __init__(self, X_prime, vocab_size, embed_dim):
        self.X_prime = X_prime
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def create_matrices(self):
        self.w1 = np.random.randn(64, 16)
        self.b1 = np.zeros((64, 1))

        self.w2 = np.random.randn(16, 64)
        self.b2 = np.zeros((16, 1))

        self.Wout = np.random.randn(self.vocab_size, self.embed_dim)

    def relu(self, x):
        return np.maximum(0,x)

    def layer1(self):
        self.H = self.relu((self.w1 @ self.X_prime) + self.b1)
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

    def verification(self):
        print(f"Input X':        {self.X_prime.shape}")
        print(f"Hidden H:        {self.H.shape}")
        print(f"FFN output F:    {self.F.shape}")
        print(f"Output X'':      {self.X_doublep.shape}")
        print(f"Logits Z:        {self.Z.shape}")
        print(f"Probabilities P: {self.softmax_out.shape}")
        print("")
