import numpy as np

class Backpropagation:

    def __init__(self, token_ids=None, probabilities=None):
        self.token_ids = token_ids
        self.probabilities = probabilities
    
    def loss_function(self, token_ids=None, probabilities=None):
        if token_ids is not None:
            self.token_ids = token_ids
        if probabilities is not None:
            self.probabilities = probabilities

        self.target_ids = self.token_ids[1:]
        self.positions = np.arange(len(self.target_ids))

        self.target_probabilities = self.probabilities[self.target_ids, self.positions]

        self.loss = -np.mean(np.log(self.target_probabilities + 1e-12))

        return self.loss

    def dl_dz(self):
        n = len(self.target_ids)

        self.dldz = np.zeros_like(self.probabilities)

        used = self.probabilities[:, :n]
        Y = np.zeros_like(used)
        Y[self.target_ids, self.positions] = 1

        self.dldz[:, :n] = (used - Y) / n

        return self.dldz
