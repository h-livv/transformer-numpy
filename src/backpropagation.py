import numpy as np

class Backpropagation:

    def __init__(self, token_ids, probabilities):
        self.token_ids = token_ids
        self.probabilities = probabilities
    
    def loss_function(self):
        self.target_ids = self.token_ids[1:]

        self.positions = np.arange(len(self.target_ids))

        self.target_probabilities = self.probabilities[self.target_ids, self.positions]

        self.loss = -np.mean(np.log(self.target_probabilities))

        return self.loss

    def dl_dz(self):

        Y = np.zeros_like(self.probabilities)

        Y[self.target_ids, self.positions] = 1

        self.dldz = (self.probabilities - Y) / len(self.target_ids)

    def 
