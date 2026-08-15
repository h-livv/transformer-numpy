import numpy as np

class Embedder:

    #embed_dim: size of vector for each token. vocab_size = total vocab size.
    def __init__(self, embed_dim, vocab_size):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.weights = np.random.randn(self.embed_dim, self.vocab_size) * 0.01

    #sequence_length: number of tokens in the prompt.
    def get_pos_encoding(self, sequence_length):

        #Create a matrix representing the embedding of the prompt tokens.
        pe = np.zeros((self.embed_dim, sequence_length))

        #Create an evenly spaced 2D column vector representing the positions of the tokens.
        position = np.arange(sequence_length).reshape(1, -1)
        
        #The denominator in the sine/cosine functions.
        div_term = np.exp(np.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
        div_term = div_term.reshape(-1, 1)
        
        #Apply sin to even dimensions, cos to odd dimensions.
        pe[0::2, :] = np.sin(div_term @ position)
        pe[1::2, :] = np.cos(div_term @ position)

        return pe

    def get_pos_encoding_naive(self, sequence_length, embedding_dim):
        pe = np.zeros((embedding_dim, sequence_length))

        for pos in range(sequence_length):
            for i in range(embedding_dim // 2):
                denominator = 10000 ** (2 * i / embedding_dim)

                pe[2 * i, pos] = np.sin(pos / denominator)
                pe[2 * i + 1, pos] = np.cos(pos / denominator)

        return pe

    def verify_pos(self, sequence_length):
        pe_vectorized = self.get_pos_encoding(sequence_length)
        pe_naive = self.get_pos_encoding_naive(
            sequence_length,
            self.embed_dim
        )

        print(
            "Maximum difference:",
            np.max(np.abs(pe_vectorized - pe_naive))
        )

        assert np.allclose(pe_vectorized, pe_naive)

    #Perform token embedding, and add positional embedding.
    #Input is the encoded char IDs.
    def embedding(self, token_ids):
        sequence_length = len(token_ids)

        #Extract only those embeddings that correspond to the prompt.
        token_embeddings = self.weights[:, token_ids]

        #Get positional embedding.
        pos_encodings = self.get_pos_encoding(sequence_length)

        #Add the two.
        return token_embeddings + pos_encodings
