import numpy as np

class Tokenizer:

    #Input is the text you want to train the model on.
    def __init__(self, input):
        self.input = input
        self.unique = []
        self.encoder_lookup = {}
        self.decoder_lookup = {}

    def build_vocab(self):

        #Loops through each character in the input, and appends the unique ones to our list.
        for char in self.input:
            if char not in self.unique:
                self.unique.append(char)

        #Fills encoder dict with ids, decoder dict with chars.
        for idx, char in enumerate(self.unique):
            self.encoder_lookup[char] = idx
            self.decoder_lookup[idx] = char
    
    #Input is anything you want to encode/decode.

    #Loops through each char in the text, looks up the encoder dict to get its ID. If not, falls back to zero.
    #Returns a numpy array with IDs.
    def encode(self, text):
        ids = [self.encoder_lookup.get(char, 0) for char in text]
        return np.array(ids, dtype=np.int32)
        
    #Loops through each ID, looks it up in the decoder dict to get the corresponding char.
    #Works for both arrays and lists. Returns a string.
    def decode(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return "".join([self.decoder_lookup.get(id_) for id_ in ids])

    @property
    #Total vocab size. Needed for embedding.
    def vocab_size(self):
        return len(self.unique)

