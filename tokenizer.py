
class Tokenizer():
    """ Character level tokenizer """

    def __init__(self, vocab: list):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.stoi = {char:idx for idx,char in enumerate(vocab)}
        self.itos = {idx:char for idx,char in enumerate(vocab)}
    

    def encode(self, text):
        """ String -> Integer IDs """
        return [self.stoi[c] for c in text]

    def decode(self, id_list):
        """ Integer IDs -> String """
        return "".join([self.itos[_id] for _id in id_list])