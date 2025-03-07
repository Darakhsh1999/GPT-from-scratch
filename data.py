import torch
import data_utils
import numpy as np
from tokenizer import Tokenizer
from torch.utils.data import Dataset, DataLoader


class TextData(Dataset):

    def __init__(self, text, vocab, config):
        self.tokenizer = Tokenizer(vocab=vocab)
        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        self.config = config
    
    def __len__(self):
        return len(self.data)-self.config["batch_size"]

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.config["block_size"]]
        y = self.data[idx+1:idx+self.config["block_size"]+1]
        return (x,y)
    
    def get_example(self, n=3):
        indices = np.random.choice(len(self.data)-self.config["batch_size"], size=n, replace=False)
        for idx in indices:
            x,y = self.__getitem__(idx)
            print(f"x =",x)
            print(f"y =",y)
            print(5*"---")



if __name__ == "__main__":

    text = data_utils.load_text()
    vocab = data_utils.get_characters_from_text(text)
    config = {
        "block_size": 8,
        "batch_size": 4,
    }

    dataset = TextData(text, vocab, config)
    #dataset.get_example(3)


    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    for batch_idx, (x,y) in enumerate(dataloader):
        if batch_idx == 3: break
        print(f"x =",x)
        print(f"y =",y)
        print(5*"---")
    

        