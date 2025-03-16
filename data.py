import os
import torch
import tiktoken
from time import time
from model import GPTConfig
from torch.utils.data import Dataset, DataLoader


class TextData(Dataset):

    def __init__(self, data_path, config: GPTConfig, verbose=False, benchmark=False):

        with open(data_path, "r", encoding="utf-8") as f:
            self.text = f.read()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        t = time()
        self.text_tokens = torch.Tensor(self.tokenizer.encode(text=self.text)).type(torch.LongTensor)
        if benchmark: print(f"Tokenizing text took {time()-t:.2f} s")
        self.config = config

        self.n = len(self.text_tokens) // (1+self.config.context_size) 

        if verbose: print(f"Text consists of {len(self.text_tokens)} tokens, approx {self.n} batches")
    

    def decode(self, token_ids):
        return [self.tokenizer.decode(token_id) for token_id in token_ids]


    def __getitem__(self, idx):
        start_idx = idx*(1+self.config.context_size)
        x = self.text_tokens[start_idx:start_idx+self.config.context_size]
        y = self.text_tokens[1+start_idx:1+start_idx+self.config.context_size]
        return (x,y)

    def __len__(self):
        return self.n


if __name__ == "__main__":

    data_path = os.path.join("..","data","txt","1984.txt")
    config = GPTConfig()
    data = TextData(data_path=data_path, config=config, verbose=True)

    data_loader = DataLoader(data, batch_size=2, drop_last=True, shuffle=False)
    data_iter = iter(data_loader)

    for _ in range(1):
        x,y = next(data_iter)
        print(x.shape, y.shape)