import os
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from train import train
from data import TextData
from model import GPT, GPTConfig
from torch.utils.data import DataLoader


if __name__ == "__main__":


    is_save_model = False

    # Load in data
    config = GPTConfig(vocab_size=50304) # extend pseudo tokens to increase vocab from 50257 -> 50304
    data_path = os.path.join("..","data","txt","1984.txt")
    data = TextData(data_path=data_path, config=config, verbose=True)

    # Data Loaders
    train_loader = DataLoader(data, batch_size=config.batch_size, pin_memory=True)

    # Load in model
    model = GPT(config)
    model.to(config.device)
    #model = torch.compile(model)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1, fused=True)

    # Train model
    train(model, optimizer, train_loader, config, verbose=False, overfit=False)


    # Save model
    if is_save_model:
        model.to("cpu")
        torch.save()