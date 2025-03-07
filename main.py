import os
import random
import torch
import data_utils

from data import TextData
from model import Transformer
from torch.optim import AdamW
from train import train_transformer
from torch.utils.data import Subset,  DataLoader


# Hyperparameters
config = {
    "block_size": 8,
    "batch_size": 16,
    "n_epochs": 10,
    "head_size": 16,
    "n_heads": 4,
    "d_embed": 64,
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Load data
text = data_utils.load_text()
vocab = data_utils.get_characters_from_text(text)
data = TextData(text=text, vocab=vocab, config=config)
train_data = Subset(data, indices=range(0,int(0.9*len(data))))
val_data = Subset(data, indices=range(int(0.9*len(data)), len(data)))

# Data Loaders
train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=False, drop_last=True)
val_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=False, drop_last=True)


# Load model
model = Transformer(vocab_size=len(vocab), config=config)
model.to(config["device"])

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=config["lr"])
loss_fn = torch.nn.CrossEntropyLoss()

train_transformer(model, optimizer, loss_fn, train_loader, val_loader, config)

exit(0)

# Train model
try:
    train_transformer(model, optimizer, loss_fn, train_loader, config)
except:
    # Save model
    model.to("cpu")
    torch.save(model, os.path.join("models",f"AraGPT{random.randint(0,10000)}.pt"))