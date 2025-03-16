import torch
import torch.nn as nn
from tqdm import tqdm
from config import GPTConfig


def train(model, optimizer, train_loader, config:GPTConfig, verbose=False, overfit=False):
    print(f"Started training on device: {config.device}")
    torch.set_float32_matmul_precision("high")

    # Train
    model.train()
    for i in tqdm(range(config.n_epochs), desc="Training"):

        for batch_idx, (x,y) in enumerate(tqdm(train_loader, desc="Batch")):
            if overfit and (batch_idx != 0): break

            # Send data to GPU
            x, y = x.to(config.device), y.to(config.device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                _, loss = model(x,y)
            
            # Calculate backward gradients
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Implement gradient normalization

            # Update weights
            optimizer.step()

            if verbose and (batch_idx % 5 == 0):
                print(f"Epoch {i}, Batch {batch_idx}, loss: {loss.item():.4f}")
    
    # Example validate
    model.eval()

    