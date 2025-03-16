import torch
from tqdm import tqdm



def train_transformer(model, optimizer, loss_fn, train_loader, val_loader, config):
    

    for epoch_idx in tqdm(range(config["n_epochs"])):

        model.train()
        for x,y in tqdm(enumerate(train_loader)):

            logtis, loss = model(x,y)
            optimizer.zero_grad()
            loss_fn.backward()
            optimizer.step()
        

        # Validation
        model.eval()
        val_metrics = test(model, loss_fn, val_loader)



    print("Training finished")
    return






@torch.no_grad()
def test(model, loss_fn, val_loader):
    pass