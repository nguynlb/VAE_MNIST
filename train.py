import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from typing import Dict

def kl_div(mean: torch.Tensor, std: torch.Tensor):
    return -torch.sum(1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2))


def train_step(model: nn.Module,
               train_dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    model.train()
    loop_bar = tqdm(enumerate(train_dataloader))
    total_loss = 0

    for idx, (X, _) in loop_bar:
        # Send to device
        X = X.to(device)

        # Forward
        y_preds, mean, std = model(X) 

        # Calculate loss value
        loss = loss_fn(y_preds, X) + kl_div(mean, std)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update loop bar information
        loop_bar.set_postfix(loss=loss.item())

        with torch.inference_mode():
            total_loss += loss

    with torch.inference_mode():
        total_loss /= len(train_dataloader)
        print(f"Train loss {total_loss}")
        
    return total_loss


def train_loop(model: nn.Module,
               train_dataloader: DataLoader,
               epochs: int,
               lr: float,
               device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Training loop through dataloader
    
    parameters:
        - model: Variational AE model
        - train_dataloader: DataLoader instance to train model
        - epochs: times to loop through all data
        - lr: learning rate
        - device: GPU or CPU
    """
    loss_fn = nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    results = {"train_loss": []}

    for epoch in range(epochs):
        print(f"Epoch {epoch}:\n----------\n")
        loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
        results["train_loss"].append(loss)
    return results
