import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

def train_step(model: nn.Module,
               train_dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device):
    model.train()
    loop_bar = tqdm(enumerate(train_dataloader))
    for idx, (X, _) in loop_bar:
        X = X.to(device)
        y_preds = model(X)

        loss = loss_fn(y_preds, X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.inference_mode():
            acc = accuracy_fn(y_preds, X)

            if idx % 10 == 0:
                print(f"Trained on {len(X) * idx}/{len(train_dataloader.dataset)}")
                print(f"Trained on {len(X) * idx}/{len(train_dataloader.dataset)}")





