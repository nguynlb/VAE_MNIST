from typing import Tuple
import torch
import torch.nn as nn

class VAE(nn.Module):
    def encoder(self, in_dim: int, h_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def decoder(self, in_dim: int, h_dim: int) -> torch.Tensor:
        pass

    def __int__(self):
        pass

    def forward(self, x: torch.Tensor):
        pass