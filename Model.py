from typing import Tuple
import torch
import torch.nn as nn

class VAE(nn.Module):
    def encoder(self, x: torch.Tensor) -> Tuple[nn.Module, nn.Module]:
        submodule = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=256),
            nn.ReLU()
        )

        mean_layer = nn.Linear(in_features=256, out_features=200)
        std_layer = nn.Linear(in_features=256, out_features=self.h_dim)

        _hidden = submodule(x)
        return mean_layer(nn.Softmax(_hidden)), std_layer(nn.ReLu(_hidden))

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        model = nn.Sequential(
            nn.Linear(in_features=self.h_dim, out_features=256),
            nn.Linear(in_features=256, out_features=28*28)
        )

        return model(z).reshape((-1, 1, 28, 28))

    def reparameter_trick(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        pass

    def __init__(self, h_dim: int) -> None:
        self.h_dim = h_dim


    def forward(self, x: torch.Tensor) -> torch.Tensor:
