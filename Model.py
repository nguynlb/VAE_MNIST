from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        submodule = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=self.h_dim),
            nn.ReLU()
        )

        mean_layer = nn.Linear(in_features=self.h_dim, out_features=self.z_dim)
        std_layer = nn.Linear(in_features=self.h_dim, out_features=self.z_dim)

        _hidden = submodule(x)
        return mean_layer(F.sigmoid(_hidden)), std_layer(F.relu(_hidden))

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        model = nn.Sequential(
            nn.Linear(in_features=self.z_dim, out_features=self.h_dim),
            nn.Linear(in_features=self.h_dim, out_features=28*28)
        )

        return model(z).reshape((-1, 1, 28, 28))

    def reparameter_trick(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        epi = torch.randn_like(mean)
        return mean + epi * std

    def __init__(self, h_dim: int, z_dim: int) -> None:
        super().__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
         mean, std = self.encoder(x)
         z_hidden = self.reparameter_trick(mean, std)
         return self.decoder(z_hidden), mean, std


def check():
    x = torch.ones(1, 1, 28, 28)
    h_dim, z_dim = 256, 32
    model = VAE(h_dim=h_dim, z_dim=z_dim)
    mean, std = model.encoder(x)
    assert model.forward(x)[0].shape == x.shape
    assert mean.shape == (1, z_dim)
    assert model.reparameter_trick(mean, std).shape == mean.shape
    print("OK")


X = torch.Tensor([[1, 2, 3], [5, 2, 3]])
Y = torch.Tensor([[1, 2, 3], [5, 3, 2]])


