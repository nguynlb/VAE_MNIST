from typing import Tuple
import torch
import torch.nn as nn


class VAE(nn.Module):
    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_layer(self.sigmoid(z)).reshape(-1, 1, 28, 28)

    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _hidden = self.encoder_layer(x)
        mean = self.mean_layer(self.sigmoid(_hidden))
        std = self.std_layer(self.relu(_hidden))
        return mean, std

    def __init__(self, h_dim: int, z_dim: int) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Encoder
        self.encoder_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=self.h_dim),
            self.relu
        )
        self.mean_layer = nn.Linear(in_features=self.h_dim, out_features=self.z_dim)
        self.std_layer = nn.Linear(in_features=self.h_dim, out_features=self.z_dim)

        # Decoder
        self.decoder_layer = nn.Sequential(
            nn.Linear(in_features=self.z_dim, out_features=self.h_dim),
            self.relu,
            nn.Linear(in_features=self.h_dim, out_features=28 * 28),
            self.sigmoid
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
         mean, std = self.encoder(x)
         z_hidden = mean + std * torch.randn_like(mean)
         return self.decoder(z_hidden), mean, std


def check():
    x = torch.ones(1, 1, 28, 28)
    h_dim, z_dim = 256, 32
    model = VAE(h_dim=h_dim, z_dim=z_dim)
    print(model.parameters())
    mean, std = model.encoder(x)
    assert model.forward(x)[0].shape == x.shape
    assert mean.shape == (1, z_dim)
    assert model.reparameter_trick(mean, std).shape == mean.shape
    print("OK")
