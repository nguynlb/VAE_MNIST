from typing import Tuple
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, img_shape: int, h_dim: int = 256, z_dim: int = 32 ) -> None:
        super().__init__()
        # Hyperparameter
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.img_shape = img_shape

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Encoder
        self.encoder_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=self.h_dim),
            self.relu
        )
        self.mean_layer = nn.Linear(in_features=self.h_dim, out_features=self.z_dim)
        self.sigmoid_layer = nn.Linear(in_features=self.h_dim, out_features=self.z_dim)

        # Decoder
        self.decoder_layer = nn.Sequential(
            nn.Linear(in_features=self.z_dim, out_features=self.h_dim),
            self.relu,
            nn.Linear(in_features=self.h_dim, out_features=self.img_shape * self.img_shape),
            self.sigmoid
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, sigmoid = self.encoder(x)
        z_hidden = mean + sigmoid * torch.randn_like(sigmoid)
        return self.decoder(z_hidden), mean, sigmoid

    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _hidden = self.encoder_layer(x)
        mean = self.mean_layer(self.sigmoid(_hidden))
        sigmoid = self.sigmoid_layer(self.relu(_hidden))
        return mean, sigmoid

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        reconstruct_image = self.decoder_layer(z)
        return reconstruct_image.reshape(-1, 1, self.img_shape, self.img_shape)
    # MNIST dataset -> so I specify image channel to 1


def check():
    x = torch.ones(1, 1, 28, 28)
    model = VAE(img_shape=28)
    mean, sigmoid = model.encoder(x)
    print(mean.shape)
    print(model(x)[0].shape)
    assert model.forward(x)[0].shape == x.shape
    assert mean.shape == (1, 32)
    # assert model.reparameter_trick(mean, sigmoid).shape == mean.shape
    print("OK")

if __name__ == "__main__":
    check()