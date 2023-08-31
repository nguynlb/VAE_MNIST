import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Compose


@torch.inference_mode()
def generative_random_image(model: nn.Module,
                            dataset: Dataset,
                            reversed_transforms: Compose,
                            device: torch.device,
                            seed: int = 82):
    image = dataset[10][0].unsqueeze(dim=0).to(device)
    mean, std = model.encoder(image)
    z_dim = mean + std * torch.randn_like(mean)
    decode_image = model.decoder(z_dim).squeeze(dim=0)
    reconstruct_image = reversed_transforms(decode_image)

    # plt.imshow(image.squeeze(dim=0).permute(1, 2, 0).cpu())
    plt.imshow(reconstruct_image.cpu(), cmap='gray')
    plt.show()

