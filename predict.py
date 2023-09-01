import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.utils import save_image
from pathlib import Path

@torch.inference_mode()
def generative_random_image(model: nn.Module,
                            dataset: Dataset,
                            reversed_transforms: Compose,
                            device: torch.device,
                            seed: int = 82):
    torch.manual_seed(seed)
    idx = torch.randint(0, 800, (10, ))
    image_path = Path("./image", parents=True, exist_ok=True)
    if not image_path.is_dir():
        image_path.mkdir()

    for i in idx:
        image = dataset[i][0].unsqueeze(dim=0).to(device)
        mean, std = model.encoder(image)
        z_dim = mean + std * torch.randn_like(mean)
        decode_image = model.decoder(z_dim).squeeze(dim=0)
        # reconstruct_image = reversed_transforms(decode_image) 

        # plt.imshow(image.squeeze(dim=0).permute(1, 2, 0).cpu())
        fp = image_path / f"generative_{dataset[i][1]}.png"
        save_image(decode_image.cpu(), fp)

