import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.utils import save_image
from pathlib import Path


@torch.inference_mode()
def save_generative_image(model: nn.Module,
                            dataset: Dataset,
                            device: torch.device,
                            seed: int = 28):
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


@torch.inference_mode()
def visualize_generative_image(vae_model: nn.Module, 
                               dataset: Dataset, 
                               device: torch.device, 
                               size: int=10, 
                               seed: int=34):
    torch.manual_seed(seed)

    z_parameters = [] # store mean and std
    random_idx = torch.randint(0, len(dataset), (size, ))
    for i in random_idx:
        image, _ = dataset[i]
        mean, std = vae_model.encoder(image.unsqueeze(dim=0).to(device))
        z_parameters.append((mean, std))

    nrows, ncols = size, 8
    _std_shape = z_parameters[0][1].shape # To parameter for torch.randn_like()
    
    _, axes = plt.subplots(nrows, ncols, figsize=(6, 4))
    for x in range(nrows):
        ax = axes[x, 0]
        ax.imshow(dataset[random_idx[x]][0].permute(1, 2, 0), cmap='gray')
        ax.axis("off")

        mean, std = z_parameters[x]
        for y in range(1, ncols):
            epsilon = torch.randn(_std_shape, device=device)
            latent_vector = mean + std * epsilon
            reconstructor_image = \
                vae_model.decoder(latent_vector).squeeze(dim=0).permute(1,2,0)

            ax = axes[x, y]
            ax.imshow(reconstructor_image.cpu(), cmap='gray')
            ax.axis("off")

    plt.show()

