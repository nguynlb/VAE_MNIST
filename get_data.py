import os
from typing import Tuple, List
from pathlib import Path
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.utils.data import DataLoader

# Hyperparameter
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()


# DataLoader
def create_data(data_dir: str,
                train_transform: Compose,
                batch_size: int = BATCH_SIZE,
                num_workers: int = NUM_WORKERS) -> \
        Tuple[DataLoader, List[str]]:
    """
       Pipeline handle data. Download MNIST dataset and create dataloader instance.
       :param data_dir: path to data directory, if not exits, create
       :param train_transform: train image transformer
       :param num_workers:
       :param batch_size:
       :return: Tuple of train dataloader
       """
    data_path_dir = Path(data_dir)

    # Check data path is existed.
    # If true, skip downloading.
    if data_path_dir.is_dir():
        print("Data has been created. Skip downloading...")
        train_dataset = MNIST(root=data_path_dir / "train",
                              train=True,
                              transform=train_transform,
                              download=False,
                              target_transform=None)

    else:
        data_path_dir.mkdir(parents=True, exist_ok=True)
        print(f"Start downloading")
        train_dataset = MNIST(root=data_path_dir / "train",
                              train=True,
                              transform=train_transform,
                              download=True,
                              target_transform=None)

        print("Download successfully")

    # Create dataloader
    class_names = train_dataset.classes
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True,
                                  pin_memory=True)

    return train_dataloader, class_names


def simple_transform() -> Tuple[Compose, Compose]:
    transforms = Compose([
        ToTensor(),
    ])
    reversed_transforms = Compose([

        Lambda(lambda x: x * 255),
        Lambda(lambda x: x.permute(1, 2, 0))
    ])

    return transforms, reversed_transforms

