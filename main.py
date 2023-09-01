import os
from argparse import ArgumentParser
from get_data import create_data, simple_transform
import torch.cuda
from model import VAE
from train import train_loop
from predict import generative_random_image
from pathlib import Path

def main(arg):
    if not arg.disable_cuda and torch.cuda.is_available():
        arg.device = torch.device("cuda")
    else:
        arg.device = torch.device("cpu")

    train_transforms, reversed_transforms = simple_transform()
    train_dataloader, class_names = create_data(arg.dir, train_transforms, arg.batch_size)

    epochs = arg.epochs
    lr = arg.lr
    device = arg.device
    # device = torch.device("cpu")

    h_dim, z_dim = arg.h_dim, arg.z_dim

    model = VAE(h_dim=h_dim, z_dim=z_dim).to(device)

    save_path = Path(arg.save_model)
    model_name = "VAE.pth"
    if not arg.load_model and save_path.is_dir():
        model.load_state_dict(torch.load(save_path / model_name))
        train_query = input("Train more? [Y/N] (default N) :")
        if train_query.upper() == 'Y':
            results = train_loop(model=model,
                                 train_dataloader=train_dataloader,
                                 epochs=epochs,
                                 lr=lr,
                                 device=device)
            torch.save(obj=model.state_dict(), f=save_path / model_name)

    else:
        save_path.mkdir(parents=True, exist_ok=True)
        results = train_loop(model=model,
                             train_dataloader=train_dataloader,
                             epochs=epochs,
                             lr=lr,
                             device=device)
        torch.save(obj=model.state_dict(), f=save_path / model_name)

    
    generative_random_image(model,
                            train_dataloader.dataset,
                            reversed_transforms,
                            device)

if __name__ == "__main__":
    NUM_WORKERS = os.cpu_count()

    parser = ArgumentParser(description="Config parameter and hyperparameter")
    parser.add_argument("-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", "-hd", type=int, default=200)
    parser.add_argument("--disable-cuda", "-cpu", default=False, action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-model", help="Directory to save model", metavar="L", default="./models")
    parser.add_argument("--dir", "-d", type=str, default='./data')
    parser.add_argument("--h-dim", type=int, default=256, help="hidden dimension")
    parser.add_argument("--z-dim", type=int, default=32, help="latent dimension")
    parser.add_argument("--epochs", "-e", help="Epochs", type=int, default=16)
    parser.add_argument("--num-workers", "-n", default=NUM_WORKERS)
    parser.add_argument("--load-model", help='Load existing model', default=False)


    arg = parser.parse_args()

    main(arg)