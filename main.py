from argparse import ArgumentParser
from get_data import create_data, simple_transform
import torch.cuda
from Model import VAE
from train import train_loop

if __name__ == "__main__":
    parser = ArgumentParser(description="Config parameter and hyperparameter")
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", "-hd", type=int, default=200)
    parser.add_argument("--disable-cuda", "-cpu", default=False, action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-model", help="Directory to save model", metavar="L", default="./model")
    parser.add_argument("--dir", "-d", type=str, default='./data')
    parser.add_argument("--h-dim", type=int, default=256, help="hidden dimension")
    parser.add_argument("--z-dim", type=int, default=32, help="latent dimension")
    parser.add_argument("--epochs", "-e", help="Epochs", type=int, default=10)

    arg = parser.parse_args()

    if not arg.disable_cuda and torch.cuda.is_available():
        arg.cuda = torch.device("cuda")
    else:
        arg.cuda = torch.device("cpu")

    train_transforms, reversed_transforms = simple_transform()
    train_dataloader, class_names = create_data(arg.dir, train_transforms, arg.batch_size)

    epochs = arg.epochs
    lr = arg.lr
    device = arg.device

    h_dim, z_dim = arg.h_dim, arg.z_dim
    model = VAE(h_dim=h_dim, z_dim=z_dim).to(arg.device)
    results = train_loop(model=model,
                         train_dataloader=train_dataloader,
                         epochs=epochs,
                         lr=lr,
                         device=device)

    print(results)

