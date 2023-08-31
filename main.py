from argparse import ArgumentParser
from get_data import create_data, simple_transform
import torch.cuda
import torchvision.transforms as transforms

if __name__ == "__main__":
    parser = ArgumentParser(description="Config parameter and hyperparameter")
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", "-hd", type=int, default=200)
    parser.add_argument("--disable-device", "-cpu", type=bool, default=False, action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-model", help="Directory to save model", metavar="L", default="./model")
    parser.add_argument("--dir", "-d", type=str, default='./data')

    arg = parser.parse_args()

    if not arg.disable_cuda and torch.cuda.is_available():
        arg.cuda = torch.device("cuda")
    else:
        arg.cuda = torch.device("cpu")

    train_transforms, reversed_transforms = simple_transform()
    train_dataloader, class_names = create_data(arg.dir, train_transforms, arg.batch_size)

