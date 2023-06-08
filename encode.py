import argparse

import torch
from torchvision.models import resnet18

from models import create_resnet_autoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='encode',
        description='encodes images')

    parser.add_argument('-B', type=int,
                        help = "",
                        default='6')

    parser.add_argument('-model_name', '-m', type=str,
                        help = '',
                        default='resnet18_autoencoder')

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    B = 2
    LOAD_PATH = "weights\\residual_decoder__upsample__B_6__63_epochs_2023-06-08T05_56.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if
    up_func_name = "upsample"
    resnet_autoencoder = create_resnet_autoencoder(
        resnet18,
        up_func_name = up_func_name,
        B=B,
    ).to(device)
