import argparse

import torch
import torch.nn as nn
from torchvision.models import resnet18

from models import create_resnet_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='encode',
        description='encodes images')

    parser.add_argument('-B', type=int,
                        help = "",
                        default='6')

    parser.add_argument('--model_name', '-m', type=str,
                        help = '',
                        default='encoder__resnet18__32x_512')
    
    parser.add_argument('--model_weights', '-w', type=str,
                        help = '',
                        default='weights/full__resnet_autoencoder__512x16x16__upsample__B_6__63_epochs_2023-06-08T05_56.pt')

    parser.add_argument('--device', '-d', type=str,
                        help = '',
                        default='cpu')

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(args.device)

    if args.model_name == "encoder__resnet18__32x_512":
        encoder = create_resnet_encoder(
            resnet18(), nn.Identity(), nn.Sigmoid())
    
    encoder.load_state_dict(torch.load(args.model_weights))