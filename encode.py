import argparse

import torch
import torch.nn as nn
from torchvision.models import resnet18

from models import create_resnet_encoder
from utils.encoder_pipeline import encoder_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='encode',
        description='encodes images')

    parser.add_argument('-B', type=int,
                        help = '',
                        default='6')

    parser.add_argument('--encoder_name', '-m', type=str,
                        help = '',
                        default='encoder__resnet18__32x_512__sigmoid_before_skipconncetion')
    
    parser.add_argument(
        '--encoder_weights', '-w', type=str,
        help = '',
        default='weights/resnet_autoencoder_abs/encoder__resnet_autoencoder__512x16x16__upsample__B_6__66_epochs__2023-06-10T00_39.pt')

    parser.add_argument('--device', '-d', type=str,
                        help = '',
                        default='cpu')
    
    parser.add_argument('--image_path', '-i', type=str,
                        help = 'image_to_encode')

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()

    device = torch.device(args.device)

    if args.encoder_name == "encoder__resnet18__32x_512__sigmoid_before_skipconncetion":
        encoder = create_resnet_encoder(
            resnet18(), nn.Identity(), nn.Sigmoid())
    
    encoder.load_state_dict(torch.load(args.encoder_weights))
    
    out = encoder_pipeline(encoder, args.image_path)
