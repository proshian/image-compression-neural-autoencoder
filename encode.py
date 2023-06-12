import argparse

import torch

from encoder_pipeline import encoder_pipeline
from looseless_compressors import Huffman 
from trained_models import get_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='encode',
        description='encodes images')

    parser.add_argument('--image_path', '-i', type=str,
                        help = 'image_to_encode')
    
    parser.add_argument('-B', type=int,
                        help = '',
                        default='6')

    parser.add_argument('--model_name', '-m', type=str,
                        help = '',
                        default='default')
    
    parser.add_argument('--device', '-d', type=str,
                        help = '',
                        default='cpu')

    parser.add_argument('--compressor_state_path', '-s',
                        type=str, help = '', default=None)
    
    parser.add_argument('--encode_output_path', '-o',
                        type=str, help = '', default=None)
    
    parser.add_argument('--looseless_compressor_name', '-l',
                        type=str, help = '', default="huffman")

    args = parser.parse_args()
    
    return args



if __name__ == "__main__":
    args = parse_args()

    # not used
    device = torch.device(args.device)

    encoder = get_encoder(args.model_name, args.B)
    encoder.eval()
    
    if args.looseless_compressor_name == "huffman":
        looseless_compressor = Huffman()
    else:
        raise NotImplementedError(
            "{args.looseless_compressor_name} is not emplemented")

    encoder_pipeline(
        encoder, args.image_path, args.B, args.compressor_state_path,
        args.encode_output_path, looseless_compressor)