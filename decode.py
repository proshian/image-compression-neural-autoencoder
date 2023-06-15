import argparse

import torch

from decoder_pipeline import decoder_pipeline
from looseless_compressors import Huffman 
from trained_models import get_decoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='encode',
        description='encodes images')

    parser.add_argument('--compressed_img_path', '-i', type=str,
                        help = 'compressed image to be decoded')
    
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
    
    parser.add_argument('--decode_output_path', '-o',
                        type=str, help = '', default=None)
    
    parser.add_argument('--looseless_compressor_name', '-l',
                        type=str, help = '', default="huffman")

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()

    # not used
    device = torch.device(args.device)

    decoder = get_decoder(args.model_name, args.B)
    decoder.eval()
    
    if args.looseless_compressor_name == "huffman":
        looseless_compressor = Huffman()
    else:
        raise NotImplementedError(
            "{args.looseless_compressor_name} is not emplemented")

    decoder_pipeline(
        decoder, args.compressed_img_path, args.B, args.compressor_state_path,
        args.decode_output_path, looseless_compressor)