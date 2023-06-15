from typing import List, Tuple
import os

import torch
import numpy as np
from PIL import Image

from trained_models import get_decoder
from looseless_compressors import LooselessCompressor, Huffman


def denormalize(img: torch.Tensor, means: List[float], stds: List[float]):
    result = torch.zeros_like(img)
    for i, (chan, mean, std) in enumerate(zip(img, means, stds)):
        result[i] = chan * std + mean
    return result

def denormalize_imagenet(img: torch.Tensor):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return denormalize(img, imagenet_mean, imagenet_std)


def decode_binary_file(filename):
    with open(filename, 'rb') as f:
        binary_data = f.read()
    str_with_padding = ''.join(
        format(byte, '08b') for byte in binary_data)
    pad_end = str_with_padding.rfind('1')
    return str_with_padding[:pad_end]


def get_quant_error_normal(shape: Tuple[int, ...], B: int) -> torch.Tensor:
    mean = torch.full(shape, -0.5)
    std = torch.full(shape, 0.5)
    quan_err = 0.5**B * torch.normal(mean = mean, std = std)
    return quan_err


def get_quant_error_uniform(shape: Tuple[int, ...], B: int) -> torch.Tensor:
    min_noise = -1
    max_noise = 1
    quan_err = 0.5**B * (max_noise - min_noise) * (torch.rand(shape)) + min_noise
    return quan_err



def decoder_pipeline(decoder, compressed_img_path: str, B: int,
                     compressor_state_path: str = None,
                     decoder_output_path: str = None,
                     looseless_compressor: LooselessCompressor = Huffman()):
    compressed_img_path_no_ext = os.path.splitext(compressed_img_path)[0]
    if decoder_output_path is None:
        decoder_output_path = f"{compressed_img_path_no_ext}_decoder_output.bmp"
    
    if compressor_state_path is None:
        compressor_state_path = f"{compressed_img_path_no_ext}_state.json"


    llc = looseless_compressor
    
    decoder.eval()

    binary_string = decode_binary_file(compressed_img_path)
    llc.init_from_file(compressor_state_path)
    quantized = torch.tensor(llc.decode(binary_string))
    encoder_output_flat = quantized / 2**B
    height = width = int((len(encoder_output_flat)/decoder.in_channels)**0.5)

    encoder_output = encoder_output_flat.reshape(
        1, decoder.in_channels, height, width)
    
    # decoded_tensor_imagenet_norm = decoder(encoder_output.type(torch.float32))
    decoded_tensor_imagenet_norm = decoder(
        encoder_output.type(torch.float32) + get_quant_error_normal(encoder_output.shape, B))

    decoded_tensor = denormalize_imagenet(
        decoded_tensor_imagenet_norm.squeeze(0))
    
    np_decoded_img = decoded_tensor.cpu().detach().numpy().transpose(1,2,0)
    np_decoded_img = np.clip(np_decoded_img, 0, 1)
    np_decoded_img = (np_decoded_img*255).astype(np.uint8)
    pil_img = Image.fromarray(np_decoded_img, 'RGB')
    pil_img.save(decoder_output_path, "BMP")
    return np_decoded_img