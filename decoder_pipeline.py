from typing import List, Tuple, Callable, Optional
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

def get_zero_noise(shape: Tuple[int, ...], B: int) -> torch.Tensor:
    return torch.zeros(shape)

def torch_img_to_np_img(tensor_img: torch.Tensor) -> np.ndarray:
    np_decoded_img = tensor_img.cpu().detach().numpy().transpose(1,2,0)
    np_decoded_img = np.clip(np_decoded_img, 0, 1)
    np_decoded_img = (np_decoded_img*255).astype(np.uint8)
    return np_decoded_img


# def binarystring_to_decoder_input(binary_string: str, B: int,
#                                   looseless_compressor: LooselessCompressor,
#                                   decoder_in_channels: int) -> torch.Tensor:
#     quantized = torch.tensor(looseless_compressor.decode(binary_string))
#     dequantized = quantized / 2**B
#     height = width = int((len(dequantized)/decoder_in_channels)**0.5)

#     return dequantized.reshape(
#         1, decoder_in_channels, height, width)


def decoder_pipeline(decoder, compressed_img_path: str, B: int,
                     compressor_state_path: str,
                     decoder_output_path: Optional[str] = None,
                     looseless_compressor: LooselessCompressor = Huffman(),
                     get_noise: Callable = get_quant_error_normal):    
    decoder.eval()

    binary_string = decode_binary_file(compressed_img_path)
    looseless_compressor.init_from_file(compressor_state_path)
    quantized = torch.tensor(looseless_compressor.decode(binary_string))
    dequantized = quantized / 2**B
    height = width = int((len(dequantized)/decoder.in_channels)**0.5)

    encoder_output = dequantized.reshape(
        1, decoder.in_channels, height, width)
    
    # decoded_tensor_imagenet_norm = decoder(encoder_output.type(torch.float32))
    decoded_tensor_imagenet_norm = decoder(
        encoder_output.type(torch.float32) + get_noise(encoder_output.shape, B))

    decoded_tensor = denormalize_imagenet(
        decoded_tensor_imagenet_norm.squeeze(0))
    
    np_decoded_img = torch_img_to_np_img(decoded_tensor)
    
    if decoder_output_path is not None:
        pil_img = Image.fromarray(np_decoded_img, 'RGB')
        pil_img.save(decoder_output_path, "BMP")
    return np_decoded_img