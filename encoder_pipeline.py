from typing import List
import numbers
import os

import torch
from torchvision import transforms
from torchvision.transforms.functional import pad
from PIL import Image
import skimage.io

from looseless_compressors import LooselessCompressor, Huffman


class PadDivisibleBy32(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return pad(img, self._get_padding(img), self.padding_mode, self.fill)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)
    
    @staticmethod
    def _get_padding(image):  
        ch, w, h = image.shape

        w_pad = w%32
        h_pad = h%32

        l_pad = w_pad//2 + w_pad%2
        r_pad = w_pad//2

        t_pad = h_pad//2 + h_pad%2
        b_pad = h_pad//2 
                    
        return int(l_pad), int(t_pad), int(r_pad), int(b_pad)



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
imagenet_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


inference_data_transform = transforms.Compose([
    # If we use resnet autoencoder the image size should be a multiple of 32.
    # Otherwise the decoded shape would be different from original.
    # PadDivisibleBy32(),
    transforms.ToTensor(),
    imagenet_normalize
])


def img_path_to_model_input(img_path: str, inference_data_transform = inference_data_transform):
    image = Image.fromarray(skimage.io.imread(img_path))
    image = inference_data_transform(image)
    return image


def quantize(encoder_out: torch.Tensor, B: int):
    quantized = torch.round(encoder_out * 2**B)
    return quantized.type(torch.int8)


def save_binary_string_to_file(binary_string, filename):
    """
    Converts a string of 0 and 1 to a bytearray and writes it into file
    """
    # We add 1 to the end to be able to find the end of the string when decoding.
    # Otherwise it won't be possible to know if the zeros at the end are padding
    # or correct data.
    binary_string += '1'

    if len(binary_string) % 8 != 0:
        # Pad the string with 0s to the nearest multiple of 8
        padding = 8 - len(binary_string) % 8
        binary_string += '0' * padding
    binary_bytes = bytearray(
        int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8))

    # Write the bytes to the file
    with open(filename, 'wb') as f:
        f.write(binary_bytes)


def encoder_pipeline(encoder, img_path: str, B: int,
                     compressor_state_path: str = None,
                     compressed_img_path: str = None,
                     looseless_compressor: LooselessCompressor = Huffman()):
    if compressed_img_path is None:
        compressed_img_path = f"{os.path.splitext(img_path)[0]}.neural"
    
    if compressor_state_path is None:
        compressor_state_path = f"{img_path}__CS__B_{B}"

    encoder.eval()
    img = img_path_to_model_input(img_path)
    unsqueezed = img.unsqueeze(0)
    encoder_out = encoder(unsqueezed)
    quantized = quantize(encoder_out, B)
    # flat_out = quantized.flatten().cpu().detach().numpy()
    flat_out = [int(x) for x in quantized.flatten()]
    looseless_compressor.init_from_sequence(flat_out)
    looseless_compressor.save_state_to_file(compressor_state_path)
    encoded = looseless_compressor.encode(flat_out)
    save_binary_string_to_file(encoded, compressed_img_path)
    return encoder_out, encoded  # for debug purposes only