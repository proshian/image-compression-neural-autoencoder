from torchvision.transforms.functional import pad
from torchvision import transforms
import numpy as np
import numbers

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
        return F.pad(img, self._get_padding(img), self.padding_mode, self.fill)
    
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