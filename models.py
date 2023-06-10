from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import (alexnet,
                                AlexNet_Weights,
                                resnet18,
                                ResNet18_Weights,
                                ResNet,
                                resnet101)


############### Blocks

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,
                 last_activation: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            last_activation
        )
        
    def forward(self, img: torch.Tensor):
        return self.double_conv(img)
    
    
class SimpleResidualUpsampleDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 mid_channels=None, up_func_name = "upsample",
                 last_activation = nn.ReLU(inplace = True)):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = DoubleConv(
            in_channels, out_channels, last_activation = last_activation)
        self.upscale = self.make_upscaler(
            in_channels,out_channels, up_func_name)
        self.conv_to_match_dims = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2)

    @staticmethod
    def make_upscaler(in_channels, out_channels, up_func_name):
        if up_func_name == "upsample":
            return nn.Upsample(
                scale_factor=2,
                mode='nearest')
        
        elif up_func_name == "deconv":
            return nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2)
        
        else:
            raise ValueError(f"unknown upscaler {up_func_name}")
        
        
    def forward(self, x):
        skip_connection = self.conv_to_match_dims(x)
        out = self.upscale(x)
        out = self.double_conv(out)
        out = out + skip_connection
        return out
    

############### Decoders


class SequentialDecoder32x(nn.Module):
    def __init__(self, in_channels, up_func_name = "deconv",
                 last_activation: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        out_chan_nums = [512, 256, 128, 64, 3]
        
        decoder_modules = []

        for out_channels in out_chan_nums[:-1]:
            # мб было бы красивее здесь создавать upscaler
            decoder_modules.append(
                nn.Sequential(
                    self.make_upscaler(in_channels, in_channels, up_func_name),
                    DoubleConv(in_channels=in_channels, out_channels=out_channels),
                )
            )
            in_channels = out_channels

        decoder_modules.append(
            nn.Sequential(
                self.make_upscaler(in_channels, in_channels, up_func_name),
                DoubleConv(in_channels=in_channels,
                           out_channels=out_chan_nums[-1],
                           last_activation=last_activation),
            )
        )

        self.decoder = nn.Sequential(*decoder_modules)
    
    @staticmethod
    def make_upscaler(in_channels, out_channels, up_func_name):
        if up_func_name == "upsample":
            return nn.Upsample(
                scale_factor=2,
                mode='nearest'
            )
        
        elif up_func_name == "deconv":
            return nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2)
        
        else:
            raise ValueError(f"unknown upscaler {up_func_name}")
    
    def forward(self, img: torch.Tensor):
        return self.decoder(img)



class SimpleResidualDecoder32x(nn.Module):
    def __init__(self, in_channels, up_func_name = "upsample",
                 last_activation: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        out_chan_nums = [512, 256, 128, 64, 3]
        
        decoder_modules = []

        for out_channels in out_chan_nums[:-1]:
            # мб было бы красивее здесь создавать upscaler
            decoder_modules.append(
                SimpleResidualUpsampleDoubleConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    up_func_name=up_func_name)
            )
            in_channels = out_channels

        decoder_modules.append(
            SimpleResidualUpsampleDoubleConv(
                in_channels=in_channels,
                out_channels=out_chan_nums[-1],
                up_func_name=up_func_name,
                last_activation=last_activation)
        )

        self.decoder = nn.Sequential(*decoder_modules)
    
    def forward(self, img: torch.Tensor):
        return self.decoder(img)



# def simple_decoder_32x_upsample_constructor(in_chan_num):
#     out_chan_nums = [512, 256, 128, 64, 3]

#     decoder_modules = []

#     for out_chan_num in out_chan_nums:
#         decoder_modules.append(
#             nn.Sequential(
#                 nn.Upsample(
#                     scale_factor=2,
#                     mode='nearest'
#                 ),
#                 nn.Conv2d(in_channels=in_chan_num,
#                           out_channels=out_chan_num,
#                           kernel_size=3, stride=1, padding=1),
#                 nn.ReLU()
#             )
#         )

#         in_chan_num = out_chan_num
    
#     return nn.Sequential(*decoder_modules)




############ Encoder backbones

def create_resnet_encoder_backbone(resnet: ResNet):
    return nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4)


############# Encoders

class Encoder(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 feature_extraction: nn.Module,
                 normalising_activation: nn.Module = nn.Sigmoid(),
                 freeze_backbone = False
                 ):
        super().__init__()
        self.backbone = backbone
        self.feature_extraction = feature_extraction
        self.normalising_activation = normalising_activation
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.backbone(x)
        out = self.feature_extraction(out)
        out = self.normalising_activation(out)
        return out



def create_resnet_encoder(
        resnet: ResNet,
        feature_extraction = nn.Identity(),
        normalising_activation: nn.Module = nn.Sigmoid()):
    return Encoder(
        backbone = create_resnet_encoder_backbone(resnet),
        feature_extraction = feature_extraction,
        normalising_activation = normalising_activation
    )



############# NeuralImageCompressors (Autoencoders)

class NeuralImageCompressor(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: nn.Module,
                 B: int = 1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.B = B
            
    def _get_quantization_error(self, shape: Tuple[int, ...]):
        mean = torch.full(shape, -0.5)
        std = torch.full(shape, 0.5)
        quan_err = 0.5**self.B * torch.normal(mean = mean, std = std)
        return quan_err
    
    def forward(self, x):
        out = self.encoder(x)
        quant_err = self._get_quantization_error(out.shape).to(out.device)
        out = out + quant_err
        out = self.decoder(out)
        return out


def create_resnet_autoencoder(resnet: ResNet, autoenc_feat_extract: nn.Module = nn.Identity(),
                              decoder = None, decoder_in_channels: int = 512,
                              normalising_activation: nn.Module = nn.Sigmoid(), B: int = 16,
                              up_func_name = "upsample"):
    resnet_encoder = create_resnet_encoder(
        resnet, autoenc_feat_extract, normalising_activation)
    if decoder is None:
        decoder = SimpleResidualDecoder32x(decoder_in_channels, up_func_name = up_func_name)
    resnet_autoencoder = NeuralImageCompressor(resnet_encoder, decoder, B)
    return resnet_autoencoder


## AutoencoderAlexNet was not used
# class AutoencoderAlexNet(NeuralImageCompressor):
#     def __init__(self, B: int = 1, normalising_activation: nn.Module = nn.Sigmoid()):
#         encoder = alexnet(weights=AlexNet_Weights.DEFAULT).features
#         decoder = self._get_decoder(encoder)
#         super().__init__(encoder, decoder, normalising_activation, B)
    
#     @staticmethod
#     def _get_decoder(encoder: nn.Sequential):
#         decoder_modules = []

#         for module in reversed(encoder):
#             if isinstance(module, nn.Conv2d):
#                 trans_conv = nn.ConvTranspose2d(
#                     in_channels=module.out_channels,
#                     out_channels=module.in_channels,
#                     kernel_size=module.kernel_size,
#                     stride=module.stride,
#                     padding=module.padding
#                 )
#                 decoder_modules.append(trans_conv)
#             elif isinstance(module, nn.ReLU):
#                 decoder_modules.append(nn.ReLU(inplace=True))
#             elif isinstance(module, nn.MaxPool2d):
#                 # We can use MaxUnpool if we're going to save MaxPool indicies
#                 # unpool = nn.MaxUnpool2d(
#                 #     kernel_size=module.kernel_size,
#                 #     stride=module.stride,
#                 #     padding=module.padding
#                 # )
#                 upsample = nn.Upsample(
#                     scale_factor=2,
#                     mode='nearest'
#                 )
#                 decoder_modules.append(upsample)
#             else:
#                 raise ValueError(f"unexpected module {module}")
            
#         decoder = nn.Sequential(*decoder_modules)
#         return decoder
