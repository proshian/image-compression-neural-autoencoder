class AutoencoderAlexNet(NeuralImageCompressor):
    def __init__(self, B: int = 1, normalising_activation: nn.Module = nn.Sigmoid()):
        encoder = alexnet(weights=AlexNet_Weights.DEFAULT).features
        decoder = self._get_decoder(encoder)
        super().__init__(encoder, decoder, normalising_activation, B)
    
    @staticmethod
    def _get_decoder(encoder: nn.Sequential):
        decoder_modules = []

        for module in reversed(encoder):
            if isinstance(module, nn.Conv2d):
                trans_conv = nn.ConvTranspose2d(
                    in_channels=module.out_channels,
                    out_channels=module.in_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding
                )
                decoder_modules.append(trans_conv)
            elif isinstance(module, nn.ReLU):
                decoder_modules.append(nn.ReLU(inplace=True))
            elif isinstance(module, nn.MaxPool2d):
                # We can use MaxUnpool if we're going to save MaxPool indicies
                # unpool = nn.MaxUnpool2d(
                #     kernel_size=module.kernel_size,
                #     stride=module.stride,
                #     padding=module.padding
                # )
                upsample = nn.Upsample(
                    scale_factor=2,
                    mode='nearest'
                )
                decoder_modules.append(upsample)
            else:
                raise ValueError(f"unexpected module {module}")
            
        decoder = nn.Sequential(*decoder_modules)
        return decoder
    




def simpler_decoder_8x_upsample_constructor(in_chan_num):
    out_chan_nums = [512, 128, 3]
    upsample_factors = [4, 4, 2]

    decoder_modules = []

    for out_chan_num, upsample_factor in zip(out_chan_nums, upsample_factors):
        decoder_modules.append(
            nn.Sequential(
                nn.Upsample(
                    scale_factor=upsample_factor,
                    mode='nearest'
                ),
                nn.Conv2d(in_channels=in_chan_num, out_channels=out_chan_num, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )
        )

        in_chan_num = out_chan_num
    
    return nn.Sequential(*decoder_modules)





class AutoEncoderResNet(NeuralImageCompressor):
    def __init__(self, resnet: ResNet, decoder: nn.Module,
                 normalising_activation: nn.Module = nn.Sigmoid(), B: int = 1):
        encoder = self._get_resnet_encoder(resnet)
        super().__init__(encoder, decoder, normalising_activation, B)
    
    @staticmethod
    def _get_resnet_encoder(resnet: ResNet):
        return nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4)