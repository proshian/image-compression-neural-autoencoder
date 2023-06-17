import torch.nn as nn
from torchvision.models import resnet18
import torch

from models import create_resnet_encoder, SimpleResidualDecoder32x_ABS


resnet18x32__512ch__abs__sigmoid__no_last_activation = {
    "creation": {
        "encoder": lambda: create_resnet_encoder(
            resnet18(), nn.Identity(), nn.Sigmoid()), 
        "decoder": lambda: SimpleResidualDecoder32x_ABS(
            512, "upsample", nn.Sigmoid())
    },
    "weights": {
        "B6": {
            "encoder": 'weights/resnet_autoencoder_abs/' \
                'encoder__resnet_autoencoder__512x16x16__upsample__B_6__' \
                '66_epochs__2023-06-10T00_39.pt',
            "decoder": 'weights/resnet_autoencoder_abs/' \
                'decoder__resnet_autoencoder__512x16x16__upsample__B_6__' \
                '66_epochs__2023-06-10T00_39.pt',
        }
    }
}


resnet18x32__512ch__abs__relu__no_last_activation__normal_noise = {
    "creation": {
        "encoder": lambda: create_resnet_encoder(
            resnet18(), nn.Identity(), nn.Sigmoid()), 
        "decoder": lambda: SimpleResidualDecoder32x_ABS(
            512, "upsample", nn.ReLU()),
    },
    "weights": {
        "B1": {
            "encoder": 'weights/resnet_autoencoder_abs/' \
                'encoder__resnet_autoencoder__512x16x16__upsample__' \
                'B_1__6_epochs__last_relu__2023-06-12T19_54.pt',
            "decoder": 'weights/resnet_autoencoder_abs/' \
                'decoder__resnet_autoencoder__512x16x16__upsample__' \
                'B_1__6_epochs__last_relu__2023-06-12T19_54.pt',
        },
        "B2": {
            "encoder": 'weights/resnet_autoencoder_abs/' \
                'encoder__resnet_autoencoder__512x16x16__upsample__' \
                'B_2__6_epochs__last_relu__2023-06-08T12_35.pt',
            "decoder": 'weights/resnet_autoencoder_abs/' \
                'decoder__resnet_autoencoder__512x16x16__upsample__' \
                'B_2__6_epochs__last_relu__2023-06-08T12_35.pt',
        },
        "B4": {
            "encoder": 'weights/resnet_autoencoder_abs/' \
                'encoder__resnet_autoencoder__512x16x16__upsample__' \
                'B_4__6_epochs__last_relu__2023-06-12T19_29.pt',
            "decoder": 'weights/resnet_autoencoder_abs/' \
                'decoder__resnet_autoencoder__512x16x16__upsample__' \
                'B_4__6_epochs__last_relu__2023-06-12T19_29.pt',
        },
        "B6": {
            "encoder": 'weights/resnet_autoencoder_abs/'
                'encoder__resnet_autoencoder__512x16x16__upsample__' \
                'B_6__13_epochs__last_relu__2023-06-14T01_07.pt',
            "decoder": 'weights/resnet_autoencoder_abs/'
                'decoder__resnet_autoencoder__512x16x16__upsample__' \
                'B_6__13_epochs__last_relu__2023-06-14T01_07.pt',
        },
    }
}


resnet18x32__512ch__abs__relu__no_last_activation = {
    "creation": {
        "encoder": lambda: create_resnet_encoder(
            resnet18(), nn.Identity(), nn.Sigmoid()), 
        "decoder": lambda: SimpleResidualDecoder32x_ABS(
            512, "upsample", nn.ReLU()),
    },
    "weights": {
        "B6": {
            "encoder": 'weights/resnet_autoencoder_abs_uniform/' \
                'encoder__resnet_autoencoder__512x16x16__upsample__' \
                'B_6__3_epochs__last_relu__2023-06-17T20_48.pt',
            "decoder": 'weights/resnet_autoencoder_abs_uniform/' \
                'decoder__resnet_autoencoder__512x16x16__upsample__' \
                'B_6__3_epochs__last_relu__2023-06-17T20_48.pt',
        },
    }
}



resnet18x32__512ch__abs__relu__no_last_activation__uniform_noise = {
    "creation": {
        "encoder": lambda: create_resnet_encoder(
            resnet18(), nn.Identity(), nn.Sigmoid()), 
        "decoder": lambda: SimpleResidualDecoder32x_ABS(
            512, "upsample", nn.ReLU()),
    },
    "weights": {
        "B2": {
            "encoder": 'weights/resnet_autoencoder_abs/' \
                'encoder__resnet_autoencoder__512x16x16__upsample__' \
                'B_2__6_epochs__uniform__last_relu__2023-06-14T14_18.pt',
            "decoder": 'weights/resnet_autoencoder_abs/' \
                'decoder__resnet_autoencoder__512x16x16__upsample__' \
                'B_2__6_epochs__uniform__last_relu__2023-06-14T14_18.pt',
        }
    }
}





model_dicts = {
    "resnet18x32__512ch__abs__sigmoid__no_last_activation": resnet18x32__512ch__abs__sigmoid__no_last_activation,
    "resnet18x32__512ch__abs__relu__no_last_activation": resnet18x32__512ch__abs__relu__no_last_activation__normal_noise,
    "resnet18x32__512ch__abs__relu__no_last_activation__uniform_noise": resnet18x32__512ch__abs__relu__no_last_activation__uniform_noise,
    "resnet18x32__512ch__abs__relu__no_last_activation": resnet18x32__512ch__abs__relu__no_last_activation,
    "default": resnet18x32__512ch__abs__sigmoid__no_last_activation,
}


def get_encoder(model_name: str, B: int):
    model_dict = model_dicts[model_name]
    encoder = model_dict["creation"]["encoder"]()
    weights = model_dict["weights"][f"B{B}"]["encoder"]
    encoder.load_state_dict(torch.load(weights))
    return encoder

def get_decoder(model_name: str, B: int):
    model_dict = model_dicts[model_name]
    decoder = model_dict["creation"]["decoder"]()
    weights = model_dict["weights"][f"B{B}"]["decoder"]
    decoder.load_state_dict(torch.load(weights))
    return decoder