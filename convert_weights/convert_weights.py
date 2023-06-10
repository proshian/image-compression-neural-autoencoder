import sys; sys.path.append('..')

import os
from typing import List

import torch
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import skimage
from PIL import Image
from torchvision import transforms


import deprecated_models as dm
import models as m

def get_old_model(src_w_path,  device = None, up_func_name = "upsample", B = 6):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet_autoencoder = dm.create_resnet_autoencoder(
        resnet18(),
        up_func_name = up_func_name,
        B=B,
    ).to(device)
    resnet_autoencoder.load_state_dict(torch.load(src_w_path, map_location=torch.device(device)))
    return resnet_autoencoder

def get_new_model(device = None, up_func_name = "upsample", B = 6):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet_autoencoder = m.create_resnet_autoencoder(
        resnet18(),
        up_func_name = up_func_name,
        B=B,
    ).to(device)
    return resnet_autoencoder

def load_state_from_depricated_model(old_model, new_model):
    new_model.decoder.load_state_dict(old_model.decoder.state_dict()) 
    new_model.encoder.backbone.load_state_dict(old_model.encoder.state_dict()) 

def convert_weights(src_w_path, dest_w_path):
    old_model = get_old_model(src_w_path) 
    new_model = get_new_model()
    load_state_from_depricated_model(old_model, new_model)
    torch.save(new_model.state_dict(), dest_w_path)