import torch
from torchvision.models import resnet18

from models import create_resnet_autoencoder


if __name__ == "__main__":
    B = 2
    LOAD_PATH = "weights\\residual_decoder__upsample__B_6__63_epochs_2023-06-08T05_56.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    up_func_name = "upsample"
    resnet_autoencoder = create_resnet_autoencoder(
        resnet18,
        up_func_name = up_func_name,
        B=B,
    ).to(device)

    resnet_autoencoder.load_state_dict(torch.load(LOAD_PATH, map_location=torch.device(device)))
    optimizer = torch.optim.Adam(resnet_autoencoder.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss() 

    epochs = 6