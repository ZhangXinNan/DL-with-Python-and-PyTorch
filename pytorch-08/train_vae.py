
import torch

from load_data import *
from net_vae import VAE


# torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)






