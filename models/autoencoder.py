import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class NetAutoencoder(nn.Module):
    def __init__(self, dim_in, hidden_dim=10):
        super(NetAutoencoder, self).__init__()
        self.dim_in = dim_in
        self.hidden_dim = hidden_dim

        self.fc_encoder = nn.Sequential(
        nn.Linear(self.dim_in, 1500),
        nn.ReLU(True),

        nn.Linear(1500, 500),
        nn.ReLU(True),

        nn.Linear(500, 100),
        nn.ReLU(True),

        nn.Linear(100, self.hidden_dim),
        nn.ReLU(True),
        )

        self.fc_decoder = nn.Sequential(
        nn.Linear(self.hidden_dim, 100),
        nn.ReLU(True),

        nn.Linear(100, 500),
        nn.ReLU(True),

        nn.Linear(500, 1500),
        nn.ReLU(True),

        nn.Linear(1500, self.dim_in),
        nn.Sigmoid(),
        )


    def forward(self, x):
        hidden_out = self.fc_encoder(x)
        out = self.fc_decoder(hidden_out)

        return out, hidden_out

    def encode(self, x_orig):
        x_latent = self.fc_encoder(x_orig)

        return x_latent

    def decode(self, x_latent):
        x_reconstruct = self.fc_decoder(x_latent)

        return x_reconstruct



class NetDense(nn.Module):
    '''
    From design params to latent vectors.
    '''
    def __init__(self, dim_in, dim_out):
        super(NetDense, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.fc = nn.Sequential(
        nn.Linear(self.dim_in, 50),
        nn.ReLU(True),

        nn.Linear(50, 200),
        nn.ReLU(True),

        nn.Linear(200, 400),
        nn.ReLU(True),

        nn.Linear(400, 200),
        nn.ReLU(True),

        nn.Linear(200, 100),
        nn.ReLU(True),

        nn.Linear(100, 50),
        nn.ReLU(True),

        nn.Linear(50, self.dim_out),
        nn.ReLU(True),
        )


    def forward(self, x):
        out = self.fc(x)

        return out