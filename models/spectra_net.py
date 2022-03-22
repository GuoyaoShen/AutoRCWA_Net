import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SquareLayer(nn.Module):
    '''
    This layer works as a function of: y = xAx+(Bx)+C
    '''
    def __init__(self, dim_in, dim_out):
        super(SquareLayer, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=False)
        self.bilinear = nn.Bilinear(dim_in, dim_in, dim_out)

    def forward(self, x):
        out = self.linear(x) + self.bilinear(x,x)
        return  out


class SpectraNet(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SpectraNet, self).__init__()
        self.dim_out = dim_out

        self.tk = 20
        self.dim_mid1 = 500  # dimension in the middle before conv layer

        self.fc = nn.Sequential(
        nn.Linear(dim_in, 50),
        nn.ReLU(True),
        nn.Linear(50, 100),
        nn.ReLU(True),
        nn.Linear(100, 200),
        nn.ReLU(True),
        nn.Linear(200, self.dim_mid1-self.tk+1),
        nn.ReLU(True),
        )

        self.convt1 = nn.Sequential(
            nn.ConvTranspose1d(1, 2, kernel_size=self.tk, stride=1),
            nn.BatchNorm1d(2),
            nn.ReLU(True),
        )

        self.fc_sampling = nn.Sequential(
        nn.Linear(self.dim_mid1, 1000),
        nn.ReLU(True),
        nn.Linear(1000, 1500),
        nn.ReLU(True),
        nn.Linear(1500, self.dim_out),
        # nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.fc(x)
        out = out.view(out.shape[0], 1, -1)
        out = self.convt1(out)
        out = self.fc_sampling(out)

        return out