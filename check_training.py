'''
Run a test training progress.
This script also works as a template of how to use the packages.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

# import local packages
from utils.data_utils import generate_dataset
from utils.training_utils import train_model, test_model
from models.dense_convtranspose1d import DenseConvTranspose1D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ================== Data Loading ==================
PATH_ZIPSET = './data/fRT_1.npz'
# idx_pick_param = [0,2,3,4,5,6]
# dataset, dataloader = generate_dataset(PATH_ZIPSET, idx_pick_param)
dataset, dataloader = generate_dataset(PATH_ZIPSET, idx_pick_param=[], BTSZ=1)
dataset_train, dataset_test = random_split(dataset, [9, 1])
print('LEN dataset_train:', len(dataset_train))
print('LEN dataset_test:', len(dataset_test))
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
print('LEN dataloader_train:', len(dataloader_train))
print('LEN dataloader_test:', len(dataloader_test))


# ================== Import Network ==================
net_dense = DenseConvTranspose1D(6,893)
net_dense = net_dense.to(device)


# ================== Train Network ==================
learning_rate = 1e-3
optimizer = torch.optim.Adam(net_dense.parameters(), lr=learning_rate, weight_decay=0.0, amsgrad=True)
criteon = nn.MSELoss()
scheduler = None
train_model(dataloader, dataloader_test, optimizer, criteon, net_dense, device, NUM_EPOCH=4000, scheduler=scheduler)