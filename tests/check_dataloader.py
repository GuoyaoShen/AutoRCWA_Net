import numpy as np

from utils.data_utils import generate_dataset


PATH_ZIPSET = './data/fRT_1.npz'
# dataset, dataloader = generate_dataset(PATH_ZIPSET)
dataset, dataloader = generate_dataset(PATH_ZIPSET, idx_pick_param=[], BTSZ=1)

print(len(dataset))
print(len(dataloader))