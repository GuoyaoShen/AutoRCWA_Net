import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

import sys



def train_model(train_dataloader,
                test_dataloader,
                optimizer,
                loss,
                net,
                device,
                NUM_EPOCH=5,
                early_stop_parser=[True, 1000, 100]):
    '''
    Train the model.
    :param train_dataloader: training dataloader.
    :param test_dataloader: test dataloader.
    :param optimizer: optimizer.
    :param loss: loss function object.
    :param net: network object.
    :param device: device, gpu or cpu.
    :param NUM_EPOCH: number of epoch, default to 5.
    :param early_stop_parser: list [use_early_stop, num_epoch_force, num_epoch_tolerance].
        use_early_stop: flag, whether to use early stop.
        num_epoch_force: number of forced epoch to run, the least number of epoch for training.
        num_epoch_tolerance: tolerance number before early stopping.
    :return: /
    '''
    net = net.to(device)
    net.train()

    use_early_stop = early_stop_parser[0]
    num_epoch_force = early_stop_parser[1]
    num_epoch_tolerance = early_stop_parser[2]

    for i in range(NUM_EPOCH):
        running_loss = 0.0
        for idx, data in enumerate(train_dataloader):
            X, y = data

            X = X.to(device)
            y = y.to(device)

            y_pred = net(X)

            optimizer.zero_grad()
            loss_train = loss(y_pred, y)
            loss_train.backward()
            optimizer.step()
            running_loss += loss_train.item()

        loss_train = running_loss / len(train_dataloader)

        # === print a progressbar
        sys.stdout.write('\r')
        sys.stdout.write("### EPOCH [%d/%d] || AVG LOSS: %.6E" % (i + 1, NUM_EPOCH, loss_train))
        sys.stdout.flush()

        # test model for each epoch
        loss_test = test_model(test_dataloader, loss, net, device, i, NUM_EPOCH)

        # early stopping
        if use_early_stop:
            # monitor each epoch
            if i == 0:  # 1st epoch
                metric_best = loss_test  # record the best metric performance in history
                model_best = net  # record the best model
                num_patience_count = 0  # count the number of worse case for early stopping
            else:  # other epochs
                if loss_test <= metric_best:  # if better performance
                    metric_best = loss_test
                    model_best = net
                    num_patience_count = 0
                else:
                    num_patience_count += 1
            # early stop trigger
            if num_patience_count > num_epoch_tolerance and i >= num_epoch_force:
                break

        else:
            model_best = net
    sys.stdout.write('\n')

    return loss_train, loss_test, model_best  # return loss for the last epoch



def test_model(test_dataloader,
               loss,
               net,
               device,
               i,
               NUM_EPOCH):
    net = net.to(device)
    net.eval()

    running_loss = 0.0
    for idx, data in enumerate(test_dataloader):
        X, y = data

        X = X.to(device)
        y = y.to(device)

        y_pred = net(X)

        loss_train = loss(y_pred, y)
        running_loss += loss_train.item()
    loss_test = running_loss / len(test_dataloader)

    # === print a progressbar
    sys.stdout.write('\r')
    sys.stdout.write("### EPOCH [%d/%d] || TEST LOSS: %.6E" % (i + 1, NUM_EPOCH, loss_test))
    sys.stdout.flush()

    return loss_test