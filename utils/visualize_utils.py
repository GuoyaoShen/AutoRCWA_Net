import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visualize_trained_net(dataloader_test, net, device, size_fig=(21, 12)):
    '''
    Visualize trained network's predicted spectra on the test dataloader.
    '''
    for idx, data in enumerate(dataloader_test):
        X, y = data
        X = X.to(device)
        y = y.to(device)
        y_pred = net(X)  # [N_eg,2,N_freq]

        y_np = y.cpu().detach().numpy()
        y_pred_np = y_pred.cpu().detach().numpy()
        if idx == 0:
            y_test_np = y_np
            y_test_pred_np = y_pred_np
        else:
            y_test_np = np.concatenate((y_test_np, y_np), axis=0)
            y_test_pred_np = np.concatenate((y_test_pred_np, y_pred_np), axis=0)

        if y_test_pred_np.shape[0] >= 4:
            break
    print('y_test_np.shape:', y_test_np.shape)
    print('y_test_pred_np.shape:', y_test_pred_np.shape)

    fig1 = plt.figure(1, figsize=size_fig)
    fig1.suptitle('Training result on test set: freq-R')
    for idx_fig in range(4):
        plt.subplot(2, 2, idx_fig + 1)
        plt.scatter(np.arange(y_test_np.shape[2]), y_test_np[idx_fig, 0, :], c='r')
        plt.scatter(np.arange(y_test_pred_np.shape[2]), y_test_pred_np[idx_fig, 0, :], c='b')
        plt.legend(('true', 'prediction'))
        plt.title('sample idx: ' + str(idx_fig))

    fig2 = plt.figure(2, figsize=size_fig)
    fig2.suptitle('Training result on test set: freq-T')
    for idx_fig in range(4):
        plt.subplot(2, 2, idx_fig + 1)
        plt.scatter(np.arange(y_test_np.shape[2]), y_test_np[idx_fig, 1, :], c='r')
        plt.scatter(np.arange(y_test_pred_np.shape[2]), y_test_pred_np[idx_fig, 1, :], c='b')
        plt.legend(('true', 'prediction'))
        plt.title('sample idx: ' + str(idx_fig))
    plt.show()


def visualize_spectra_search(params_pick, R_pick, T_pick, size_fig=(21, 12)):
    '''
    Visualize picked (top 4) spectra search results.
    R_pick, T_pick: shape [N_pick, N_freq]
    '''
    fig1 = plt.figure(1, figsize=size_fig)
    fig1.suptitle('Spectra search result: freq-R')
    for idx_fig in range(4):
        plt.subplot(2, 2, idx_fig + 1)
        plt.scatter(np.arange(R_pick.shape[1]), R_pick[idx_fig, :], c='r')
        plt.title(str(idx_fig + 1) + 'th best match')

    fig2 = plt.figure(2, figsize=size_fig)
    fig2.suptitle('Spectra search result: freq-T')
    for idx_fig in range(4):
        plt.subplot(2, 2, idx_fig + 1)
        plt.scatter(np.arange(T_pick.shape[1]), T_pick[idx_fig, :], c='r')
        plt.title(str(idx_fig + 1) + 'th best match')
    plt.show()


def visualize_learned_and_rcwa(param_pick, R_pick, T_pick, R_simu, T_simu, size_fig=(21, 12)):
    '''
    Visualize comparisons (top 4) between predicted spectra from DL and rcwa simulation spectra.
    R_pick, T_pick, R_simu, T_simu: shape [N_pick, N_freq]
    '''
    fig1 = plt.figure(1, figsize=size_fig)
    fig1.suptitle('Spectra of learned network and RCWA: freq-R')
    for idx_fig in range(4):
        plt.subplot(2, 2, idx_fig + 1)
        plt.plot(np.arange(R_pick.shape[1]), R_pick[idx_fig, :], c='r')
        plt.plot(np.arange(R_simu.shape[1]), R_simu[idx_fig, :], c='b')
        plt.legend(('learned spectra', 'RCWA spectra'))
        plt.title(str(idx_fig + 1) + 'th best match')

    fig2 = plt.figure(2, figsize=size_fig)
    fig2.suptitle('Spectra of learned network and RCWA: freq-T')
    for idx_fig in range(4):
        plt.subplot(2, 2, idx_fig + 1)
        plt.plot(np.arange(T_pick.shape[1]), T_pick[idx_fig, :], c='r')
        plt.plot(np.arange(T_simu.shape[1]), T_simu[idx_fig, :], c='b')
        plt.legend(('learned spectra', 'RCWA spectra'))
        plt.title(str(idx_fig + 1) + 'th best match')
    plt.show()