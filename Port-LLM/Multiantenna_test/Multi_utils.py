import numpy as np
import scipy.io as scio
import random
from einops import rearrange
import torch
import os
import h5py
import gc

'''
Verify the performance of our trained model when the number of antennas at the BS is 32×8 and the test velocity of UE is 150 km/h.
'''


## load data
def load_data(file, index1):
    ## file path
    files_score = os.path.join(file, index1, 'ports.mat') # channel tables
    files_href = os.path.join(file, index1, 'h_ref.mat')  # reference channel

    ##
    href = torch.tensor(scio.loadmat(files_href)['h_ref'])
    href = torch.stack((href.real, href.imag), dim=2)

    ##
    score = h5py.File(files_score, 'r')['ports']
    score_real = torch.tensor(score['real'])
    score_imag = torch.tensor(score['imag'])

    del score
    gc.collect()

    score_real = rearrange(score_real, 'a b c d e->e b d c a')
    score_imag = rearrange(score_imag, 'a b c d e->e b d c a')
    score = torch.stack((score_real, score_imag), dim=2)

    del score_real, score_imag
    gc.collect()

    _, L, _, _, _, _ = score.shape
    ##
    H_ref = torch.zeros(10, 1, 2, int(L), 256)
    for i in range(10):
        for j in range(int(L)):
            H_ref[i, :, :, j, :] = href[i, :, :, :]
    H_ref = H_ref.squeeze(1)

    del href
    gc.collect()

    ## data processing
    S = 8  # split step
    T = 8  # the length of historical moments
    F = 8  # the length of forecasting moments
    B = int(torch.floor(torch.Tensor([L - F - T]) / torch.Tensor([S])) + 1)
    X = torch.zeros(10, B, T, 2, 100, 50, 256)
    Y = torch.zeros(10, B, F, 2, 100, 50, 256)
    Y_ref = torch.zeros(10, B, 2, F, 256)

    ##
    # score:10*46*2*100*50*64
    # H_ref:10*2*46*64
    # p:10*46*2
    for i in range(B):
        X[:, i, :, :, :, :, :] = score[:, i * S:i * S + T, :, :, :, :]  # 10*B*T*2*100*50*32
        Y[:, i, :, :, :, :, :] = score[:, i * S + T:i * S + T + F, :, :, :, :]  # 10*B*F*2*100*50*32
        Y_ref[:, i, :, :, :] = H_ref[:, :, i * S + T:i * S + T + F, :]  # 10*B*2*F*32

    return X, Y, Y_ref


# process data
def process_data(file):
    list_x = []
    list_y = []
    list_href = []

    for j in [150]:
        print('V_j:', j)
        index = 'V_' + str(j)

        X, Y, Y_ref= load_data(file, index)
        list_x.append(X)
        list_y.append(Y)
        list_href.append(Y_ref)

        del X, Y, Y_ref
        gc.collect()

    return list_x, list_y, list_href


# concatenate and normalize data
def concatenate_and_normalize(list_x, list_y, list_href):
    # concatenate data
    # list_scores[i]: B*T*2*100*50
    print('Concatenating data...')
    inputs = torch.cat(list_x, dim=0)  # #10*B*T*2*100*50*64
    outputs = torch.cat(list_y, dim=0)  # #10*B*F*2*100*50*64
    outputs_ref = torch.cat(list_href, dim=0)  # 10*B*2*F*64
    print('Finished concatenating data.')

    # normalize data
    inputs = rearrange(inputs, 'a b c d e f g-> a b c e f g d')  # 10*B*T*2*100*50*16->10*B*T*100*50*16*2
    outputs = rearrange(outputs, 'a b c d e f g-> a b c e f g d')  # 10*B*F*2*100*50*16->10*B*F*100*50*16*2

    ## mean and standard deviation of the training set
    mean_s = torch.tensor([-5.7430e-07, 2.1659e-08])  # 2
    std_s = torch.tensor([0.6991, 0.6991])

    # mean-std nomalization
    inputs = (inputs - mean_s) / std_s
    outputs = (outputs - mean_s) / std_s

    inputs = rearrange(inputs, 'a b c d e f g-> a b c g d e f')  # 10*B*T*100*50*16*2->10*B*T*2*100*50*16
    outputs = rearrange(outputs, 'a b c d e f g-> a b c g d e f')  # B*F*100*50*2->B*F*2*100*50

    # outputs_ref:10*B*2*F*64->10*B*F*64*2
    outputs_ref = rearrange(outputs_ref, 'a b c d e-> a b d e c')
    outputs_ref = (outputs_ref - mean_s) / std_s  # 10*10*B*2*F
    outputs_ref = rearrange(outputs_ref, 'a b c d e-> a b c e d')  # 10*B*F*16*2->10*B*F*2*16

    inputs = rearrange(inputs, 'a b c d e f g->(a b) c d e f g')  # 10*B*T*2*100*50*16->(10*B)*T*2*100*50*16
    outputs = rearrange(outputs, 'a b c d e f g->(a b) c d e f g')  # 10*B*F*2*100*50*16->(10*B)*F*2*100*50*16
    outputs_ref = rearrange(outputs_ref, 'a b c d e->(a b) c d e')  # 10*B*F*2*16->(10*B)*F*2*16

    return inputs, outputs, outputs_ref, mean_s, std_s


# main
def utils():
    file = '/home/edu/ZhangYali/SE_DATA/32_8/'
    list_x, list_y, list_href = process_data(file)
    inputs, outputs, outputs_ref, mean_s, std_s = concatenate_and_normalize(list_x, list_y, list_href)

    return inputs, outputs, outputs_ref, mean_s, std_s