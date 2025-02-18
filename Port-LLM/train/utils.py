import numpy as np
import scipy.io as scio
import random
from einops import rearrange
import torch
import os
import h5py
import gc


# load data
def load_data(file, index1, index2):
    ## file path
    files_score = os.path.join(file, index1, index2, 'ports.mat')  # channel tables
    files_href = os.path.join(file, index1, index2, 'h_ref.mat')  # reference channel

    ##
    href = torch.tensor(scio.loadmat(files_href)['h_ref'])
    href = torch.stack((href.real, href.imag), dim=2)

    ##
    score = h5py.File(files_score, 'r')['ports']
    score_real = torch.tensor(score['real'])
    score_imag = torch.tensor(score['imag'])

    score_real = rearrange(score_real, 'a b c d e->e d c b a')
    score_imag = rearrange(score_imag, 'a b c d e->e d c b a')

    score = torch.stack((score_real, score_imag), dim=3)
    _, _, L, _, _, _ = score.shape
    del score_real, score_imag
    gc.collect()

    ## extend the reference channel according to the size of the channel tables
    H_ref = torch.zeros(10, 10, 2, int(L))
    for i in range(10):
        for j in range(10):
            for k in range(int(L)):
                H_ref[i, j, :, k] = href[i, j, :]
    del href
    gc.collect()

    ## data processing
    S = 8  # split step
    T = 8  # the length of historical moments
    F = 8  # the length of forecasting moments
    B = int(torch.floor(torch.Tensor([L - F - T]) / torch.Tensor([S])) + 1)  #

    X = torch.zeros(10, 10, B, T, 2, 100, 50)
    Y = torch.zeros(10, 10, B, F, 2, 100, 50)
    Y_ref = torch.zeros(10, 10, B, 2, F)
    # score: 10*10*491*2*100*50
    for i in range(B):
        X[:, :, i, :, :, :, :] = score[:, :, i * S:i * S + T, :, :, :]
        Y[:, :, i, :, :, :, :] = score[:, :, i * S + T:i * S + T + F, :, :, :]
        Y_ref[:, :, i, :, :] = H_ref[:, :, :, i * S + T:i * S + T + F]

    X = rearrange(X, 'a b c d e f g->(a b c) d e f g')  # (10,10,B,T,2,100,50)->(10*10*B,T,2,100,50)
    Y = rearrange(Y, 'a b c d e f g->(a b c) d e f g')  # (10,10,B,F,2,100,50)->(10*10*B,F,2,100,50)
    Y_ref = rearrange(Y_ref, 'a b c d e->(a b c) d e')  # (10,10,B,2,F)->(10*10*B,2,F)
    del score, H_ref
    gc.collect()

    return X, Y, Y_ref


# process data
def process_data(file, batch_size=1000):
    list_x = []
    list_y = []
    list_href = []

    for j in [90, 120, 150]:
        print('V_j:', j)
        index_1 = 'V_' + str(j)
        for i in [10, 6, 5]:
            print('T_i:', i)
            index_2 = 'T_' + str(i)
            X, Y, Y_ref = load_data(file, index_1, index_2)
            list_x.append(X)
            list_y.append(Y)
            list_href.append(Y_ref)
            del X, Y, Y_ref
            gc.collect()

    # Concatenate in batches to reduce memory usage
    inputs = []
    outputs = []
    outputs_ref = []

    print('Concatenating data...')
    for i in range(0, len(list_x), batch_size):
        batch_x = torch.cat(list_x[i:i + batch_size], dim=0)  # B*T*2*100*50
        batch_y = torch.cat(list_y[i:i + batch_size], dim=0)  # B*F*2*100*50
        batch_y_ref = torch.cat(list_href[i:i + batch_size], dim=0)  # B*2*F

        inputs.append(batch_x)
        outputs.append(batch_y)
        outputs_ref.append(batch_y_ref)

        del batch_x, batch_y, batch_y_ref
        gc.collect()
    print('Finished concatenating data.')

    inputs = torch.cat(inputs, dim=0)
    outputs = torch.cat(outputs, dim=0)
    outputs_ref = torch.cat(outputs_ref, dim=0)

    del list_x, list_y, list_href
    gc.collect()

    return inputs, outputs, outputs_ref


# concatenate and normalize data
def concatenate_and_normalize(inputs, outputs, outputs_ref):
    # normalize data
    inputs = rearrange(inputs, 'a b c d e-> a b d e c')  # B*T*2*100*50->B*T*100*50*2
    outputs = rearrange(outputs, 'a b c d e-> a b d e c')  # B*F*2*100*50->B*F*100*50*2
    scores = torch.cat((inputs, outputs), dim=1)  # (B,T+F,100,50,2)
    scores = rearrange(scores, 'a b c d e-> (a b c d) e')  # (B,T+F,100,50,2)->(B*(T+F)*100*50,2)

    # mean-std normalization
    mean_s = torch.mean(scores, dim=0)
    std_s = torch.std(scores, dim=0)

    del scores
    gc.collect()

    inputs = (inputs - mean_s) / std_s
    outputs = (outputs - mean_s) / std_s

    inputs = rearrange(inputs, 'a b c d e-> a b e c d')  # B*T*100*50*2->B*T*2*100*50
    outputs = rearrange(outputs, 'a b c d e-> a b e c d')  # B*F*100*50*2->B*F*2*100*50

    mean_s = mean_s.reshape(2, 1)
    std_s = std_s.reshape(2, 1)

    outputs_ref = (outputs_ref - mean_s) / std_s  # B*2*F
    outputs_ref = rearrange(outputs_ref, 'a b c-> a c b')  # B*2*F->B*F*2

    # random
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(3407)

    # shuffle data
    DataIndex = [i for i in range(len(inputs))]
    random.shuffle(DataIndex)
    inputs = inputs[DataIndex]
    outputs = outputs[DataIndex]
    outputs_ref = outputs_ref[DataIndex]

    return inputs, outputs, outputs_ref, mean_s, std_s


# main
def utils():
    print('start!')
    file = '/home/edu/ZhangYali/Data_new'
    inputs, outputs, outputs_ref = process_data(file)
    inputs, outputs, outputs_ref, mean_s, std_s = concatenate_and_normalize(inputs, outputs, outputs_ref)
    print('finished!')

    return inputs, outputs, outputs_ref, mean_s, std_s