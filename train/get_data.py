from utils import *
import torch
import gc

from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, x, y, y_ref):
        self.x = x
        self.y = y
        self.y_ref = y_ref

        self._len = len(x)

    def __getitem__(self, item: int):
        #
        return self.x[item], self.y[item], self.y_ref[item]

    def __len__(self):
        return self._len


def get_data():
    # obtain data
    X, Y, Y_ref, _, _ = utils()
    print('X:', X.shape)
    print('Y:', Y.shape)
    print('Y_ref:', Y_ref.shape)

    data_train = MyDataSet(X[:40725, :, :, :, :], Y[:40725, :, :, :, :], Y_ref[:40725, :, :])
    data_test = MyDataSet(X[40725:, :, :, :, :], Y[40725:, :, :, :, :], Y_ref[40725:, :, :])
    del X, Y, Y_ref
    gc.collect()

    # load data
    trainloader = torch.utils.data.DataLoader(dataset=data_train, batch_size=200, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=data_test, batch_size=200, shuffle=False)

    return trainloader, testloader