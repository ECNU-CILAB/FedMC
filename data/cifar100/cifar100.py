import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tensorflow import keras
from utils.tools import *
from prettytable import PrettyTable


def load_data_from_keras():
    (x_train_fine, y_train_fine), (x_test_fine, y_test_fine) = keras.datasets.cifar100.load_data(label_mode='fine')
    (x_train_coarse, y_train_coarse), (x_test_coarse, y_test_coarse) = keras.datasets.cifar100.load_data(
        label_mode='coarse')

    return (x_train_fine, y_train_fine), (x_test_fine, y_test_fine), (x_train_coarse, y_train_coarse), (
        x_test_coarse, y_test_coarse)


class CIFAR100_DATASET(Dataset):
    def __init__(self, X, Y, ids, transform=None):
        self.X = X
        self.Y = Y
        self.ids = ids
        self.transform = transform

    def __getitem__(self, item):
        x, y = self.X[self.ids[item]], self.Y[self.ids[item]]
        assert x.shape == (32, 32, 3)
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        return x, y

    def __len__(self):
        return len(self.ids)


def get_cifar100_data_loaders(batch_size=10, train_transform=None, test_transform=None):
    N = 100
    setup_seed(24)
    (x_train_fine, y_train_fine), (x_test_fine, y_test_fine), (x_train_coarse, y_train_coarse), (
        x_test_coarse, y_test_coarse) = load_data_from_keras()

    y_train_fine = y_train_fine.reshape((50000,))
    y_test_fine = y_test_fine.reshape((10000,))
    y_train_coarse = y_train_coarse.reshape((50000,))
    y_test_coarse = y_test_coarse.reshape((10000,))

    # print("x_train_fine", x_train_fine.shape, "y_train_fine", y_train_fine.shape)
    # print("x_test_fine", x_test_fine.shape, "y_test_fine", y_test_fine.shape)
    #
    # print("x_train_coarse", x_train_coarse.shape, "y_train_coarse", y_train_coarse.shape)
    # print("x_test_coarse", x_test_coarse.shape, "y_test_coarse", y_test_coarse.shape)

    table = PrettyTable(['TrainingX.', 'TrainingY.', 'TestX.', 'TestY.'])
    table.add_row([x_train_fine.shape, y_train_fine.shape, x_test_fine.shape, y_test_fine.shape])
    print(table)

    assert (x_train_fine == x_train_coarse).all()
    assert (x_test_fine == x_test_coarse).all()

    coarse_group_train_ids = {i: [] for i in range(20)}
    for i in range(len(y_train_coarse)):
        coarse_l = y_train_coarse[i]
        coarse_group_train_ids[coarse_l].append(i)

    coarse_group_test_ids = {i: [] for i in range(20)}
    for i in range(len(y_test_coarse)):
        coarse_l = y_test_coarse[i]
        coarse_group_test_ids[coarse_l].append(i)

    for key in coarse_group_train_ids.keys():
        random.shuffle(coarse_group_train_ids[key])
        random.shuffle(coarse_group_test_ids[key])

    train_loaders = {}
    test_loaders = {}
    TOTAL_TRAIN_SAMPLES = 0
    TOTAL_TEST_SAMPLES = 0
    for user in range(N):
        group_id = int(user / 5)
        drift = user % 5
        train_size = int(len(coarse_group_train_ids[group_id]) / 5)
        train_left = len(coarse_group_train_ids[group_id]) - 5 * train_size
        test_size = int(len(coarse_group_test_ids[group_id]) / 5)
        test_left = len(coarse_group_test_ids[group_id]) - 5 * test_size

        train_ids = coarse_group_train_ids[group_id][drift * train_size: (drift + 1) * train_size]
        test_ids = coarse_group_test_ids[group_id][drift * test_size: (drift + 1) * test_size]
        if drift < train_left:
            train_ids.append(coarse_group_train_ids[-train_left + drift])
        if drift < test_left:
            test_ids.append(coarse_group_test_ids[-test_left + drift])

        trainSet = CIFAR100_DATASET(X=x_train_fine, Y=y_train_fine, ids=train_ids, transform=train_transform)
        loader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, num_workers=0)
        TOTAL_TRAIN_SAMPLES += len(loader.sampler)
        train_loaders.update({user: loader})

        testSet = CIFAR100_DATASET(X=x_test_fine, Y=y_test_fine, ids=test_ids, transform=test_transform)
        loader = DataLoader(dataset=testSet, batch_size=batch_size, shuffle=False, num_workers=0)
        TOTAL_TEST_SAMPLES += len(loader.sampler)
        test_loaders.update({user: loader})
    assert TOTAL_TRAIN_SAMPLES == 50000, TOTAL_TEST_SAMPLES == 10000
    return [i for i in range(N)], train_loaders, test_loaders
