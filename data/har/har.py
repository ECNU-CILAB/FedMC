import os
import torch
import json
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from utils.tools import setup_seed
from prettytable import PrettyTable


def prepare_dataset():
    BASE = '../data/har/'

    f = open(BASE + 'UCI HAR Dataset/train/X_train.txt')
    X_TRAIN = []
    for line in f.readlines():
        lst = line.strip().split()
        lst = [float(e) for e in lst]
        X_TRAIN.append(lst)

    f = open(BASE + 'UCI HAR Dataset/train/y_train.txt')
    Y_TRAIN = []
    for line in f.readlines():
        label = int(line.strip()) - 1
        Y_TRAIN.append(label)

    f = open(BASE + 'UCI HAR Dataset/train/subject_train.txt')
    subject_TRAIN = []
    for line in f.readlines():
        s = int(line.strip())
        subject_TRAIN.append(s)

    f = open(BASE + 'UCI HAR Dataset/test/X_test.txt')
    X_TEST = []
    for line in f.readlines():
        lst = line.strip().split()
        lst = [float(e) for e in lst]
        X_TEST.append(lst)

    f = open(BASE + 'UCI HAR Dataset/test/y_test.txt')
    Y_TEST = []
    for line in f.readlines():
        label = int(line.strip()) - 1
        Y_TEST.append(label)

    f = open(BASE + 'UCI HAR Dataset/test/subject_test.txt')
    subject_TEST = []
    for line in f.readlines():
        s = int(line.strip())
        subject_TEST.append(s)

    assert len(X_TRAIN) == len(Y_TRAIN) == len(subject_TRAIN)
    assert len(X_TEST) == len(Y_TEST) == len(subject_TEST)

    X = []
    X.extend(X_TRAIN)
    X.extend(X_TEST)

    Y = []
    Y.extend(Y_TRAIN)
    Y.extend(Y_TEST)

    SUBJECT = []
    SUBJECT.extend(subject_TRAIN)
    SUBJECT.extend(subject_TEST)

    assert len(X) == len(Y) == len(SUBJECT)

    user_ids = {}
    for i in range(len(X)):
        person = SUBJECT[i] - 1
        if person in user_ids:
            user_ids[person].append(i)
        else:
            user_ids.update({person: []})

    return X, Y, user_ids


class HAR_DATASET(Dataset):
    def __init__(self, ids, X, Y):
        self.ids = ids
        self.X = X
        self.Y = Y

    def __getitem__(self, item):
        x = self.X[self.ids[item]]
        y = self.Y[self.ids[item]]
        x = np.array(x).reshape(1, 561)
        y = np.array(y)
        assert x.shape == (1, 561)
        x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        return x, y

    def __len__(self):
        return len(self.ids)


def get_har_dataLoaders(batch_size=10):
    setup_seed(24)
    X, Y, user_ids = prepare_dataset()

    table = PrettyTable(['users.', 'X.', 'Y.'])
    table.add_row([len(list(user_ids.keys())), f"{len(X), len(X[0])}", f"{len(Y), 6}"])
    print(table)

    train_loaders = {}
    test_loaders = {}

    all_clients = list(user_ids.keys())

    for user, id_lst in user_ids.items():
        shuffle(id_lst)
        train_ids = id_lst[0: int(len(id_lst) * 0.7)]
        test_ids = id_lst[int(len(id_lst) * 0.7):]
        dataset = HAR_DATASET(ids=train_ids, X=X, Y=Y)
        new_batch_size = min(batch_size, len(train_ids))
        data_loader = DataLoader(dataset=dataset, batch_size=new_batch_size, shuffle=True, num_workers=0)
        train_loaders[user] = data_loader

        dataset = HAR_DATASET(ids=test_ids, X=X, Y=Y)
        new_batch_size = min(batch_size, len(test_ids))
        data_loader = DataLoader(dataset=dataset, batch_size=new_batch_size, shuffle=True, num_workers=0)
        test_loaders[user] = data_loader

    return all_clients, train_loaders, test_loaders
