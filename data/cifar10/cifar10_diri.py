import random
import torch
import numpy as np
import torch
from tensorflow import keras
from torch.utils.data import Dataset, DataLoader
from utils.tools import setup_seed


def partition_data(users_num=100, alpha=0.5, **data_kwargs):
    x_train, y_train, x_test, y_test = data_kwargs['x_train'], data_kwargs['y_train'], data_kwargs['x_test'], \
                                       data_kwargs['y_test']
    client_ids = {i: {'train': [], 'test': []} for i in range(users_num)}
    for k in range(10):  # 10个类别
        train_ids, test_ids = np.where(y_train == k)[0], np.where(y_test == k)[0]
        np.random.shuffle(train_ids)
        np.random.shuffle(test_ids)
        proportions = np.random.dirichlet(np.repeat(alpha, users_num))
        train_batch_sizes = [int(p * len(train_ids)) for p in proportions]
        test_batch_sizes = [int(p * len(test_ids)) for p in proportions]
        train_start = 0
        test_start = 0
        for i in range(users_num):
            train_size = train_batch_sizes[i]
            test_size = test_batch_sizes[i]
            client_ids[i]['train'] += train_ids[train_start: train_start + train_size].tolist()
            client_ids[i]['test'] += test_ids[test_start: test_start + test_size].tolist()
            train_start += train_size
            test_start += test_size

    return client_ids


class CIFAR10_DATASET(Dataset):
    def __init__(self, X, Y, ids, transform=None):
        self.X = X
        self.Y = Y
        self.ids = ids
        self.transform = transform

    def __getitem__(self, item):
        x = self.X[self.ids[item]]
        y = self.Y[self.ids[item]]
        assert x.shape == (32, 32, 3)
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        return x, y

    def __len__(self):
        return len(self.ids)


def get_cifar10_dirichlet_data_loaders(users_num=100, batch_size=50, alpha=0.5, train_transform=None,
                                       test_transform=None):
    print("CIFAR10 dirichlet non-IID distribution")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.reshape((50000,))
    y_test = y_test.reshape((10000,))
    data_kwargs = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test
    }
    setup_seed(24)
    if alpha <= 0.001:
        users, train_loaders, test_loaders = get_extreme_non_iid(num_users=users_num, batch_size=batch_size,
                                                                 train_transform=train_transform,
                                                                 test_transform=test_transform, **data_kwargs)
        return users, train_loaders, test_loaders
    client_ids = partition_data(users_num=users_num, alpha=alpha, **data_kwargs)
    train_loaders = {}
    test_loaders = {}
    users = []
    for user, train_test_ids in client_ids.items():
        train_ids = train_test_ids['train']
        test_ids = train_test_ids['test']

        # assert len(train_ids) > 0
        # assert len(test_ids) > 0
        if len(train_ids) > 0 and len(test_ids) > 0:
            users.append(user)
            # print("user: ", len(train_ids), len(test_ids))

            dataset = CIFAR10_DATASET(X=x_train, Y=y_train, ids=train_ids, transform=train_transform)
            train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            train_loaders[user] = train_loader
            dataset = CIFAR10_DATASET(X=x_test, Y=y_test, ids=test_ids, transform=test_transform)
            test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loaders[user] = test_loader

    # print("total users: ", len(users))
    return users, train_loaders, test_loaders


def _get_extreme_non_iid(num_users=100, use='train', batch_size=50, transform=None, rand_set_all=[], **data_kwargs):
    x_train, y_train, x_test, y_test = data_kwargs['x_train'], data_kwargs['y_train'], data_kwargs['x_test'], \
                                       data_kwargs['y_test']
    if use == 'train':
        X, Y = x_train, y_train
        all_clients = [i for i in range(num_users)]
    else:
        X, Y = x_test, y_test
        all_clients = [i for i in range(num_users)]

    dict_users = {i: np.array([], dtype='int64') for i in range(100)}

    idxs_dict = {}
    for i in range(len(X)):
        label = Y[i]
        if label not in idxs_dict.keys():
            idxs_dict[label] = []
        idxs_dict[label].append(i)

    num_classes = 10
    shard_per_client = 1
    shard_per_class = int(shard_per_client * num_users / num_classes)  # 1 shards per client * 30 clients / 10 classes
    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x
    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))
    # divide and assign
    for i in range(num_users):
        rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(Y[value])
        assert len(x) <= shard_per_client
        test.append(value)
    test = np.concatenate(test)
    assert len(test) == len(X)
    assert len(set(list(test))) == len(X)

    data_loaders = {}

    for client in all_clients:
        dataset = CIFAR10_DATASET(X=X, Y=Y, ids=dict_users[client], transform=transform)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        data_loaders[client] = data_loader

    return all_clients, data_loaders, rand_set_all


def get_extreme_non_iid(num_users=100, batch_size=50, train_transform=None, test_transform=None, **data_kwargs):
    train_all_clients, train_loaders, rand_set_all = _get_extreme_non_iid(num_users=num_users, use='train',
                                                                          batch_size=batch_size,
                                                                          transform=train_transform, rand_set_all=[],
                                                                          **data_kwargs)
    test_all_clients, test_loaders, rand_set_all = _get_extreme_non_iid(num_users=num_users, use='test',
                                                                        batch_size=batch_size,
                                                                        transform=test_transform,
                                                                        rand_set_all=rand_set_all, **data_kwargs)
    train_all_clients.sort()
    test_all_clients.sort()
    assert train_all_clients == test_all_clients
    return train_all_clients, train_loaders, test_loaders
