import torch
from torchvision.transforms import transforms

# datasets
from data.mnist.mnist import get_mnist_data_loaders
from data.cifar10.cifar10 import get_cifar10_data_loaders
from data.cifar10.cifar10_diri import get_cifar10_dirichlet_data_loaders
from data.cifar100.cifar100 import get_cifar100_data_loaders
from data.har.har import get_har_dataLoaders

# models
from models import *


def setup_datasets(dataset, batch_size, num_users=100, alpha=None, classes_per_client=2):
    users, train_loaders, test_loaders = [], [], []
    if dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        users, train_loaders, test_loaders = get_mnist_data_loaders(num_users=num_users,
                                                                    batch_size=batch_size,
                                                                    train_transform=train_transform,
                                                                    test_transform=test_transform)
    elif dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                 std=[0.247, 0.243, 0.262])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                 std=[0.247, 0.243, 0.262])
        ])
        users, train_loaders, test_loaders = get_cifar10_data_loaders(num_users=num_users,
                                                                      batch_size=batch_size,
                                                                      train_transform=train_transform,
                                                                      test_transform=test_transform,
                                                                      shard_per_client=classes_per_client)
    elif dataset == 'cifar10_diri':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                 std=[0.247, 0.243, 0.262])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                 std=[0.247, 0.243, 0.262])
        ])
        users, train_loaders, test_loaders = get_cifar10_dirichlet_data_loaders(users_num=num_users, alpha=alpha,
                                                                               batch_size=batch_size,
                                                                               train_transform=train_transform,
                                                                               test_transform=test_transform)
    elif dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])
        users, train_loaders, test_loaders = get_cifar100_data_loaders(batch_size=batch_size,
                                                                      train_transform=train_transform,
                                                                      test_transform=test_transform)
    elif dataset == 'har':
        users, train_loaders, test_loaders = get_har_dataLoaders(batch_size=10)

    return users, train_loaders, test_loaders


def select_model(algorithm, model_name, **kwargs):
    model = None
    if algorithm in ['fedavg', 'fedprox', 'mocha', 'fedper', 'fedfomo', 'fedrep', 'lgfedavg']:
        if model_name == 'mnist':
            model = FedAvg_MNIST()
        elif model_name == 'cifar10':
            model = FedAvg_CIFAR10()
        elif model_name == 'cifar100':
            model = FedAvg_CIFAR100()
        elif model_name == 'har':
            model = FedAvg_HAR()
        else:
            print(f"Unimplemented Model {model_name}")
    elif algorithm == 'per_fedavg':
        if model_name == 'mnist':
            model = Per_FedAvg_MNIST()
        elif model_name == 'cifar10':
            model = Per_FedAvg_CIFAR10()
        elif model_name == 'cifar100':
            model = Per_FedAvg_CIFAR100()
        elif model_name == 'har':
            model = Per_FedAvg_HAR()
        else:
            print(f"Unimplemented Model {model_name}")
    elif algorithm == 'fedmc':
        if model_name == 'mnist':
            model = FedMC_MNIST()
        elif model_name == 'cifar10':
            model = FedMC_CIFAR10(dropout=kwargs['dropout'])
        elif model_name == 'cifar100':
            model = FedMC_CIFAR100(dropout=kwargs['dropout'])
        elif model_name == 'har':
            model = FedMC_HAR()
        else:
            print(f"Unimplemented Model {model_name}")
    else:
        print(f"Unimplemented Algorithm {algorithm}")
    return model


def fed_average(updates):
    total_weight = 0
    (client_samples_num, new_params) = updates[0][0], updates[0][1]

    for item in updates:
        (client_samples_num, client_params) = item[0], item[1]
        total_weight += client_samples_num

    for k in new_params.keys():
        for i in range(0, len(updates)):
            client_samples, client_params = updates[i][0], updates[i][1]
            # weight
            w = client_samples / total_weight
            if i == 0:
                new_params[k] = client_params[k] * w
            else:
                new_params[k] += client_params[k] * w
    # return global model params
    return new_params


def avg_metric(metricList):
    total_weight = 0
    total_metric = 0
    for (samples_num, metric) in metricList:
        total_weight += samples_num
        total_metric += samples_num * metric
    average = total_metric / total_weight

    return average
