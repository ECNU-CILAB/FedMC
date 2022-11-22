import numpy as np
import argparse

from data.mnist.mnist import get_mnist_data_loaders

parser = argparse.ArgumentParser()

parser.add_argument('--name',
                    help='name of dataset to evaluate; default: cifar10;',
                    type=str,
                    choices=['mnist'],
                    default='mnist')

parser.add_argument('--alpha',
                    help='alpha for dirichlet distribution',
                    type=float,
                    default=0.5)

args = parser.parse_args()
dataset = args.name

N_class = None
users, train_loaders, test_loaders = None, None, None

if dataset == 'mnist':
    users, train_loaders, test_loaders = get_mnist_data_loaders(batch_size=10)
    N_class = 10

total_dataset_distribution = {i: 0 for i in range(N_class)}
total_dataset_num_samples = 0
users_histogram = []
for user in users:

    train_loader = train_loaders[user]
    distribution = {i: 0 for i in range(N_class)}

    num_samples = len(train_loader.sampler)
    total_dataset_num_samples += num_samples

    for step, (data, labels) in enumerate(train_loader):
        for label in labels:
            distribution[label.item()] += 1

    histogram = []
    for label, num in distribution.items():
        total_dataset_distribution[label] += num
        histogram.append(num / num_samples)
    users_histogram.append((num_samples, histogram))

total_dataset_histogram = np.zeros(N_class)
for num, histogram in users_histogram:
    total_dataset_histogram = np.add(total_dataset_histogram, num * np.array(histogram))

total_dataset_histogram /= total_dataset_num_samples
print(total_dataset_histogram)

distances = []

for num, histogram in users_histogram:
    # print(total_dataset_histogram, histogram)
    d = np.linalg.norm(total_dataset_histogram - histogram, ord=1)
    distances.append(d)

print("avg emd: ", sum(distances) / len(distances))
