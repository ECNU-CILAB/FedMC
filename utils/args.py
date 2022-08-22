import argparse

ALGORITHMS = ['fedavg', 'fedmc', 'fedprox', 'lgfedavg', 'per_fedavg', 'mocha', 'fedper', 'fedfomo', 'fedrep']
DATASETS = ['cifar10', 'mnist', 'cifar100', 'har', 'cifar10_diri']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logname',
                        help='logname for tensorboard?',
                        type=str,
                        required=True)

    parser.add_argument('--algorithm',
                        help='algorithm',
                        choices=ALGORITHMS,
                        required=True)

    parser.add_argument('--dataset',
                        help='name of dataset',
                        choices=DATASETS,
                        required=True)

    parser.add_argument('--model',
                        help='name of model',
                        type=str,
                        required=True)

    parser.add_argument('--numRounds',
                        help='# of communication round',
                        type=int,
                        default=100)

    parser.add_argument('--evalInterval',
                        help='communication rounds between two evaluation',
                        type=int,
                        default=1)

    parser.add_argument('--clientsPerRound',
                        help='# of selected clients per round',
                        type=int,
                        default=1)

    parser.add_argument('--epoch',
                        help='# of epochs when clients train on data',
                        type=int,
                        default=1)

    parser.add_argument('--batchSize',
                        help='batch size when clients train on data',
                        type=int,
                        default=1)

    parser.add_argument('--lr',
                        help='learning rate for local optimizers',
                        type=float,
                        default=3e-4)

    parser.add_argument('--lrDecay',
                        help='decay rate for learning rate',
                        type=float,
                        default=0.99)

    parser.add_argument('--decayStep',
                        help='decay rate for learning rate',
                        type=int,
                        default=200)

    parser.add_argument('--alpha',
                        help='alpha for dirichlet distribution partition',
                        type=float,
                        default=0.5)

    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting',
                        type=int,
                        default=24)

    parser.add_argument('--cuda',
                        help='using cuda',
                        type=int,
                        default=1)

    parser.add_argument('--cudaNo',
                        help="cuda # 0, 1",
                        type=int,
                        default=0)

    parser.add_argument('--mu',
                        help='coefficient for balancing cross entropy loss and critic loss',
                        type=float,
                        default=0.1)

    parser.add_argument('--omega',
                        help='coefficient for balancing w-distance loss and gradient penalty loss',
                        type=float,
                        default=0.1)

    parser.add_argument('--diffCo',
                        help='coefficient for balancing classification loss and regularizer loss (model difference) in FedProx',
                        type=float,
                        default=0.1)

    parser.add_argument('--depth',
                        help='num of shared layers for lg-fedavg or private layers for fedper',
                        type=int,
                        default=1)

    parser.add_argument('--mode',
                        help='Integration of global encoder and local encoder',
                        choices=['concat', 'addition'],
                        default='concat')

    parser.add_argument('--earlyStopping',
                        help='do early-stopping for fedsp or fedmc',
                        type=int,
                        default=0)

    parser.add_argument('--drop',
                        help='dropout for cifar10 and cifar100 model, (shared global encoder drop1, shared global encoder drop2, private local encoder drop1, private local encoder drop2, clf drop, critic drop)',
                        choices=['small', 'big'],
                        default='small')

    parser.add_argument('--threads',
                        help='how many threads running for training and testing? default 1, i.e., one by one',
                        type=int,
                        default=1)

    parser.add_argument('--innerUpdateLr',
                        help='[Per-FedAvg]inner update learning rate, i.e., alpha in Per-FedAvg',
                        type=float,
                        default=0.01)

    parser.add_argument('--outerUpdateLr',
                        help='[Per-FedAvg]outer update learning rate, i.e., beta in Per-FedAvg',
                        type=float,
                        default=0.01)

    parser.add_argument('--secondOrder',
                        help='[Per-FedAvg]second-order MAML? default False(0) for first-order MAML (FOMAML)',
                        type=int,
                        default=0)

    parser.add_argument('--numLayersKeep',
                        help='global model layers for mocha, if 2, layer = 2/2 = 1',
                        type=int,
                        default=2)

    parser.add_argument('--numQueryClients',
                        help='number of models need to be downloaded for FedFomo, default 5',
                        type=int,
                        default=5)

    parser.add_argument('--valFomo',
                        help='adopt validation set for fedfomo',
                        type=int,
                        default=1)

    parser.add_argument('--localRepEpoch',
                        help="the number of local epochs for the representation for FedRep",
                        type=int,
                        default=2)

    parser.add_argument('--keepMagnitude',
                        help='keep magnitude while performing gradient surgery',
                        type=int,
                        default=0)

    parser.add_argument('--numUsers',
                        help='number of users for partitioning cifar10(two classes per client)',
                        type=int,
                        default=100)

    parser.add_argument('--classesPerClient',
                        help='pathological partitioning, classes per client',
                        type=int,
                        default=2)

    parser.add_argument('--K',
                        help='[MOCHA] computation steps',
                        type=int,
                        default=2000)

    parser.add_argument("--Lk",
                        help="[MOCHA] Regularization term",
                        type=float,
                        default=0.004)

    return parser.parse_args()
