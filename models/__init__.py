from .fedavg.mnist.MNIST import MNIST as FedAvg_MNIST
from .fedavg.cifar10.CIFAR10 import CIFAR10 as FedAvg_CIFAR10
from .fedavg.cifar100.CIFAR100 import CIFAR100 as FedAvg_CIFAR100
from .fedavg.har.HAR import HAR as FedAvg_HAR

from .fedmc.cifar10.CIFAR10 import CIFAR10 as FedMC_CIFAR10
from .fedmc.cifar100.CIFAR100 import CIFAR100 as FedMC_CIFAR100
from .fedmc.mnist.MNIST import MNIST as FedMC_MNIST
from .fedmc.har.HAR import HAR as FedMC_HAR

from .perfedavg.mnist.MNIST import MNIST as Per_FedAvg_MNIST
from .perfedavg.cifar10.CIFAR10 import CIFAR10 as Per_FedAvg_CIFAR10
from .perfedavg.cifar100.CIFAR100 import CIFAR100 as Per_FedAvg_CIFAR100
from .perfedavg.har.HAR import HAR as Per_FedAvg_HAR

__all__ = [
    'FedAvg_MNIST',
    'FedAvg_CIFAR10',
    'FedAvg_CIFAR100',
    'FedAvg_HAR',
    'FedMC_CIFAR10',
    'FedMC_CIFAR100',
    'FedMC_MNIST',
    'FedMC_HAR',
    'Per_FedAvg_MNIST',
    'Per_FedAvg_CIFAR10',
    'Per_FedAvg_CIFAR100',
    'Per_FedAvg_HAR'
]
