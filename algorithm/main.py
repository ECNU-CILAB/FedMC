import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
from utils.args import *
from torch.utils.tensorboard import SummaryWriter
from algorithm.fedavg.server import SERVER as FedAvg_SERVER
from algorithm.fedprox.server import SERVER as FedProx_SERVER
from algorithm.mocha.server import SERVER as Mocha_SERVER
from algorithm.fedper.server import SERVER as FedPer_SERVER
from algorithm.lgfedavg.server import SERVER as LG_FedAvg_SERVER
from algorithm.per_fedavg.server import SERVER as Per_FedAvg_SERVER
from algorithm.fedrep.server import SERVER as FedRep_SERVER
from algorithm.fedfomo.server import SERVER as FedFomo_SERVER
from algorithm.fedmc.server import SERVER as FedMC_SERVER

if __name__ == '__main__':
    args = parse_args()
    DROPOUTS = None
    if args.dataset == 'cifar100':
        DROPOUTS = {'small': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                    'big': [0.75, 0.75, 0.9, 0.9, 0.9, 0.5]}
    elif args.dataset == 'cifar10' or args.dataset == 'cifar10_diri':
        if args.algorithm in ['fedmc']:
            DROPOUTS = {'small': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                        'big': [0.5, 0.5, 0.9, 0.9, 0.9, 0.5]}
        elif args.algorithm in ['fedper', 'fedrep', 'lgfedavg']:
            DROPOUTS = {'small': [0.25, 0.25, 0.5],
                        'big': [0.5, 0.5, 0.75]}
    else:
        DROPOUTS = {'small': [0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                    'big': [0.75, 0.75, 0.9, 0.9, 0.9, 0.5]}
    if DROPOUTS is not None:
        args.dropout = DROPOUTS[args.drop]

    writer = SummaryWriter(f'../runs/{args.algorithm}/{args.dataset}/{args.logname}')
    args.writer = writer
    config = args

    server = None
    if config.algorithm == 'fedavg':
        server = FedAvg_SERVER(config=config)
    elif config.algorithm == 'fedprox':
        server = FedProx_SERVER(config=config)
    elif config.algorithm == 'mocha':
        server = Mocha_SERVER(config=config)
    elif config.algorithm == 'fedper':
        server = FedPer_SERVER(config=config)
    elif config.algorithm == 'lgfedavg':
        server = LG_FedAvg_SERVER(config=config)
    elif config.algorithm == 'per_fedavg':
        server = Per_FedAvg_SERVER(config=config)
    elif config.algorithm == 'fedrep':
        server = FedRep_SERVER(config=config)
    elif config.algorithm == 'fedfomo':
        server = FedFomo_SERVER(config=config)
    elif config.algorithm == 'fedmc':
        server = FedMC_SERVER(config=config)

    server.federate()
