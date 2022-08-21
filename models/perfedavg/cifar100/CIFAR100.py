from models.perfedavg.Learner import Learner


class CIFAR100(Learner):
    def __init__(self):
        config = [
            ('conv2d', [32, 3, 3, 3, 1, 0]),
            ('relu', [True]),
            ('conv2d', [64, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('max_pool2d', [2, 2, 0]),
            # ('dropout', [0.25]),
            ('dropout', [0.75]),

            ('conv2d', [64, 64, 3, 3, 1, 0]),
            ('relu', [True]),
            ('conv2d', [64, 64, 3, 3, 1, 0]),
            ('relu', [True]),
            ('max_pool2d', [2, 2, 0]),
            # ('dropout', [0.25]),
            ('dropout', [0.75]),

            ('flatten', []),
            ('linear', [512, 64 * 5 * 5]),
            ('relu', [True]),
            ('dropout', [0.9]),
            ('linear', [100, 512])
        ]

        super(CIFAR100, self).__init__(config=config)
