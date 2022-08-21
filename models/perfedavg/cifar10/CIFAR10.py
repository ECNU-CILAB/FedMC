from models.perfedavg.Learner import Learner


class CIFAR10(Learner):
    def __init__(self):
        config = [
            ('conv2d', [32, 3, 3, 3, 1, 0]),
            ('relu', [True]),
            ('conv2d', [64, 32, 3, 3, 1, 0]),
            ('relu', [True]),
            ('max_pool2d', [2, 2, 0]),
            ('dropout', [0.25]),

            ('conv2d', [64, 64, 3, 3, 1, 0]),
            ('relu', [True]),
            ('conv2d', [64, 64, 3, 3, 1, 0]),
            ('relu', [True]),
            ('max_pool2d', [2, 2, 0]),
            ('dropout', [0.25]),

            ('flatten', []),
            ('linear', [512, 64 * 5 * 5]),
            ('relu', [True]),
            ('dropout', [0.5]),
            ('linear', [10, 512])
        ]
        super(CIFAR10, self).__init__(config=config)