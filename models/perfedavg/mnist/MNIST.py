from models.perfedavg.Learner import Learner


class MNIST(Learner):
    def __init__(self):
        config = [
            ('conv2d', [32, 1, 5, 5, 1, 2]),
            ('relu', [True]),
            ('max_pool2d', [2, 2, 0]),
            ('conv2d', [64, 32, 5, 5, 1, 2]),
            ('relu', [True]),
            ('max_pool2d', [2, 2, 0]),
            ('flatten', []),
            ('linear', [512, 64 * 7 * 7]),
            ('linear', [10, 512])
        ]
        super(MNIST, self).__init__(config=config)