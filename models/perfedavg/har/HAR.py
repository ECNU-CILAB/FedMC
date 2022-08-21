from models.perfedavg.Learner import Learner


class HAR(Learner):
    def __init__(self):
        config = [
            ('conv1d', [16, 1, 2, -1, 1, 2]),
            ('relu', [True]),
            ('max_pool1d', [2, 2, 0]),
            ('conv1d', [16, 16, 2, -1, 1, 2]),
            ('relu', [True]),
            ('max_pool1d', [2, 2, 0]),
            ('conv1d', [32, 16, 2, -1, 1, 2]),
            ('relu', [True]),
            ('max_pool1d', [2, 2, 0]),
            ('conv1d', [32, 32, 2, -1, 1, 2]),
            ('relu', [True]),
            ('max_pool1d', [2, 2, 0]),
            ('flatten', []),
            ('linear', [256, 1184]),
            ('linear', [6, 256])
        ]
        super(HAR, self).__init__(config=config)