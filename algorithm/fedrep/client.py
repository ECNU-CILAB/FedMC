import torch
from utils.flutils import *
import torch.optim as optim
import numpy as np


class CLIENT:
    def __init__(self, user_id, train_loader, test_loader, config):
        self.config = config
        self.user_id = user_id
        use_cuda = config.cuda and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{config.cudaNo}" if use_cuda else "cpu")
        kwargs = {'dropout': self.config.dropout}
        self.model = select_model(algorithm=config.algorithm, model_name=config.model, **kwargs)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.stats = {
            'train-samples': 0,
            'test-samples': 0,
            'GM-train-accuracy': 0,
            'GM-test-accuracy': 0,
            'GM-train-loss': None,
            'GM-test-loss': None
        }
        self.w_glob_keys = self.set_w_glob_keys()

    def set_w_glob_keys(self):
        keys = list(self.model.state_dict().keys())
        keys = list(reversed(keys))  # [top -> down(near the data)]
        return keys[2 * self.config.depth:]

    @property
    def train_samples_num(self):
        return len(self.train_loader.dataset) if self.train_loader else None

    @property
    def test_samples_num(self):
        return len(self.test_loader.dataset) if self.test_loader else None

    def train(self, round_th):
        model = self.model
        model.to(self.device)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()

        bias_p = []
        weight_p = []
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        optimizer = optim.SGD(params=[{'params': weight_p, 'weight_decay': 1e-4},
                                      {'params': bias_p, 'weight_decay': 0}],
                              lr=self.config.lr * self.config.lrDecay ** (round_th / self.config.decayStep))

        mean_loss = []

        head_eps = self.config.epoch - self.config.localRepEpoch

        for epoch in range(self.config.epoch):
            # training head first
            if epoch < head_eps:
                for name, param in model.named_parameters():
                    if name in self.w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            # then training base
            else:
                for name, param in model.named_parameters():
                    if name in self.w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            for step, (data, labels) in enumerate(self.train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                mean_loss.append(loss.item())

        if np.isnan(sum(mean_loss) / len(mean_loss)):
            print(f"client {self.user_id}, loss NAN")
            exit(0)

        train_samples_num, update = self.train_samples_num, self.get_params()
        return train_samples_num, update, sum(mean_loss) / len(mean_loss)

    def fine_tuning_test(self):
        train_samples, GM_train_acc, GM_train_loss = self.test(dataset='train')
        test_samples, GM_test_acc, GM_test_loss = self.test(dataset='test')

        self.stats.update({
            'train-samples': train_samples,
            'test-samples': test_samples,
            'GM-train-accuracy': GM_train_acc,
            'GM-test-accuracy': GM_test_acc,
            'GM-train-loss': GM_train_loss,
            'GM-test-loss': GM_test_loss
        })

    def test(self, dataset='test'):
        model = self.model
        model.eval()
        model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()

        if dataset == 'test':
            data_loader = self.test_loader
        else:
            data_loader = self.train_loader

        total_right = 0
        total_samples = 0
        mean_loss = []
        with torch.no_grad():
            for step, (data, labels) in enumerate(data_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = model(data)
                loss = criterion(output, labels)
                mean_loss.append(loss.item())
                output = torch.argmax(output, dim=-1)
                total_right += torch.sum(output == labels)
                total_samples += len(labels)
            acc = float(total_right) / total_samples

        return total_samples, acc, sum(mean_loss) / len(mean_loss)

    def get_params(self):
        return self.model.cpu().state_dict()

    def set_params(self, model_params):
        self.model.load_state_dict(model_params)

    def set_shared_params(self, params):
        tmp_params = self.get_params()
        for (key, value) in params.items():
            if key in self.w_glob_keys:
                tmp_params[key] = value
        self.set_params(tmp_params)

    def update(self, client):
        self.user_id = client.user_id
        self.model.load_state_dict(client.model.state_dict())
        self.train_loader = client.train_loader
        self.test_loader = client.test_loader
