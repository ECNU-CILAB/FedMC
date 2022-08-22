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

    @property
    def train_samples_num(self):
        return len(self.train_loader.dataset) if self.train_loader else None

    @property
    def test_samples_num(self):
        return len(self.test_loader.dataset) if self.test_loader else None

    def train(self, round_th):
        print(f"{self.user_id} is training.")
        model = self.model
        model.to(self.device)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=model.parameters(),
                              lr=self.config.lr * self.config.lrDecay ** (round_th / self.config.decayStep),
                              weight_decay=1e-4)

        mean_loss = []
        for epoch in range(self.config.epoch):
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
        keys = list(self.model.state_dict().keys())
        keys = list(reversed(keys))  # [top -> down(near the data)]
        tmp_params = self.get_params()
        for (key, value) in params.items():
            if keys.index(key) < 2 * self.config.depth:
                tmp_params[key] = value
        self.set_params(tmp_params)

    def update(self, client):
        self.user_id = client.user_id
        self.model.load_state_dict(client.model.state_dict())
        self.train_loader = client.train_loader
        self.test_loader = client.test_loader
