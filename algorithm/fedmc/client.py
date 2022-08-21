import torch
from utils.flutils import *
from torch import autograd
import torch.optim as optim
import numpy as np


class CLIENT:
    def __init__(self, user_id, train_loader, test_loader, config):
        self.config = config
        self.user_id = user_id
        use_cuda = config.cuda and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{config.cudaNo}" if use_cuda else "cpu")
        kwargs = {'dropout': self.config.dropout}
        self.model = select_model(algorithm=config.algorithm, model_name=config.model, mode=config.mode, **kwargs)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.stats = {
            'train-samples': 0,
            'test-samples': 0,
            'GM-train-accuracy': 0,
            'GM-test-accuracy': 0,
            'GM-train-loss': None,
            'GM-test-loss': None,
        }

    @property
    def train_samples_num(self):
        return len(self.train_loader.dataset) if self.train_loader else None

    @property
    def test_samples_num(self):
        return len(self.test_loader.dataset) if self.test_loader else None

    def calc_gradient_penalty(self, model, real_data, fake_data):
        assert not (real_data.requires_grad or fake_data.requires_grad)
        alpha = torch.rand(real_data.shape[0], 1)
        alpha = alpha.expand(real_data.shape[0], real_data.shape[1]).contiguous()
        alpha = alpha.to(self.device)
        interpolates = alpha * real_data + ((torch.ones_like(alpha) - alpha) * fake_data)
        interpolates = interpolates.to(self.device)
        interpolates = interpolates.clone().detach().requires_grad_(True)

        critic_interpolates = model.metaCritic(interpolates)

        gradients = autograd.grad(outputs=critic_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(critic_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def meta_train(self, round_th):
        model = self.model
        model.to(self.device)
        model.train()

        # frozen
        for (key, param) in model.named_parameters():
            if key.startswith('critic'):
                param.requires_grad = False

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
                g_feature, l_feature, g_value, l_value, output = model(data)
                clf_loss = criterion(output, labels)
                WD = (g_value - l_value).mean()
                gradient_penalty = self.calc_gradient_penalty(model, g_feature.data, l_feature.data)
                loss = clf_loss + self.config.mu * (- WD + self.config.omega * gradient_penalty)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                mean_loss.append(clf_loss.item())

        # unfrozen
        for (key, param) in model.named_parameters():
            if key.startswith('critic'):
                param.requires_grad = True

        # loss NAN detection
        if np.isnan(sum(mean_loss) / len(mean_loss)):
            print(f"client {self.user_id}, loss NAN")
            exit(0)

        train_samples_num, update = self.train_samples_num, self.get_params()
        return train_samples_num, update, sum(mean_loss) / len(mean_loss)

    def meta_test(self, round_th):
        model = self.model
        model.to(self.device)
        model.train()

        # frozen
        for (key, param) in model.named_parameters():
            if key.startswith('shared'):
                param.requires_grad = False

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
                g_feature, l_feature, g_value, l_value, output = model(data)
                clf_loss = criterion(output, labels)
                WD = (g_value - l_value).mean()
                gradient_penalty = self.calc_gradient_penalty(model, g_feature.data, l_feature.data)
                loss = clf_loss + self.config.mu * (- WD + self.config.omega * gradient_penalty)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                mean_loss.append(clf_loss.item())

        # unfrozen
        for (key, param) in model.named_parameters():
            if key.startswith('shared'):
                param.requires_grad = True

        # loss NAN detection
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
                g_feature, l_feature, g_value, l_value, output = model(data)
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

    def set_shared_critic_params(self, params):
        tmp_params = self.get_params()
        for (key, value) in params.items():
            if key.startswith('shared') or key.startswith('critic'):
                tmp_params[key] = value
        self.set_params(tmp_params)

    def update(self, client):
        self.model.load_state_dict(client.model.state_dict())
        self.train_loader = client.train_loader
        self.test_loader = client.test_loader
