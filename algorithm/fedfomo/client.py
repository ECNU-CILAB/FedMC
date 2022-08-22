import torch
from copy import deepcopy
from utils.flutils import *
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np


class CLIENT:
    def __init__(self, user_id, train_loader, test_loader, config):
        self.config = config
        self.user_id = user_id
        use_cuda = config.cuda and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{config.cudaNo}" if use_cuda else "cpu")
        self.model = select_model(algorithm=config.algorithm, model_name=config.model)
        self.train_loader = train_loader
        self.test_loader = test_loader
        if self.config.valFomo and self.train_loader:
            # split train/val/test
            self.train_full_dataset = self.train_loader.dataset
            self.train_full_size = len(self.train_full_dataset)
            self.train_size = int(0.8 * self.train_full_size)
            self.val_size = self.train_full_size - self.train_size
            train_dataset, val_dataset = torch.utils.data.random_split(self.train_full_dataset, [self.train_size, self.val_size])
            self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.config.batchSize, shuffle=True, num_workers=0)
            self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.config.batchSize, shuffle=True, num_workers=0)

        self.stats = {
            'train-samples': 0,
            'test-samples': 0,
            'GM-train-accuracy': 0,
            'GM-test-accuracy': 0,
            'GM-train-loss': None,
            'GM-test-loss': None
        }
        self.federating_clients = None

    @property
    def train_samples_num(self):
        return len(self.train_loader.dataset) if self.train_loader else None

    @property
    def test_samples_num(self):
        return len(self.test_loader.dataset) if self.test_loader else None

    def update_client_weights(self):
        baseline_model = deepcopy(self.model)
        comparison_models = []
        federating_clients_ids = []
        for idx, fed_client in enumerate(self.federating_clients):
            comparison_m = deepcopy(fed_client.model)
            comparison_models.append(comparison_m)
            federating_clients_ids.append(fed_client.user_id)

        weights = self.compute_fomo_weights(baseline_model=baseline_model, comparison_models=comparison_models)

        # update model weights and client-to-client weights
        normalization_factor = np.abs(np.sum(weights))
        if normalization_factor < 1e-9:
            normalization_factor += 1e-9
        all_deltas_normed = list(np.array(weights) / normalization_factor)

        # only consider positive models
        positive_deltas_all = []
        positive_models = []
        for idx, item in enumerate(all_deltas_normed):
            if item > 0:
                positive_deltas_all.append(item)
                positive_models.append(comparison_models[idx])
        if np.sum(positive_deltas_all) > 0:
            positive_deltas_all = np.array(positive_deltas_all) / np.sum(positive_deltas_all)
        else:
            print(f'No models performed higher than baseline for client {self.user_id}')
            positive_deltas_all = [1.]
            positive_models = [baseline_model]
        updates = []
        for idx in range(len(positive_deltas_all)):
            updates.append((positive_deltas_all[idx], deepcopy(positive_models[idx].state_dict())))
        personalized_params = fed_average(updates)
        self.model.load_state_dict(personalized_params)
        return federating_clients_ids, all_deltas_normed

    def train(self, round_th):
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

        train_samples_num, update = self.train_samples_num, self.get_params()

        if np.isnan(sum(mean_loss) / len(mean_loss)):
            print(f"client {self.user_id}, loss NAN")
            exit(0)

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

    def compute_fomo_weights(self, baseline_model, comparison_models):
        loss_deltas = []
        weight_deltas = []

        model = self.model.to(self.device)
        model.load_state_dict(baseline_model.state_dict())
        model.eval()
        if self.config.valFomo:
            _, _, baseline_loss = self.test(dataset='val')
        else:
            _, _, baseline_loss = self.test(dataset='train')
        for comparison_model in comparison_models:
            model.load_state_dict(comparison_model.state_dict())
            model.eval()
            if self.config.valFomo:
                _, _, comparison_loss = self.test(dataset='val')
            else:
                _, _, comparison_loss = self.test(dataset='train')
            loss_deltas.append(baseline_loss - comparison_loss)

        for comparison_model in comparison_models:
            weight_delta = self.compute_parameter_difference(comparison_model.cpu(), baseline_model.cpu(), 'l1')
            weight_deltas.append(weight_delta)
        # Should broadcast correctly
        return [loss_deltas[i] / weight_deltas[i] for i in range(len(loss_deltas))]

    def test(self, dataset='test'):
        model = self.model
        model.eval()
        model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()

        if dataset == 'test':
            data_loader = self.test_loader
        elif dataset == 'train':
            data_loader = self.train_loader
        else:
            data_loader = self.val_loader

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
            if key.startswith('shared'):
                tmp_params[key] = value
        self.set_params(tmp_params)

    def update(self, client):
        self.user_id = client.user_id
        self.federating_clients = client.federating_clients
        self.model.load_state_dict(client.model.state_dict())
        self.train_loader = client.train_loader
        self.test_loader = client.test_loader
        if self.config.valFomo:
            self.val_loader = client.val_loader

    @staticmethod
    def compute_parameter_difference(model_a, model_b, norm='l2'):
        """
        Compute difference between two model parameters
        """
        if norm == 'l1':
            total_diff = 0.
            total_diff_l2 = 0.
            # Compute L1-norm, i.e. ||w_a - w_b||_1
            for w_a, w_b in zip(model_a.parameters(), model_b.parameters()):
                total_diff += (w_a - w_b).norm(1).item()
                total_diff_l2 += torch.pow((w_a - w_b).norm(2), 2).item()
            return total_diff

        elif norm == 'l2_root':
            total_diff = 0.
            for w_a, w_b in zip(model_a.parameters(), model_b.parameters()):
                total_diff += (w_a - w_b).norm(2).item()
            return total_diff

        total_diff = 0.
        model_a_params = []
        for p in model_a.parameters():
            model_a_params.append(p.detach().cpu().numpy().astype(np.float64))

        for ix, p in enumerate(model_b.parameters()):
            p_np = p.detach().cpu().numpy().astype(np.float64)
            diff = model_a_params[ix] - p_np
            scalar_diff = np.sum(diff ** 2)
            total_diff += scalar_diff
        # Can be vectorized as
        # np.sum(np.power(model_a.parameters().detach().cpu().numpy() -
        #                 model_a.parameters().detach().cpu().numpy(), 2))
        return total_diff  # Returns distance^2 between two model parameters




