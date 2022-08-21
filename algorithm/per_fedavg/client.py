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
        self.model = select_model(algorithm=config.algorithm, model_name=config.model)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.message = {
            'samples': 0,
            'update': None,
            'loss': None
        }

        self.stats = {
            'train-samples': 0,
            'test-samples': 0,
            'GM-train-accuracy': 0,
            'GM-test-accuracy': 0,
            'GM-train-loss': None,
            'GM-test-loss': None,
            'PM-train-accuracy': 0,
            'PM-test-accuracy': 0,
            'PM-train-loss': None,
            'PM-test-loss': None
        }

        self.iter_train_loader = None

    @property
    def train_samples_num(self):
        return len(self.train_loader.dataset) if self.train_loader else None

    @property
    def test_samples_num(self):
        return len(self.test_loader.dataset) if self.test_loader else None

    def fine_tuning_test(self, round_th):
        # w/o. fine-tuning, GM(global model)
        train_samples, GM_train_acc, GM_train_loss = self.test(dataset='train')
        test_samples, GM_test_acc, GM_test_loss = self.test(dataset='test')
        # fine-tuning on training set，PM(personalized model)
        model = self.model
        model.to(self.device)
        model.train()
        # equals to randomly sampling a batch
        criterion = torch.nn.CrossEntropyLoss()
        # alpha（inner update）
        innerLr = self.config.innerUpdateLr * self.config.lrDecay ** (round_th / self.config.decayStep)
        optimizer = optim.SGD(params=model.parameters(), lr=innerLr)
        for step, (data, labels) in enumerate(self.train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            break  # fine-tuning only on a batch

        # fine-tuning finishes, test on train and test set
        _, PM_train_acc, PM_train_loss = self.test(dataset='train')
        _, PM_test_acc, PM_test_loss = self.test(dataset='test')

        self.stats.update({
            'train-samples': train_samples,
            'test-samples': test_samples,
            'GM-train-accuracy': GM_train_acc,
            'GM-test-accuracy': GM_test_acc,
            'GM-train-loss': GM_train_loss,
            'GM-test-loss': GM_test_loss,
            'PM-train-accuracy': PM_train_acc,
            'PM-test-accuracy': PM_test_acc,
            'PM-train-loss': PM_train_loss,
            'PM-test-loss': PM_test_loss
        })

    def get_next_batch(self):
        if not self.iter_train_loader:
            self.iter_train_loader = iter(self.train_loader)
        try:
            (X, y) = next(self.iter_train_loader)
        except StopIteration:
            self.iter_train_loader = iter(self.train_loader)
            (X, y) = next(self.iter_train_loader)
        return X, y

    def train(self, round_th):
        model = self.model
        model.to(self.device)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()
        innerLr = self.config.innerUpdateLr * self.config.lrDecay ** (round_th / self.config.decayStep)
        outerLr = self.config.outerUpdateLr * self.config.lrDecay ** (round_th / self.config.decayStep)
        meta_optimizer = optim.SGD(params=model.parameters(), lr=outerLr)  # beta

        # step2 loss list
        mean_loss = []
        for epoch in range(self.config.epoch):
            updates_nums = int(self.train_samples_num / self.config.batchSize)
            if updates_nums == 0:
                updates_nums = 1
            assert updates_nums != 0
            for iter_th in range(updates_nums):
                # Step 1 (corresponding to support set in MAML)
                data, labels = self.get_next_batch()
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = model(data)
                loss = criterion(output, labels)
                if self.config.secondOrder:
                    one_order_gradients = torch.autograd.grad(outputs=loss, inputs=model.parameters(),
                                                              create_graph=True)
                else:
                    # first-order approximation
                    one_order_gradients = torch.autograd.grad(outputs=loss, inputs=model.parameters())

                fast_weights = list(map(lambda p: p[1] - innerLr * p[0], zip(one_order_gradients, model.parameters())))

                # Step 2 (corresponding to query set in MAML)
                data, labels = self.get_next_batch()
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = model(data, fast_weights)
                loss = criterion(output, labels)
                meta_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                meta_optimizer.step()
                mean_loss.append(loss.item())

        train_samples_num, update = self.train_samples_num, self.get_params()

        if np.isnan(sum(mean_loss) / len(mean_loss)):
            print(f"client {self.user_id}, loss NAN")
            exit(0)
        self.message.update({
            'samples': train_samples_num,
            'update': update,
            'loss': sum(mean_loss) / len(mean_loss)
        })

    def test(self, dataset='test'):
        model = self.model
        model.eval()
        model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()

        if dataset == 'test':
            dataLoader = self.test_loader
        else:
            dataLoader = self.train_loader

        total_right = 0
        total_samples = 0
        mean_loss = []
        with torch.no_grad():
            for step, (data, labels) in enumerate(dataLoader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = model(data)
                loss = criterion(output, labels)
                mean_loss.append(loss.item())
                output = torch.argmax(output, dim=-1)
                total_right += torch.sum(output == labels)
                total_samples += len(labels)
            acc = float(total_right) / total_samples
        print(f"client {self.user_id} ends testing on {dataset} set.")
        return total_samples, acc, sum(mean_loss) / len(mean_loss)

    def get_params(self):
        return self.model.cpu().state_dict()

    def set_params(self, model_params):
        self.model.load_state_dict(model_params)

    def set_shared_params(self, params):
        self.set_params(params)

    def update(self, client):
        self.user_id = client.user_id
        self.model.load_state_dict(client.model.state_dict())
        self.train_loader = client.train_loader
        self.test_loader = client.test_loader
        self.message = client.message
        self.stats = client.stats
