import copy
import numpy as np
from utils.flutils import *
from utils.tools import *
from algorithm.mocha.client import CLIENT
from tqdm import tqdm
import time


class SERVER:
    def __init__(self, config):
        self.config = config
        use_cuda = config.cuda and torch.cuda.is_available()
        self.device = torch.device(f"cuda:{config.cudaNo}" if use_cuda else "cpu")
        self.clients = self.setup_clients()
        self.surrogates = self.setup_surrogates()
        self.selected_clients = []
        self.updates = []
        # affect server initialization
        setup_seed(config.seed)
        self.model = select_model(algorithm=self.config.algorithm, model_name=self.config.model)
        self.params = self.model.state_dict()
        self.GM_best_test_acc = -1

        self.model.train()
        num_layers_keep = config.numLayersKeep
        weight_keys = list(self.params.keys())
        self.weight_keys = weight_keys
        total_num_layers = len(weight_keys)
        w_glob_keys = self.weight_keys[total_num_layers - num_layers_keep:]
        self.w_glob_keys = w_glob_keys
        num_param_glob = 0
        num_param_local = 0
        for key in self.model.state_dict().keys():
            num_param_local += self.model.state_dict()[key].numel()
            if key in w_glob_keys:
                num_param_glob += self.model.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))

        self.m = self.config.clientsPerRound
        I = torch.ones((self.m, self.m))
        i = torch.ones((self.m, 1))
        omega = I - 1 / self.m * i.mm(i.T)
        omega = omega ** 2
        self.omega = omega.to(self.device)

        W = [self.params[key].flatten() for key in w_glob_keys]
        W = torch.cat(W)
        d = len(W)
        self.d = d

    def setup_clients(self):
        users, train_loaders, test_loaders = setup_datasets(dataset=self.config.dataset,
                                                            batch_size=self.config.batchSize,
                                                            alpha=self.config.alpha)
        clients = [
            CLIENT(user_id=user_id,
                   train_loader=train_loaders[user_id],
                   test_loader=test_loaders[user_id],
                   config=self.config)
            for user_id in users]
        return clients

    def select_clients(self, round_th):
        np.random.seed(seed=self.config.seed + round_th)
        return np.random.choice(self.clients, self.config.clientsPerRound, replace=False)

    def setup_surrogates(self):
        surrogates = [
            CLIENT(user_id=i,
                   train_loader=None,
                   test_loader=None,
                   config=self.config)
            for i in range(self.config.clientsPerRound)]
        return surrogates

    def federate(self):
        print(f"Training with {len(self.clients)} clients!")
        for i in tqdm(range(self.config.numRounds)):
            start_time = time.time()
            self.selected_clients = self.select_clients(round_th=i)

            W = torch.zeros((self.d, self.m)).to(self.device)  # [| | | |]
            # update W
            for idx, c in enumerate(self.selected_clients):
                W_local = [c.model.state_dict()[key].flatten() for key in self.w_glob_keys]
                W_local = torch.cat(W_local)
                W[:, idx] = W_local

            for k in range(len(self.selected_clients)):
                surrogate = self.surrogates[k]
                c = self.selected_clients[k]
                # surrogate <-- c
                surrogate.update(c)
                # update local model
                trainSamplesNum, update, loss = surrogate.train(round_th=i, omega=self.omega, W_glob=W.clone(),
                                                                w_glob_keys=self.w_glob_keys, idx=k)
                # c <-- surrogate
                c.update(surrogate)
                self.updates.append((trainSamplesNum, copy.deepcopy(update)))

            end_time = time.time()
            print(f"training costs {end_time - start_time}(s)")
            if i == 0 or (i + 1) % self.config.evalInterval == 0:
                GM_training_acc_list, GM_training_loss_list, GM_test_acc_list, GM_test_loss_list = self.test()
                # print and log
                self.print_and_log(i, GM_training_acc_list, GM_training_loss_list, GM_test_acc_list, GM_test_loss_list)
            self.updates = []

    def test(self):
        GM_training_acc_list, GM_training_loss_list = [], []
        GM_test_acc_list, GM_test_loss_list = [], []
        surrogate = self.surrogates[0]
        for c in self.clients:
            surrogate.update(c)
            surrogate.fine_tuning_test()
            GM_training_acc_list.append(
                (surrogate.stats['train-samples'], surrogate.stats['GM-train-accuracy']))
            GM_training_loss_list.append((surrogate.stats['train-samples'], surrogate.stats['GM-train-loss']))
            GM_test_acc_list.append((surrogate.stats['test-samples'], surrogate.stats['GM-test-accuracy']))
            GM_test_loss_list.append((surrogate.stats['test-samples'], surrogate.stats['GM-test-loss']))
        return GM_training_acc_list, GM_training_loss_list, GM_test_acc_list, GM_test_loss_list

    def print_and_log(self, round_th, GM_training_acc_list, GM_training_loss_list, GM_test_acc_list, GM_test_loss_list):
        GM_trainingAcc = avg_metric(GM_training_acc_list)
        GM_trainingLoss = avg_metric(GM_training_loss_list)
        GM_testAcc = avg_metric(GM_test_acc_list)
        GM_testLoss = avg_metric(GM_test_loss_list)

        # update best test acc
        if GM_testAcc > self.GM_best_test_acc:
            self.GM_best_test_acc = GM_testAcc

        # post data error, encoder error, trainingAcc. format
        summary = {
            "round": round_th,
            "GMTrainingAcc": GM_trainingAcc,
            "GMTestAcc": GM_testAcc,
            "GMTrainingLoss": GM_trainingLoss,
            "GMTestLoss": GM_testLoss,
            "GMBestTestAcc": self.GM_best_test_acc
        }
        for tag, value in summary.items():
            self.config.writer.add_scalar(tag, value, round_th)
