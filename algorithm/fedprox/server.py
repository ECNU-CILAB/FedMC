import copy
import wandb
import numpy as np
from utils.flutils import *
from utils.tools import *
from algorithm.fedprox.client import CLIENT
from tqdm import tqdm


class SERVER:
    def __init__(self, config):
        self.config = config
        self.clients = self.setup_clients()
        self.surrogates = self.setup_surrogates()
        self.clientsTrainSamplesNum = {client.user_id: client.train_samples_num for client in self.clients}
        self.clientsTestSamplesNum = {client.user_id: client.test_sample_num for client in self.clients}
        self.selected_clients = []
        self.updates = []
        # affect server initialization
        setup_seed(config.seed)
        self.model = select_model(algorithm=self.config.algorithm, model_name=self.config.model)
        self.params = self.model.state_dict()

        self.GM_best_test_acc = -1

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
            self.selected_clients = self.select_clients(round_th=i)
            for k in range(len(self.selected_clients)):
                surrogate = self.surrogates[k]
                c = self.selected_clients[k]
                # surrogate <-- c
                surrogate.update(c)
                surrogate.set_shared_params(self.params)
                trainSamplesNum, update, loss = surrogate.train(round_th=i)
                # c <-- surrogate
                c.update(surrogate)
                self.updates.append((trainSamplesNum, copy.deepcopy(update)))

            # update global params
            self.params = fed_average(self.updates)

            if i == 0 or (i + 1) % self.config.evalInterval == 0:
                # print(f"\nRound {i}")
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
            surrogate.set_shared_params(self.params)
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