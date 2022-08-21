import copy
import time
import wandb
import torch.nn as nn
import numpy as np
from utils.flutils import *
from utils.tools import *
from algorithm.per_fedavg.client import CLIENT
from tqdm import tqdm
from prettytable import PrettyTable
from threading import Thread


class SERVER:
    def __init__(self, config):
        self.config = config
        self.clients = self.setup_clients()
        self.surrogates = self.setup_surrogates()
        self.clientsTrainSamplesNum = {client.user_id: client.train_samples_num for client in self.clients}
        self.clientsTestSamplesNum = {client.user_id: client.test_samples_num for client in self.clients}
        self.selected_clients = []
        self.updates = []
        # affect server initialization
        setup_seed(config.seed)
        # using fedavg initialization
        self.tmp_model = select_model(algorithm='fedavg', model_name=self.config.model)
        self.model = select_model(algorithm=self.config.algorithm, model_name=self.config.model)
        self.model_init()

        self.params = self.model.state_dict()

        # BEST TEST ACCURACY
        self.GM_best_test_acc = -1
        self.PM_best_test_acc = -1

    def model_init(self):
        assert len(list(self.tmp_model.parameters())) == len(list(self.model.parameters()))
        for (tmp_param, model_param) in zip(self.tmp_model.parameters(), self.model.parameters()):
            model_param.data = tmp_param.data.clone()

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
            for i in range(self.config.threads)]
        return surrogates

    def federate(self):
        print(f"Training with {len(self.clients)} clients!")
        for i in tqdm(range(self.config.numRounds)):
            start_time = time.time()
            self.selected_clients = self.select_clients(round_th=i)
            surrogate_containers = []
            for k in range(len(self.selected_clients)):
                batch_index = k // len(self.surrogates)  # 放到surrogate_containers中的第几批clients
                surrogate_index = k % len(self.surrogates)
                surrogate = self.surrogates[surrogate_index]
                c = self.selected_clients[k]
                # surrogate <-- c
                surrogate.update(c)
                surrogate.set_shared_params(self.params)
                surrogate_containers.append(surrogate)
                # check
                if k == len(self.selected_clients) - 1 or len(surrogate_containers) == len(self.surrogates):
                    # 最后一批，或者surrogate_containers满
                    print(f"\n{[cc.user_id for cc in surrogate_containers]} is ready for training....")
                    threads = [Thread(target=surrogate.train, args=(i,)) for surrogate in surrogate_containers]
                    [t.start() for t in threads]
                    [t.join() for t in threads]
                    # 训练完，把代理客户端的参数更新到对应原客户端上
                    for in_container_index in range(len(surrogate_containers)):
                        c_index = batch_index * len(self.surrogates) + in_container_index
                        # c <-- surrogate
                        self.selected_clients[c_index].update(surrogate_containers[in_container_index])
                        self.updates.append(
                            (self.selected_clients[c_index].message['samples'],
                             copy.deepcopy(self.selected_clients[c_index].message['update'])))

                    surrogate_containers.clear()

            # update global params
            self.params = fed_average(self.updates)
            end_time = time.time()

            print(f"training costs {end_time - start_time}(s)")

            if i % self.config.evalInterval == 0 or (i + 1) == self.config.numRounds:
                # evaluate one step
                print("Evaluate global model with one step update.")
                # print(f"\nRound {i}")
                # test on training and test set
                GM_training_acc_list, GM_training_loss_list, GM_test_acc_list, GM_test_loss_list, \
                PM_training_acc_list, PM_training_loss_list, PM_test_acc_list, PM_test_loss_list = self.test(round_th=i)
                # print and log
                self.print_and_log(i, GM_training_acc_list, GM_training_loss_list, GM_test_acc_list, GM_test_loss_list,
                                 PM_training_acc_list, PM_training_loss_list, PM_test_acc_list, PM_test_loss_list, )
            self.updates = []

    def test(self, round_th):
        GM_training_acc_list, GM_training_loss_list = [], []
        GM_test_acc_list, GM_test_loss_list = [], []
        PM_training_acc_list, PM_training_loss_list = [], []
        PM_test_acc_list, PM_test_loss_list = [], []
        surrogate_containers = []
        for idx in range(len(self.clients)):
            su_index = idx % len(self.surrogates)
            surrogate = self.surrogates[su_index]
            surrogate.update(self.clients[idx])
            surrogate.set_shared_params(self.params)
            surrogate_containers.append(surrogate)

            # check
            if idx == len(self.clients) - 1 or len(surrogate_containers) == len(self.surrogates):
                # fine-tuning and test
                threads = [Thread(target=surrogate.fine_tuning_test, args=(round_th,)) for surrogate in
                           surrogate_containers]
                [t.start() for t in threads]
                [t.join() for t in threads]

                for surrogate in surrogate_containers:
                    GM_training_acc_list.append(
                        (surrogate.stats['train-samples'], surrogate.stats['GM-train-accuracy']))
                    GM_training_loss_list.append((surrogate.stats['train-samples'], surrogate.stats['GM-train-loss']))
                    GM_test_acc_list.append((surrogate.stats['test-samples'], surrogate.stats['GM-test-accuracy']))
                    GM_test_loss_list.append((surrogate.stats['test-samples'], surrogate.stats['GM-test-loss']))

                    PM_training_acc_list.append(
                        (surrogate.stats['train-samples'], surrogate.stats['PM-train-accuracy']))
                    PM_training_loss_list.append((surrogate.stats['train-samples'], surrogate.stats['PM-train-loss']))
                    PM_test_acc_list.append((surrogate.stats['test-samples'], surrogate.stats['PM-test-accuracy']))
                    PM_test_loss_list.append((surrogate.stats['test-samples'], surrogate.stats['PM-test-loss']))
                surrogate_containers.clear()
        return GM_training_acc_list, GM_training_loss_list, GM_test_acc_list, GM_test_loss_list, \
               PM_training_acc_list, PM_training_loss_list, PM_test_acc_list, PM_test_loss_list

    def print_and_log(self, round_th, GM_training_acc_list, GM_training_loss_list, GM_test_acc_list, GM_test_loss_list,
                    PM_training_acc_list, PM_training_loss_list, PM_test_acc_list, PM_test_loss_list):
        GM_trainingAcc = avg_metric(GM_training_acc_list)
        GM_trainingLoss = avg_metric(GM_training_loss_list)
        GM_testAcc = avg_metric(GM_test_acc_list)
        GM_testLoss = avg_metric(GM_test_loss_list)

        PM_trainingAcc = avg_metric(PM_training_acc_list)
        PM_trainingLoss = avg_metric(PM_training_loss_list)
        PM_testAcc = avg_metric(PM_test_acc_list)
        PM_testLoss = avg_metric(PM_test_loss_list)

        # update best test acc
        if GM_testAcc > self.GM_best_test_acc:
            self.GM_best_test_acc = GM_testAcc
        if PM_testAcc > self.PM_best_test_acc:
            self.PM_best_test_acc = PM_testAcc

        # post data error, encoder error, trainingAcc. format
        summary = {
            "round": round_th,
            "GMTrainingAcc": GM_trainingAcc,
            "GMTestAcc": GM_testAcc,
            "GMTrainingLoss": GM_trainingLoss,
            "GMTestLoss": GM_testLoss,
            "GMBestTestAcc": self.GM_best_test_acc,

            "PMTrainingAcc": PM_trainingAcc,
            "PMTestAcc": PM_testAcc,
            "PMTrainingLoss": PM_trainingLoss,
            "PMTestLoss": PM_testLoss,
            "PMBestTestAcc": self.PM_best_test_acc
        }
        for tag, value in summary.items():
            self.config.writer.add_scalar(tag, value, round_th)