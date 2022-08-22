import copy
import numpy as np
from utils.flutils import *
from utils.tools import *
from algorithm.fedfomo.client import CLIENT
from tqdm import tqdm
import time


class SERVER:
    def __init__(self, config):
        self.config = config
        self.clients = self.setup_clients()
        self.surrogates = self.setup_surrogates()
        self.selected_clients = []
        self.updates = []
        # affect server initialization
        setup_seed(config.seed)
        self.model = select_model(algorithm=self.config.algorithm, model_name=self.config.model)
        self.params = self.model.state_dict()
        # affinity matrix P composed of vectors p_i = <p_i1, ..., p_iK>, totally K clients
        # self weight 0.1
        self.P = np.zeros((len(self.clients), len(self.clients))) + 0.1 * np.identity(len(self.clients))
        self.federating_clients = None
        self.all_activated_clients = []
        self.GM_best_test_acc = -1
        self.starting_epsilon = 0.3
        self.epsilon_decay = 0.003

    def get_federating_clients(self, num_query_clients=None, round_th=None):
        """
        :param num_query_clients:
        :param round_th:
        :return:
        """
        for client in self.selected_clients:
            queried_clients = []
            # e_greedy method
            possible_clients = [c for c in self.clients if c.user_id != client.user_id]
            possible_ids = [c.user_id for c in possible_clients]
            client_weights = [self.P[client.user_id][c_id] for c_id in possible_ids]

            # argsort but with random tie-breaking
            random_vals = np.random.random(len(client_weights))
            top_clients_ix = list(
                np.lexsort((random_vals, client_weights))[::-1])  # sort by client_weights then by random_vals

            # Essentially just shuffle here
            rand_clients = list(np.random.choice(possible_clients, size=len(possible_clients), replace=False))

            for ix in range(num_query_clients):
                explore = np.random.uniform(0, 1)
                current_epsilon = max([self.starting_epsilon - round_th * self.epsilon_decay, 0])
                # Loop through until we get a suitable client
                client_chosen = False
                while not client_chosen:
                    # If exploring, take the first random client and remove
                    if explore < current_epsilon:
                        possible_client = rand_clients.pop(0)
                    else:  # Otherwise take the first top client and remove
                        possible_client = possible_clients[top_clients_ix.pop(0)]
                    if possible_client not in queried_clients:
                        queried_clients.append(possible_client)
                        client_chosen = True
            client.federating_clients = queried_clients

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
            self.get_federating_clients(num_query_clients=self.config.numQueryClients, round_th=i)

            for k in range(len(self.selected_clients)):
                surrogate = self.surrogates[k]
                c = self.selected_clients[k]
                # surrogate <-- c
                surrogate.update(c)
                federated_clients_ids, all_deltas_normed = surrogate.update_client_weights()
                print(federated_clients_ids)
                print(all_deltas_normed)
                for idx, delta in enumerate(all_deltas_normed):
                    fed_client_id = federated_clients_ids[idx]
                    self.P[surrogate.user_id][fed_client_id] += delta
                trainSamplesNum, update, loss = surrogate.train(round_th=i)
                # c <-- surrogate
                c.update(surrogate)
                self.updates.append((trainSamplesNum, copy.deepcopy(update)))

            # update global params
            self.params = fed_average(self.updates)
            end_time = time.time()
            print(f"training costs {end_time - start_time}(s)")
            if i == 0 or (i + 1) % self.config.evalInterval == 0:
                # print(f"\nRound {i}")
                # test on training and test set with global and fine-tuned model(fine-tuning for one local epoch)
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
