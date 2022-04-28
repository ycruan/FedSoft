from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import parameters_to_vector

import FLAG


class ValidatorConfig:
    def __init__(self, num_class=10, num_epochs=200, verbose=True, test_ds_batch_size=50, do_client_model_compare=False,
                 do_importance_estimation=True, do_client_eval=True, do_cluster_eval=True, client_eval_idx_vec=None):
        if client_eval_idx_vec is None:
            client_eval_idx_vec = [0, -1]
        self.num_class = num_class
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.test_ds_batch_size = test_ds_batch_size
        self.do_client_model_compare = do_client_model_compare
        self.do_importance_estimation = do_importance_estimation
        self.do_client_eval = do_client_eval
        self.do_cluster_eval = do_cluster_eval
        self.client_eval_idx_vec = client_eval_idx_vec


class ValidatorClassification:
    def __init__(self, test_ds_vec, config):
        self.config = config
        self.verbose = config.verbose
        self.num_class = config.num_class
        self.test_loader_vec = []
        for ds in test_ds_vec:
            self.test_loader_vec.append(DataLoader(ds, batch_size=config.test_ds_batch_size))

        # populate class_total_vec
        self.class_total_vec = []
        for loader in self.test_loader_vec:
            class_total = list(0. for _ in range(self.num_class))
            for _, y in loader:
                for i in range(len(y)):
                    class_total[y[i]] += 1
            for i in range(self.num_class):
                if class_total[i] == 0:
                    class_total[i] = -1
            self.class_total_vec.append(class_total)

    def validate(self, client_vec, cluster_vec, t):
        with torch.no_grad():
            validation_dict = {'round': t,
                               'client_model_div_mat': None,
                               'importance_estimation': {},
                               'client_eval': {},
                               'cluster_eval': {}, }
            info = '-' * 30 + ' Round {} '.format(t) + '-' * 30 + '\n'
            mse_loss = nn.MSELoss(reduction='sum')

            if self.config.do_client_model_compare:
                l2_mat = np.zeros((len(client_vec), len(client_vec)))
                for i in range(len(client_vec)):
                    for j in range(i + 1, len(client_vec)):
                        l2 = mse_loss(parameters_to_vector(client_vec[i].model.parameters()),
                                      parameters_to_vector(client_vec[j].model.parameters())).item()
                        l2_mat[i][j] = l2
                        l2_mat[j][i] = l2
                np.set_printoptions(precision=3)
                info += 'l2 mat = \n{}'.format(l2_mat)
                validation_dict['client_model_div_mat'] = l2_mat

            if self.config.do_importance_estimation:
                for client in client_vec:
                    info += 'client {} importance estimation = {}\n'.format(client.ID, client.importance_estimated.tolist())
                    validation_dict['importance_estimation']['client_' + str(client.ID)] = client.importance_estimated.tolist()

            if self.config.do_client_eval:
                client_set = [client_vec[i] for i in self.config.client_eval_idx_vec]
                if t == self.config.num_epochs-1:
                    client_set = client_vec
                for i, client in enumerate(client_set):
                    client.model.eval()
                    info += '*' * 10 + ' Client {} '.format(client.ID) + '*' * 10 + '\n'
                    validation_dict['client_eval']['client_' + str(client.ID)] = {}

                    class_correct = list(0. for _ in range(self.num_class))
                    class_total = list(0. for _ in range(self.num_class))
                    for _, y in client.loader:
                        for i in range(len(y)):
                            class_total[y[i]] += 1
                    for k in range(self.num_class):
                        if class_total[k] == 0:
                            class_total[k] = -1
                    for x, y in client.loader:
                        x = x.to(FLAG.device)
                        y = y.to(FLAG.device)
                        out = client.model(x).view(-1, self.num_class)
                        _, predicted = torch.max(out, 1)
                        correct = (predicted == y).squeeze()
                        if len(y) == 1:
                            class_correct[y.item()] += correct.item()
                        else:
                            for k in range(len(y)):
                                class_correct[y[k]] += correct[k].item()

                    class_accuracy = list(
                        1.0 * class_correct[k] / class_total[k] for k in range(self.num_class))
                    average_accuracy = 1.0 * sum(class_correct) / sum(class_total)
                    class_accuracy = [round(acc, 3) for acc in class_accuracy]
                    info += 'Client {0:d} has average ' \
                            'accuracy = {1:.3f}, class_accuracy = {2:}\n'.format(client.ID, average_accuracy,
                                                                                 class_accuracy)
                    validation_dict['client_eval']['client_' + str(client.ID)]['average_accuracy'] = average_accuracy
                    validation_dict['client_eval']['client_' + str(client.ID)]['class_accuracy'] = class_accuracy

            if self.config.do_cluster_eval:
                for i, cluster in enumerate(cluster_vec):
                    cluster.eval()
                    info += '*' * 10 + ' Cluster {} '.format(i) + '*' * 10 + '\n'
                    validation_dict['cluster_eval']['cluster_' + str(i)] = {}
                    for j, loader in enumerate(self.test_loader_vec):
                        class_correct = list(0. for _ in range(self.num_class))
                        for x, y in loader:
                            x = x.to(FLAG.device)
                            y = y.to(FLAG.device)
                            out = cluster(x)
                            _, predicted = torch.max(out, 1)
                            correct = (predicted == y).squeeze()
                            for k in range(len(y)):
                                class_correct[y[k]] += correct[k].item()

                        class_accuracy = list(
                            1.0 * class_correct[k] / self.class_total_vec[j][k] for k in range(self.num_class))
                        average_accuracy = 1.0 * sum(class_correct) / sum(self.class_total_vec[j])
                        class_accuracy = [round(acc, 3) for acc in class_accuracy]
                        info += 'Cluster {0:d} - test ds {1:d} has average ' \
                                'accuracy = {2:.3f}, class_accuracy = {3:}\n'.format(i, j, average_accuracy,
                                                                                     class_accuracy)
                        validation_dict['cluster_eval']['cluster_' + str(i)]['test_ds_' + str(j)] = {}
                        validation_dict['cluster_eval']['cluster_' + str(i)]['test_ds_' + str(j)][
                            'average_accuracy'] = average_accuracy
                        validation_dict['cluster_eval']['cluster_' + str(i)]['test_ds_' + str(j)][
                            'class_accuracy'] = class_accuracy
            if self.verbose:
                print(info)
            return validation_dict


class ValidatorRegression:
    def __init__(self, test_ds_vec, config):
        self.config = config
        self.verbose = config.verbose
        self.test_loader_vec = []
        for ds in test_ds_vec:
            self.test_loader_vec.append(DataLoader(ds, batch_size=config.test_ds_batch_size))

    def validate(self, client_vec, cluster_vec, t):
        with torch.no_grad():
            validation_dict = {'round': t,
                               'client_model_div_mat': None,
                               'importance_estimation': {},
                               'client_eval': {},
                               'cluster_eval': {}, }
            info = '-' * 30 + ' Round {} '.format(t) + '-' * 30 + '\n'
            mse_loss = nn.MSELoss(reduction='sum')

            if self.config.do_client_model_compare:
                l2_mat = np.zeros((len(client_vec), len(client_vec)))
                for i in range(len(client_vec)):
                    for j in range(i + 1, len(client_vec)):
                        l2 = mse_loss(parameters_to_vector(client_vec[i].model.parameters()),
                                      parameters_to_vector(client_vec[j].model.parameters())).item()
                        l2_mat[i][j] = l2
                        l2_mat[j][i] = l2
                np.set_printoptions(precision=3)
                info += 'l2 mat = \n{}'.format(l2_mat)
                validation_dict['client_model_div_mat'] = l2_mat

            if self.config.do_importance_estimation:
                for client in client_vec:
                    info += 'client {} importance estimation = {}\n'.format(client.ID, client.importance_estimated)
                    validation_dict['importance_estimation']['client_' + str(client.ID)] = client.importance_estimated

            if self.config.do_client_eval:
                client_set = [client_vec[i] for i in self.config.client_eval_idx_vec]
                if t == self.config.num_epochs-1:
                    client_set = client_vec
                for i, client in enumerate(client_set):
                    client.model.eval()
                    info += '*' * 10 + ' Client {} '.format(i) + '*' * 10 + '\n'
                    validation_dict['client_eval']['client_' + str(client.ID)] = {}
                    loader = client.loader
                    loss = 0.
                    for x, y in loader:
                        x = x.to(FLAG.device)
                        y = y.to(FLAG.device)
                        out = client.model(x)
                        loss += mse_loss(y, out).item()
                    average_loss = loss / len(loader) / self.config.test_ds_batch_size
                    info += 'Client {0:d} has average ' \
                            'loss = {1:.3f}\n'.format(i, average_loss)
                    validation_dict['client_eval']['client_' + str(client.ID)]['average_accuracy'] = average_loss

            if self.config.do_cluster_eval:
                for i, cluster in enumerate(cluster_vec):
                    cluster.eval()
                    info += '*' * 10 + ' Cluster {} '.format(i) + '*' * 10 + '\n'
                    validation_dict['cluster_eval']['cluster_' + str(i)] = {}
                    for j, loader in enumerate(self.test_loader_vec):
                        loss = 0.
                        for x, y in loader:
                            x = x.to(FLAG.device)
                            y = y.to(FLAG.device)
                            out = cluster(x)
                            loss += mse_loss(y, out).item()
                        average_loss = loss / len(loader) / self.config.test_ds_batch_size
                        info += 'Cluster {0:d} - test ds {1:d} has average ' \
                                'accuracy = {2:.3f}\n'.format(i, j, average_loss)
                        validation_dict['cluster_eval']['cluster_' + str(i)]['test_ds_' + str(j)] = {}
                        validation_dict['cluster_eval']['cluster_' + str(i)]['test_ds_' + str(j)][
                            'average_accuracy'] = average_loss

            if self.verbose:
                print(info)
            return validation_dict
