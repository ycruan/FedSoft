import copy

import torch
import numpy as np

from FLAG import device, validation_data_flush_interval
from utils import logger


class Server:
    def __init__(self, model_fn, client_vec, num_clusters, validator, server_solver, exp_id='default_exp'):
        self.model_fn = model_fn
        self.client_vec = client_vec
        self.validator = validator
        self.server_solver = server_solver
        self.exp_id = exp_id
        self.client_init_model = model_fn()

        self.num_clients = len(client_vec)
        self.num_clusters = num_clusters
        self.cluster_vec = []
        for _ in range(self.num_clusters):
            self.cluster_vec.append(model_fn())
        for i in range(self.num_clusters):
            self.cluster_vec[i] = self.cluster_vec[i].to(device)

        self.importance_weights_matrix = []
        self._zero_weights = None

        self.num_samples_vec = np.array([])
        for client in client_vec:
            client.attach_to_server(server=self)
            self.num_samples_vec = np.append(self.num_samples_vec, client.num_samples)
        self.num_samples = np.sum(self.num_samples_vec)
        self.number_samples_fraction_vec = self.num_samples_vec / self.num_samples

    def get_cluster_model(self, idx):
        return copy.deepcopy(self.cluster_vec[idx].state_dict())

    def generate_zero_weights(self):
        if self._zero_weights is None:
            self._zero_weights = {}
            for key, val in self.client_init_model.state_dict().items():
                self._zero_weights[key] = torch.zeros(size=val.shape, dtype=torch.float32)
        return copy.deepcopy(self._zero_weights)

    def run(self, num_global_epochs):
        self._run_fedsoft(num_global_epochs)

    def _run_fedsoft(self, num_global_epochs):
        validation_dict = {}
        for t in range(num_global_epochs):
            # Importance estimation
            if t % self.server_solver.estimation_interval == 0:
                self.importance_weights_matrix = []  # dim = (num_clients, num_clusters)
                for client in self.client_vec:
                    client.estimate_importance_weights('fedsoft')
                    self.importance_weights_matrix.append(client.get_importance())
                self.importance_weights_matrix = np.array(self.importance_weights_matrix)
                self.importance_weights_matrix /= np.sum(self.importance_weights_matrix, axis=0)

            # Client selection
            selection = []
            if self.server_solver.do_selection:
                for s in range(self.num_clusters):
                    selection.append(np.random.choice(a=range(self.num_clients), size=self.server_solver.selection_size,
                                                      p=self.importance_weights_matrix[:, s], replace=False).tolist())
                logger.log_client_selection(self.exp_id, t, self._idx_to_id(selection))
            else:
                selection = np.tile(range(self.num_clients), reps=(self.num_clusters, 1))

            # Local updates
            for k in np.unique(np.concatenate(selection).ravel()):
                self.client_vec[k].run()

            # Aggregation
            self._aggregate_fedsoft(selection)

            # Validation
            if self.validator is not None:
                validation_dict[str(t)] = self.validator.validate(client_vec=self.client_vec,
                                                                  cluster_vec=self.cluster_vec, t=t)
                if t % validation_data_flush_interval == 0 or t == num_global_epochs - 1:
                    logger.log_validation_data(self.exp_id, validation_dict)

    def _aggregate_fedsoft(self, selection):
        for s in range(self.num_clusters):
            next_weights = self.generate_zero_weights()
            for k in selection[s]:
                if self.server_solver.do_selection:
                    aggregation_weight = 1. / self.server_solver.selection_size
                else:
                    aggregation_weight = self.importance_weights_matrix[k][s]
                client_weights = self.client_vec[k].get_model_dict()
                for key in next_weights.keys():
                    next_weights[key] += aggregation_weight * client_weights[key].cpu()
            self.cluster_vec[s].load_state_dict(state_dict=next_weights)

    def _idx_to_id(self, mat):
        return mat
        # retval = []
        # for vec in mat:
        #     retval.append([self.client_vec[k].ID for k in vec])
        # return retval
