import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy

import FLAG


class Client:
    def __init__(self, ID, ds, solver, tag=''):
        self.ID = ID
        self.tag = tag

        self.ds = ds
        self.num_samples = len(ds)
        self.solver = solver
        self.loader = DataLoader(self.ds, batch_size=self.solver.batch_size)

        self.random = np.random.RandomState(seed=ID)
        self.server = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

        self.importance_estimated = []

    def attach_to_server(self, server):
        self.server = server
        self.model = copy.deepcopy(server.client_init_model)
        self.model = self.model.to(FLAG.device)
        self.optimizer = self.solver.optimizer_type(params=self.model.parameters(), lr=self.solver.lr)
        self.lr_scheduler = self.solver.lr_scheduler_type(optimizer=self.optimizer,
                                                          step_size=self.solver.lr_step,
                                                          gamma=self.solver.lr_gamma)

    def estimate_importance_weights(self):
        with torch.no_grad():
            table = np.zeros((self.server.num_clusters, self.num_samples))
            start_idx = 0
            nst_cluster_sample_count = [0] * self.server.num_clusters
            sample_loader = DataLoader(self.ds, batch_size=256)
            for x, y in sample_loader:
                x = x.to(FLAG.device)
                y = y.to(FLAG.device)
                for s, cluster in enumerate(self.server.cluster_vec):
                    cluster.eval()
                    out = cluster(x)
                    if self.solver.classification:
                        out = out.view(-1, self.solver.num_classes)
                    elif self.solver.sequence:
                        out = out.view(-1, self.solver.num_tokens)
                        y = y.view(-1)
                    loss = self.solver.estimate_criterion(out, y)

                    table[s][start_idx:start_idx + len(x)] = loss.cpu()
                start_idx += len(x)

            min_loss_idx = np.argmin(table, axis=0)
            for s in range(self.server.num_clusters):
                nst_cluster_sample_count[s] += np.sum(min_loss_idx == s)
            for s in range(self.server.num_clusters):
                if nst_cluster_sample_count[s] == 0:
                    nst_cluster_sample_count[s] = self.solver.count_smoother * self.num_samples
            self.importance_estimated = np.array([1.0 * nst / self.num_samples for nst in nst_cluster_sample_count])
           
    def get_importance(self, count=True):
        if count:
            return [ust * self.num_samples for ust in self.importance_estimated]
        else:
            return self.importance_estimated

    def get_model_dict(self):
        return copy.deepcopy(self.model.state_dict())

    def run(self):
        self.model.train()
        self._local_train()
        self.model.eval()

    def _local_train(self):
        for _ in range(self.solver.local_epoch):
            for x, y in self.loader:
                x = x.to(FLAG.device)
                y = y.to(FLAG.device)
                out = self.model(x)
                if self.solver.classification:
                    out = out.view(-1, self.solver.num_classes)
                loss = self.solver.criterion(out, y)

                mse_loss = nn.MSELoss(reduction='sum')
                for i, cluster in enumerate(self.server.cluster_vec):
                    l2 = None
                    for (param_local, param_cluster) in zip(self.model.parameters(), cluster.parameters()):
                        if l2 is None:
                            l2 = mse_loss(param_local, param_cluster)
                        else:
                            l2 += mse_loss(param_local, param_cluster)
                    loss += self.solver.reg_weight / 2 * self.importance_estimated[i] * l2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def eval(self, model=None):
        if model is None:
            model = self.model
        model.eval()

        loss = 0.
        with torch.no_grad():
            for x, y in self.loader:
                x = x.to(FLAG.device)
                y = y.to(FLAG.device)
                out = model(x)
                loss += self.solver.criterion(out, y).item()
        return loss

    def save(self, path):
        filename = path + "client_" + str(self.ID)
        import pickle
        client_dict = {"ID": self.ID, "tag": self.tag}
        with open(filename + "_dict.pkl", "wb") as f:
            pickle.dump(client_dict, f)
        torch.save(self.ds, filename + "_ds.pth")

    def load(self, path):
        filename = path + "client_" + str(self.ID)
        import pickle
        with open(filename + "_dict.pkl", "rb") as f:
            client_dict = pickle.load(f)
        self.ID = client_dict["ID"]
        self.tag = client_dict["tag"]
        self.ds = torch.load(filename + "_ds.pth")
