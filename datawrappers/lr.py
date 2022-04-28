import torch
from torch.utils.data import Dataset
import numpy as np

from FLAG import path_to_data


RECREATE = False
LR_DIM = 10
LR2_TRAIN_SIZE = 50000
LR2_DEV_SIZE = 1000


class LRLocal(Dataset):
    def __init__(self, weight, noise_var, size):
        self.weight = weight
        self.noise_var = noise_var
        self.dim = LR_DIM
        self.size = size
        self.X = []
        self.Y = []

        for _ in range(size):
            x = np.random.multivariate_normal(np.zeros((self.dim,)), np.eye(self.dim))
            y = np.dot(weight, x)
            y += np.random.normal(0, noise_var)
            self.X.append(x)
            self.Y.append([y])

        self.X = torch.tensor(self.X, dtype=torch.float)
        self.Y = torch.tensor(self.Y, dtype=torch.float)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return self.size


LRDev = LRLocal

def __create_lr2_data():
    import os, json
    if not os.path.exists(path_to_data + '/lr2set'):
        os.makedirs(path_to_data + '/lr2set')
    elif not RECREATE:
        return

    weight_var = 10.
    weight_vec = [np.random.multivariate_normal(np.zeros((LR_DIM,)), weight_var * np.eye(LR_DIM)),
                  np.random.multivariate_normal(np.zeros((LR_DIM,)), weight_var * np.eye(LR_DIM))]
    noise_var = 1.
    dim = LR_DIM
    size_train = LR2_TRAIN_SIZE
    size_dev = LR2_DEV_SIZE

    for i in range(2):
        for filename, size in zip(['train_' + str(i), 'dev_' + str(i)], [size_train, size_dev]):
            X, Y = [], []
            for _ in range(size):
                x = np.random.multivariate_normal(np.zeros((dim,)), np.eye(dim))
                y = np.dot(weight_vec[i], x)
                y += np.random.normal(0, noise_var)
                X.append(x)
                Y.append([y])
            X = torch.tensor(X, dtype=torch.float)
            Y = torch.tensor(Y, dtype=torch.float)
            data = {'X': X, 'Y': Y}
            torch.save(data, path_to_data + '/lr2set/{}.pt'.format(filename))

    param_dict = {
        'weight_var': weight_var,
        'weight_vec': [weight.tolist() for weight in weight_vec],
        'noise_var': noise_var,
        'dim': dim,
        'size_train': size_train,
        'size_dev': size_dev
    }

    with open(path_to_data + '/lr2set/param.json', 'w') as f:
        json.dump(param_dict, f, indent=4)


__create_lr2_data()


class LR2ALocal(Dataset):
    data = torch.load(path_to_data + '/lr2set/train_0.pt')

    def __init__(self, indices):
        self.indices = indices
        self.size = len(indices)

    def __getitem__(self, item):
        return LR2ALocal.data['X'][item], LR2ALocal.data['Y'][item]

    def __len__(self):
        return self.size


class LR2BLocal(Dataset):
    data = torch.load(path_to_data + '/lr2set/train_1.pt')

    def __init__(self, indices):
        self.indices = indices
        self.size = len(indices)

    def __getitem__(self, item):
        return LR2BLocal.data['X'][item], LR2BLocal.data['Y'][item]

    def __len__(self):
        return self.size


class LR2ADev(Dataset):
    data = torch.load(path_to_data + '/lr2set/dev_0.pt')

    def __init__(self):
        self.size = len(LR2ADev.data['X'])

    def __getitem__(self, item):
        return LR2ADev.data['X'][item], LR2ADev.data['Y'][item]

    def __len__(self):
        return self.size


class LR2BDev(Dataset):
    data = torch.load(path_to_data + '/lr2set/dev_1.pt')

    def __init__(self):
        self.size = len(LR2BDev.data['X'])

    def __getitem__(self, item):
        return LR2BDev.data['X'][item], LR2BDev.data['Y'][item]

    def __len__(self):
        return self.size


class LR2Sampler:
    def sample(self, size):
        return np.random.choice(range(LR2_TRAIN_SIZE), size)
