import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

from FLAG import path_to_data


class MnistLocal(Dataset):
    data = datasets.MNIST(root=path_to_data,
                          train=True,
                          download=True,
                          transform=transforms.ToTensor())

    def __init__(self, indices):
        self.indices = indices
        self.size = len(self.indices)

    def __getitem__(self, item):
        return MnistLocal.data[self.indices[item]][0].view(1, -1), torch.tensor(MnistLocal.data[self.indices[item]][1])

    def __len__(self):
        return self.size


class MnistDev(Dataset):
    data = datasets.MNIST(root=path_to_data,
                          train=False,
                          download=True,
                          transform=transforms.ToTensor())

    def __init__(self):
        self.size = len(MnistDev.data)

    def __getitem__(self, item):
        return MnistDev.data[item][0].view(1, -1), MnistDev.data[item][1]

    def __len__(self):
        return self.size


class MnistSampler:
    def __init__(self):
        self.indices = np.load(path_to_data + '/mnist_indices.npy', allow_pickle=True)

    def sample(self, class_vec, num_samples):
        samples = list(0 for _ in range(num_samples))
        for i in range(num_samples):
            idx_class = np.random.randint(0, len(class_vec))
            idx_instance = np.random.randint(0, len(self.indices[class_vec[idx_class]]))
            samples[i] = self.indices[class_vec[idx_class]][idx_instance]
        return samples


def __create_mnist_indices():
    from os import path

    if path.exists(path_to_data + '/mnist_indices.npy'):
        return

    train_data = datasets.MNIST(root=path_to_data,
                                train=True,
                                download=True,
                                transform=transforms.ToTensor())
    indices = [[] for _ in range(10)]
    for i, (x, y) in enumerate(train_data):
        indices[y].append(i)

    np.save(path_to_data + '/mnist_indices.npy', indices)


__create_mnist_indices()
