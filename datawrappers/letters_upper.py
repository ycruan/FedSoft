import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torchvision.transforms.functional as TF

from FLAG import path_to_data


class RotationTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


def load_letters_upper_rotation_data(angle):
    return datasets.EMNIST(root=path_to_data,
                           split='byclass',
                           train=True,
                           download=True,
                           transform=transforms.Compose(
                               [transforms.ToTensor(), RotationTransform(angle)]))


def load_letters_upper_rotation_data_dev(angle):
    return datasets.EMNIST(root=path_to_data,
                           split='byclass',
                           train=False,
                           download=True,
                           transform=transforms.Compose(
                               [transforms.ToTensor(), RotationTransform(angle)]))


class LettersUpperBase(Dataset):
    data = None

    def __init__(self, indices):
        self.indices = indices
        self.size = len(self.indices)

    def __getitem__(self, item):
        return None, None

    def __len__(self):
        return self.size


class LettersUpperLocal(LettersUpperBase):
    data = datasets.EMNIST(root=path_to_data,
                           split='byclass',
                           train=True,
                           download=True,
                           transform=transforms.ToTensor())

    def __getitem__(self, item):
        return LettersUpperLocal.data[self.indices[item]][0].view(1, -1), LettersUpperLocal.data[self.indices[item]][
            1] - 10


class LettersUpperLocal90(LettersUpperBase):
    data = load_letters_upper_rotation_data(90)

    def __getitem__(self, item):
        return LettersUpperLocal90.data[self.indices[item]][0].view(1, -1), \
               LettersUpperLocal90.data[self.indices[item]][1] - 10


class LettersUpperLocal180(LettersUpperBase):
    data = load_letters_upper_rotation_data(180)

    def __getitem__(self, item):
        return LettersUpperLocal180.data[self.indices[item]][0].view(1, -1), \
               LettersUpperLocal180.data[self.indices[item]][1] - 10


class LettersUpperLocal270(LettersUpperBase):
    data = load_letters_upper_rotation_data(270)

    def __getitem__(self, item):
        return LettersUpperLocal270.data[self.indices[item]][0].view(1, -1), \
               LettersUpperLocal270.data[self.indices[item]][1] - 10


class LettersUpperDevBase(Dataset):
    data = None

    def __init__(self, indices):
        self.indices = indices
        self.size = len(indices)

    def __getitem__(self, item):
        return None, None

    def __len__(self):
        return self.size


class LettersUpperDev(LettersUpperDevBase):
    data = datasets.EMNIST(root=path_to_data,
                           split='byclass',
                           train=False,
                           download=True,
                           transform=transforms.ToTensor())

    def __getitem__(self, item):
        return LettersUpperDev.data[self.indices[item]][0].view(1, -1), LettersUpperDev.data[self.indices[item]][
            1] - 10


class LettersUpperDev90(LettersUpperDevBase):
    data = load_letters_upper_rotation_data_dev(90)

    def __getitem__(self, item):
        return LettersUpperDev90.data[self.indices[item]][0].view(1, -1), LettersUpperDev90.data[self.indices[item]][
            1] - 10


class LettersUpperDev180(LettersUpperDevBase):
    data = load_letters_upper_rotation_data_dev(180)

    def __getitem__(self, item):
        return LettersUpperDev180.data[self.indices[item]][0].view(1, -1), LettersUpperDev180.data[self.indices[item]][
            1] - 10


class LettersUpperDev270(LettersUpperDevBase):
    data = load_letters_upper_rotation_data_dev(270)

    def __getitem__(self, item):
        return LettersUpperDev270.data[self.indices[item]][0].view(1, -1), LettersUpperDev270.data[self.indices[item]][
            1] - 10


class LettersUpperSampler:
    def __init__(self):
        self.indices = np.load(path_to_data + '/letters_upper_indices.npy', allow_pickle=True)

    def sample(self, class_vec, num_samples):
        samples = list(0 for _ in range(num_samples))
        for i in range(num_samples):
            idx_class = np.random.randint(0, len(class_vec))
            idx_instance = np.random.randint(0, len(self.indices[class_vec[idx_class]]))
            samples[i] = self.indices[class_vec[idx_class]][idx_instance]
        return samples


class LettersUpperDevSampler:
    def __init__(self):
        self.indices = np.load(path_to_data + '/letters_upper_dev_indices.npy', allow_pickle=True)

    def sample(self, class_vec, num_samples):
        samples = list(0 for _ in range(num_samples))
        for i in range(num_samples):
            idx_class = np.random.randint(0, len(class_vec))
            idx_instance = np.random.randint(0, len(self.indices[class_vec[idx_class]]))
            samples[i] = self.indices[class_vec[idx_class]][idx_instance]
        return samples

    def full(self):
        full_indices = []
        for i in range(26):
            full_indices.extend(self.indices[i])
        return full_indices


def __create_letters_upper_indices():
    from os import path

    if path.exists(path_to_data + '/letters_upper_indices.npy'):
        return

    train_data = datasets.EMNIST(root=path_to_data,
                                 split='byclass',
                                 train=True,
                                 download=True,
                                 transform=transforms.ToTensor())
    indices = [[] for _ in range(26)]
    for i, (x, y) in enumerate(train_data):
        if 10 <= y < 36:
            indices[y - 10].append(i)

    np.save(path_to_data + '/letters_upper_indices.npy', indices)


def __create_letters_upper_dev_indices():
    from os import path

    if path.exists(path_to_data + '/letters_upper_dev_indices.npy'):
        return

    dev_data = datasets.EMNIST(root=path_to_data,
                               split='byclass',
                               train=False,
                               download=True,
                               transform=transforms.ToTensor())
    indices = [[] for _ in range(26)]
    for i, (x, y) in enumerate(dev_data):
        if 10 <= y < 36:
            indices[y - 10].append(i)

    np.save(path_to_data + '/letters_upper_dev_indices.npy', indices)


__create_letters_upper_indices()
__create_letters_upper_dev_indices()
