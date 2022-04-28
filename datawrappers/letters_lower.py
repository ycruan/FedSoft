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


def load_letters_lower_rotation_data(angle):
    return datasets.EMNIST(root=path_to_data,
                           split='byclass',
                           train=True,
                           download=True,
                           transform=transforms.Compose(
                               [transforms.ToTensor(), RotationTransform(angle)]))


def load_letters_lower_rotation_data_dev(angle):
    return datasets.EMNIST(root=path_to_data,
                           split='byclass',
                           train=False,
                           download=True,
                           transform=transforms.Compose(
                               [transforms.ToTensor(), RotationTransform(angle)]))


class LettersLowerBase(Dataset):
    data = None

    def __init__(self, indices):
        self.indices = indices
        self.size = len(self.indices)

    def __getitem__(self, item):
        return None, None

    def __len__(self):
        return self.size


class LettersLowerLocal(LettersLowerBase):
    data = datasets.EMNIST(root=path_to_data,
                           split='byclass',
                           train=True,
                           download=True,
                           transform=transforms.ToTensor())

    def __getitem__(self, item):
        return LettersLowerLocal.data[self.indices[item]][0].view(1, -1), LettersLowerLocal.data[self.indices[item]][
            1] - 36


class LettersLowerLocal90(LettersLowerBase):
    data = load_letters_lower_rotation_data(90)

    def __getitem__(self, item):
        return LettersLowerLocal90.data[self.indices[item]][0].view(1, -1), \
               LettersLowerLocal90.data[self.indices[item]][1] - 36


class LettersLowerLocal180(LettersLowerBase):
    data = load_letters_lower_rotation_data(180)

    def __getitem__(self, item):
        return LettersLowerLocal180.data[self.indices[item]][0].view(1, -1), \
               LettersLowerLocal180.data[self.indices[item]][1] - 36


class LettersLowerLocal270(LettersLowerBase):
    data = load_letters_lower_rotation_data(270)

    def __getitem__(self, item):
        return LettersLowerLocal270.data[self.indices[item]][0].view(1, -1), \
               LettersLowerLocal270.data[self.indices[item]][1] - 36


class LettersLowerDevBase(Dataset):
    data = None

    def __init__(self, indices):
        self.indices = indices
        self.size = len(indices)

    def __getitem__(self, item):
        return None, None

    def __len__(self):
        return self.size


class LettersLowerDev(LettersLowerDevBase):
    data = datasets.EMNIST(root=path_to_data,
                           split='byclass',
                           train=False,
                           download=True,
                           transform=transforms.ToTensor())

    def __getitem__(self, item):
        return LettersLowerDev.data[self.indices[item]][0].view(1, -1), LettersLowerDev.data[self.indices[item]][1] - 36


class LettersLowerDev90(LettersLowerDevBase):
    data = load_letters_lower_rotation_data_dev(90)

    def __getitem__(self, item):
        return LettersLowerDev90.data[self.indices[item]][0].view(1, -1), \
               LettersLowerDev90.data[self.indices[item]][1] - 36


class LettersLowerDev180(LettersLowerDevBase):
    data = load_letters_lower_rotation_data_dev(180)

    def __getitem__(self, item):
        return LettersLowerDev180.data[self.indices[item]][0].view(1, -1), \
               LettersLowerDev180.data[self.indices[item]][1] - 36


class LettersLowerDev270(LettersLowerDevBase):
    data = load_letters_lower_rotation_data_dev(270)

    def __getitem__(self, item):
        return LettersLowerDev270.data[self.indices[item]][0].view(1, -1), \
               LettersLowerDev270.data[self.indices[item]][1] - 36


class LettersLowerSampler:
    def __init__(self):
        self.indices = np.load(path_to_data + '/letters_lower_indices.npy', allow_pickle=True)

    def sample(self, class_vec, num_samples):
        samples = list(0 for _ in range(num_samples))
        for i in range(num_samples):
            idx_class = np.random.randint(0, len(class_vec))
            idx_instance = np.random.randint(0, len(self.indices[class_vec[idx_class]]))
            samples[i] = self.indices[class_vec[idx_class]][idx_instance]
        return samples


class LettersLowerDevSampler:
    def __init__(self):
        self.indices = np.load(path_to_data + '/letters_lower_dev_indices.npy', allow_pickle=True)

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


def __create_letters_lower_indices():
    from os import path

    if path.exists(path_to_data + '/letters_lower_indices.npy'):
        return

    train_data = datasets.EMNIST(root=path_to_data,
                                 split='byclass',
                                 train=True,
                                 download=True,
                                 transform=transforms.ToTensor())
    indices = [[] for _ in range(26)]
    for i, (x, y) in enumerate(train_data):
        if 36 <= y < 62:
            indices[y - 36].append(i)

    np.save(path_to_data + '/letters_lower_indices.npy', indices)


def __create_letters_lower_dev_indices():
    from os import path

    if path.exists(path_to_data + '/letters_lower_dev_indices.npy'):
        return

    dev_data = datasets.EMNIST(root=path_to_data,
                               split='byclass',
                               train=False,
                               download=True,
                               transform=transforms.ToTensor())
    indices = [[] for _ in range(26)]
    for i, (x, y) in enumerate(dev_data):
        if 36 <= y < 62:
            indices[y - 36].append(i)

    np.save(path_to_data + '/letters_lower_dev_indices.npy', indices)


__create_letters_lower_indices()
__create_letters_lower_dev_indices()
