import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

import datawrappers.mnist
from FLAG import path_to_data


class RotationTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


def load_mnist_rotation_data(angle):
    return datasets.MNIST(root=path_to_data,
                          train=True,
                          download=True,
                          transform=transforms.Compose(
                              [transforms.ToTensor(), RotationTransform(angle)]))


def load_mnist_rotation_data_dev(angle):
    return datasets.MNIST(root=path_to_data,
                          train=False,
                          download=True,
                          transform=transforms.Compose(
                              [transforms.ToTensor(), RotationTransform(angle)]))


class MnistRotationLocalBase(Dataset):
    data = None

    def __init__(self, indices):
        self.indices = indices
        self.size = len(self.indices)

    def __getitem__(self, item):
        return None, None

    def __len__(self):
        return self.size


class MnistRotationLocal90(MnistRotationLocalBase):
    data = load_mnist_rotation_data(90)

    def __getitem__(self, item):
        return MnistRotationLocal90.data[self.indices[item]][0].view(1, -1), torch.tensor(
            MnistRotationLocal90.data[self.indices[item]][1])


class MnistRotationLocal180(MnistRotationLocalBase):
    data = load_mnist_rotation_data(180)

    def __getitem__(self, item):
        return MnistRotationLocal180.data[self.indices[item]][0].view(1, -1), torch.tensor(
            MnistRotationLocal180.data[self.indices[item]][1])


class MnistRotationLocal270(MnistRotationLocalBase):
    data = load_mnist_rotation_data(270)

    def __getitem__(self, item):
        return MnistRotationLocal270.data[self.indices[item]][0].view(1, -1), torch.tensor(
            MnistRotationLocal270.data[self.indices[item]][1])


class MnistRotationDevBase(Dataset):
    data = None

    def __init__(self):
        self.size = 0

    def __getitem__(self, item):
        return None, None

    def __len__(self):
        return self.size


class MnistRotationDev90(MnistRotationDevBase):
    data = load_mnist_rotation_data_dev(90)
    
    def __init__(self):
        super(MnistRotationDev90, self).__init__()
        self.size = len(MnistRotationDev90.data)

    def __getitem__(self, item):
        return MnistRotationDev90.data[item][0].view(1, -1), MnistRotationDev90.data[item][1]


class MnistRotationDev180(MnistRotationDevBase):
    data = load_mnist_rotation_data_dev(180)
    
    def __init__(self):
        super(MnistRotationDev180, self).__init__()
        self.size = len(MnistRotationDev180.data)

    def __getitem__(self, item):
        return MnistRotationDev180.data[item][0].view(1, -1), MnistRotationDev180.data[item][1]


class MnistRotationDev270(MnistRotationDevBase):
    data = load_mnist_rotation_data_dev(270)
    
    def __init__(self):
        super(MnistRotationDev270, self).__init__()
        self.size = len(MnistRotationDev270.data)

    def __getitem__(self, item):
        return MnistRotationDev270.data[item][0].view(1, -1), MnistRotationDev270.data[item][1]


MnistRotationSampler = datawrappers.mnist.MnistSampler
datawrappers.mnist.__create_mnist_indices()
