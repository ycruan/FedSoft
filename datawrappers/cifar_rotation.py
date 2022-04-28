import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.autograd import Variable
import torch.nn.functional as F

import datawrappers.cifar
from FLAG import path_to_data


class RotationTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


def load_cifar_rotation_data(angle):
    return datasets.CIFAR10(root=path_to_data,
                            train=True,
                            download=True,
                            transform=transforms.Compose(
                                [transforms.ToTensor(),
                                 transforms.Lambda(lambda x: F.pad(
                                     Variable(x.unsqueeze(0), requires_grad=False),
                                     (4, 4, 4, 4), mode='reflect').data.squeeze()),
                                 transforms.ToPILImage(),
                                 transforms.RandomCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 RotationTransform(angle),
                                 transforms.ToTensor()]))


def load_cifar_rotation_data_dev(angle):
    return datasets.CIFAR10(root=path_to_data,
                            train=False,
                            download=True,
                            transform=transforms.Compose(
                                [RotationTransform(angle), transforms.ToTensor()]))


class CifarRotationLocalBase(Dataset):
    data = None

    def __init__(self, indices):
        self.indices = indices
        self.size = len(self.indices)

    def __getitem__(self, item):
        return None, None

    def __len__(self):
        return self.size


class CifarRotationLocal90(CifarRotationLocalBase):
    data = load_cifar_rotation_data(90)

    def __getitem__(self, item):
        return CifarRotationLocal90.data[self.indices[item]][0], CifarRotationLocal90.data[self.indices[item]][1]


class CifarRotationDevBase(Dataset):
    data = None

    def __init__(self):
        self.size = 0

    def __getitem__(self, item):
        return None, None

    def __len__(self):
        return self.size


class CifarRotationDev90(CifarRotationDevBase):
    data = load_cifar_rotation_data_dev(90)

    def __init__(self):
        super(CifarRotationDev90, self).__init__()
        self.size = len(CifarRotationDev90.data)

    def __getitem__(self, item):
        return CifarRotationDev90.data[item][0], CifarRotationDev90.data[item][1]


CifarRotationSampler = datawrappers.cifar.CifarSampler
datawrappers.cifar.__create_cifar_indices()
