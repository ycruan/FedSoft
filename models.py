import torch
import torch.nn as nn
import math


class MnistMLP(nn.Module):
    def __init__(self):
        super(MnistMLP, self).__init__()
        self.in_size = 28 * 28
        self.hidden_size = 200
        self.out_size = 10
        self.net = nn.Sequential(
            nn.Linear(in_features=self.in_size, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=self.out_size),
            nn.Softmax(dim=2)
        )

        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

    def forward(self, batch):
        return torch.squeeze(self.net(batch))


class LettersCNN(nn.Module):
    def __init__(self):
        super(LettersCNN, self).__init__()
        self.kernel_conv = (5, 5)
        self.kernel_pool = (2, 2)
        self.channel1 = 32
        self.channel2 = 64
        self.conv_out_size = self.channel2*7*7
        self.fc_size = 512
        self.out_size = 26

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channel1, kernel_size=self.kernel_conv, padding=2),
            nn.MaxPool2d(kernel_size=self.kernel_pool),
            nn.Conv2d(in_channels=self.channel1, out_channels=self.channel2, kernel_size=self.kernel_conv, padding=2),
            nn.MaxPool2d(kernel_size=self.kernel_pool)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.conv_out_size, out_features=self.fc_size),
            nn.Linear(in_features=self.fc_size, out_features=self.out_size),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

        for m in self.modules():
            if type(m) == nn.Conv2d:
                nn.init.kaiming_uniform_(m.weight)
            elif type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

    def forward(self, batch):
        out1 = self.conv(batch.view(-1, 1, 28, 28)).view(-1, self.conv_out_size)
        return self.fc(out1)


class Cifar10CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

        for m in self.modules():
            if type(m) == nn.Conv2d:
                nn.init.kaiming_uniform_(m.weight)
            elif type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

    def forward(self, xb):
        return self.network(xb)


class LR(nn.Module):
    def __init__(self):
        from datawrappers.lr import LR_DIM
        super(LR, self).__init__()
        self.linear = nn.Linear(in_features=LR_DIM, out_features=1, bias=False)
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)

    def forward(self, batch):
        out = self.linear(batch)
        return out
