import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, in_channels, out_channels, nfilters):
        super(Network, self).__init__()

        self.nfilters = nfilters

        self.features = nn.Sequential(
            # 62, nfilters
            nn.Conv1d(in_channels, nfilters, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm1d(num_features=nfilters),
            nn.LeakyReLU(0.2),
            # 32, nfitlers * 2
            nn.Conv1d(nfilters, nfilters * 2, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm1d(num_features=nfilters * 2),
            nn.LeakyReLU(0.2),
            # 17, nfilters * 4
            nn.Conv1d(nfilters * 2, nfilters * 4, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm1d(num_features=nfilters * 4),
            nn.LeakyReLU(0.2),
            # 10, nfilters * 8
            nn.Conv1d(nfilters * 4, nfilters * 8, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm1d(num_features=nfilters * 8),
            nn.LeakyReLU(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(10 * nfilters * 8, 2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, out_channels),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 10 * self.nfilters * 8)
        x = self.classifier(x)
        x = torch.tanh(x)
        return x


class FCNetwork(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FCNetwork, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 1200),
            nn.LeakyReLU(0.02),
            nn.Linear(1200, 1000),
            nn.LeakyReLU(0.02),
            nn.Linear(1000, 800),
            nn.LeakyReLU(0.02),
            nn.Linear(800, 600),
            nn.LeakyReLU(0.02),
            nn.Linear(600, 400),
            nn.LeakyReLU(0.02),
            nn.Linear(400, 150),
            nn.LeakyReLU(0.02),
            nn.Linear(150, 60),
            nn.LeakyReLU(0.02),
            nn.Linear(60, out_channels),
        )

    def forward(self, x):
        x = self.classifier(x)
        x = torch.tanh(x)
        return x
