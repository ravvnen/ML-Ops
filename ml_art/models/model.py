import omegaconf

from torch import nn
import torch.nn as nn
import torch.nn.functional as F


class ArtCNN(nn.Module):
    def __init__(self,cfg: omegaconf.omegaconf.DictConfig):
        super(ArtCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=cfg.cnn.input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pooling layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((cfg.cnn.adaptive_pool[0], cfg.cnn.adaptive_pool[1]))

        # Fully connected layers
        self.fc1 = nn.Linear(cfg.cnn.fc1[0],cfg.cnn.fc1[1])
        self.fc2 = nn.Linear(cfg.cnn.fc2[0],cfg.cnn.fc2[1])
        self.fc3 = nn.Linear(cfg.cnn.fc2[1], len(cfg.dataset.styles))

    def forward(self, x):
        # Apply convolutional layers and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Apply adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
