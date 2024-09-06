import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, 4)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(10, 10, 3)
        self.pool2 = nn.MaxPool2d(3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(40, 20)
        self.linear2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x