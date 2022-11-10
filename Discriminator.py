from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.Dis = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(6, 16, 4),
            nn.ReLU(),
            nn.MaxPool2d(3,3),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2,2), 
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #64x12x12
            nn.Flatten(0,2),
            nn.Linear(64*12*12,128),
            nn.ReLU(),
            nn.Linear(128,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.Dis(x)
    