
import torchvision.transforms.functional as F
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

torch.manual_seed (50)
network1 = nn.Sequential(
      nn.Conv2d (in_channels = 1, out_channels = 6, kernel_size = 5)
    , nn.ReLU ()
    , nn.MaxPool2d (kernel_size = 2, stride = 2)
    , nn.Conv2d (in_channels = 6, out_channels = 12, kernel_size = 5)
    , nn.ReLU ()
    , nn.MaxPool2d (kernel_size = 2, stride = 2)
    , nn.Flatten (start_dim = 1)
    , nn.Linear (in_features = 12 * 4 * 4, out_features = 120)
    , nn.ReLU ()
    , nn.Linear (in_features = 120, out_features = 60)
    , nn.ReLU ()
    , nn.Linear (in_features = 60, out_features = 10)
)

torch.manual_seed (50)
network2 = nn.Sequential (
      nn.Conv2d (in_channels = 1, out_channels = 6, kernel_size = 5)
    , nn.ReLU ()
    , nn.MaxPool2d (kernel_size = 2, stride = 2)
    , nn.BatchNorm2d (6)
    , nn.Conv2d (in_channels = 6, out_channels = 12, kernel_size = 5)
    , nn.ReLU ()
    , nn.MaxPool2d (kernel_size = 2, stride = 2)
    , nn.Flatten (start_dim = 1)
    , nn.Linear (in_features = 12 * 4 * 4, out_features = 120)
    , nn.ReLU ()
    , nn.BatchNorm1d (120)
    , nn.Linear (in_features = 120, out_features = 60)
    , nn.ReLU ()
    , nn.Linear (in_features = 60, out_features = 10)
)

networks = {
    'no_batch_norm': network1
    ,'batch_norm': network2
}

lr = [.01]
batch_size = [1000]
num_workers = [1]
device = ['cpu']
trainset = ['normal']
network = list(networks.keys())





