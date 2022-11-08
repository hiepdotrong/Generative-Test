""" Training of ProGAN using WGAN-GP loss"""

from math import log2

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from Generator import Generator
from Discriminator import Discriminator 


