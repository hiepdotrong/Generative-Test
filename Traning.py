import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DatasetLoader
import matplotlib.pyplot as plt
import Sum
# parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data-processing
dir = "/Data"
data_loader = DataLoader(yesno_data,
                                          batch_size=1,
                                          shuffle=True)
dataset = DatasetLoader('D:\images\Bright_240_270', get_transform(train=True))
# loss funtion

# tranning
sum()