import torch
import torch.nn as nn 
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Model Generator
# input: tensor1, tensor2
# shape of input tensor = 3x512x512
# output: tensor3
# shape of output tensor =
# Model Generator
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image

img_to_tensor = transforms.ToTensor()
tensor_to_img = transforms.ToPILImage()
center_crop = transforms.CenterCrop(512)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 3), # 3x512x512 -> 6x510x510
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 6x510x510 -> 6x255x255
            nn.Conv2d(6, 16, 3), # 6x255x255 -> 16x253x253
            nn.ReLU(),
            nn.MaxPool2d(2,2,padding=1), # 16x253x253 -> 16x127x127
            nn.Conv2d(16, 32, 3), # 16x127x127 -> 32x125x125
            nn.ReLU(),
            nn.MaxPool2d(2,2,padding=1), # 32x125x125 -> 32x63x63    
            nn.Conv2d(32, 64, 3, padding=1), # 32x63x63 -> 64x61x61 
            nn.ReLU(),
            nn.MaxPool2d(2,2,padding=1), # 64x32x32-> 64x32x32
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2),  
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride=2), 
            nn.ReLU(),
            nn.ConvTranspose2d(8,3, 6, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x, y):
        tensor1 = self.decoder((self.encoder(x) + self.encoder(y))/2)
        img1 = tensor_to_img(tensor1)
        img2 = center_crop(img1)
        tensor2 = img_to_tensor(img2)
        return tensor2