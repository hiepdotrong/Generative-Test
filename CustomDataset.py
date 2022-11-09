# Import
import os
import glob # lấy danh sách vào tên thư viện theo điều kiện
import random
import PIL.Image as Img
from torch.utils.data import Dataset
from torchvision import transforms

class Face_Data(Dataset): # Dataset template 
    def __init__(self, root, transform = transforms):
        self.transform = transform

        # Lấy danh sách tên file và thư mục theo điều kiện: glob.glob(pattern, *, recursive=False)
        self.real = sorted(glob.glob(root + "/*"))

    def __getitem__(self, index):
        item_real = self.transform(Img.open(self.real[index % len(self.real)]))

        return { "1" : item_real}

    def __len__(self):
        return len(self.real)
    
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
