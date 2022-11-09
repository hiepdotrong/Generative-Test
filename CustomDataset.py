import glob # đọc đuôi file 
import PIL.Image as Img
from torch.utils.data import Dataset
from torchvision import transforms
import torch
class Face_Data(Dataset): # Dataset template 
    def __init__(self, root, transform = transforms):
        self.transform = transform

        # Sắp sếp + đọc filename
        self.real = sorted(glob.glob(root + "/*"))
        self.fake = torch.tensor(3, 512, 512)
    def __getitem__(self, index):
        item_real = self.transform(Img.open(self.real[index % len(self.real)]))
        item_fake = self.fake
        return { "1" : item_real, "0" : item_fake}

    def __len__(self):
        return len(self.real)
    
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

