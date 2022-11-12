import os
from torch.utils.data import Dataset
from skimage import io
import pandas as pd
from PIL import Image

class Custom(Dataset):
    def __init__(self, csv_file, root, transforms = None):
        self.csv = pd.read_csv(csv_file)
        self.root = root
        self.transform = transforms
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.csv.iloc[index, 0])
        read = Image.open(img_path)
                
        if self.transform:
            image = self.transform(read)
            return image

