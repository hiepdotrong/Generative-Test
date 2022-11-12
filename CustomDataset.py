import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class Custom(Dataset):
    def __init__(self, csv_file, file_data, transforms = None):
        self.csv = pd.read_csv(csv_file)
        self.file_data = file_data
        self.transform = transforms
        
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.file_data, self.csv.iloc[index, 0])
        read = Image.open(img_path)
                
        if self.transform:
            image = self.transform(read)
        return image

