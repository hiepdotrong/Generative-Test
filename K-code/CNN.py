import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import  torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)  
        
        self.fc1 = nn.Linear(576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 28x28 -> 26x26 -> 13x13
        x = self.pool(F.relu(self.conv2(x))) # 14x14 (padding = 1)-> 12x12 -> 6x6
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
epochs = 5

train_dataset = datasets.MNIST(root='data/', train= True, transform= transforms.ToTensor(), download= True)
train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='data/', train= False, transform= transforms.ToTensor(), download= True)
test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle=True)

model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)
step = 0

for epoch in range(epochs):
    for batch_idx, (data, labels) in enumerate(train_loader):

        data = data.to(device)
        labels = labels.to(device)

        scores = model(data)
        loss = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()



def check_acc(loader, model):
    if loader.dataset.train:
        print("Checking on training loss")
    else:
        print("Checking on test loss")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            running_train_acc = float(num_correct)/float(data.shape[0])

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples)*100: .2f}')

    model.train()

check_acc(train_loader, model)
check_acc(test_loader, model)

