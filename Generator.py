# Model Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, 3), # 630x630x3 -> 628x628x6
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 628x628x6 -> 314x314x6
            nn.Conv2d(6, 16, 3), # 314x314x6 -> 312x312x16
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 312x312x16 -> 156x156x16
            nn.Conv2d(16, 32, 3), # 156x156x16-> 154x154x32
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 154x154x32-> 77x77x32   
            nn.Conv2d(32, 64, 3, padding=1), # 77x77x32-> 76x76x64
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 76x76x64-> 38x38x64
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 5, stride=3, padding=0),  # b, 8, 15, 15
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=3, padding=0),  # b, 8, 15, 15
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 8, 7, stride=3, padding=0,output_padding=1),  # b, 1, 28, 28
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(8,3, 7, stride=1, padding=0),  # b, 1, 28, 28
        )
    def forward(self, x, y):
        return self.decoder((self.encoder(x) + self.encoder(y))/2) 