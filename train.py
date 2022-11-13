import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
from Generator import Generator
from Discriminator import Discriminator 
from CustomDataset import Custom
import config

from torch.utils.tensorboard import SummaryWriter


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5, 0.5)
])


real = Custom(csv_file= config.CSV, root= config.DATASET, transforms= transform)
dataloader = torch.utils.data.DataLoader(real, batch_size=3, shuffle=True, pin_memory=True)
# initialize gen and disc/critic
gen = Generator().to(config.DEVICE)
dis = Discriminator().to(config.DEVICE)

# initializate optimizer
opt_gen = optim.RMSprop(gen.parameters(), lr=config.LEARNING_RATE)
opt_dis = optim.RMSprop(dis.parameters(), lr=config.LEARNING_RATE)

# for tensorboard plotting
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

for epoch in range(config.NUM_EPOCHS): # for each epoch
    # Target labels not needed! <3 unsupervised
    for batch_idx, data in enumerate(dataloader):  # for each batch
        data = data.to(config.DEVICE)
        data1 = data[0]
        data2 = data[1]
        real = data[2]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(config.CRITIC_ITERATIONS):
            fake = gen(data1,data2)
            dis_real = dis(data3)
            dis_fake = dis(fake)
            loss_dis = -(dis_real - dis_fake)
            dis.zero_grad()
            loss_dis.backward(retain_graph=True)
            opt_dis.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = dis(fake)
        loss_gen = -gen_fake
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{config.NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                  Loss D: {loss_dis:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(data1,data2)
                img_grid_real = torchvision.utils.make_grid(
                    data1, data2 , normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake, normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
            gen.train()
    dataloader = torch.utils.data.DataLoader(real, batch_size=3, shuffle=True, pin_memory=True)
