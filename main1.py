import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import itertools
from torch.autograd import Variable
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

from nets.utils import weights_init
from nets.discriminator import PatchGAN as Discriminator
from nets.generator import MiniUnet as Generator
from custom_dataset1 import CustomDataset

lr = 0.0002
num_epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
beta1 = 0.5
num_workers = int(sys.argv[3])
ngpu = int(sys.argv[4])
patch = 128 # patch size
size = 256

datapath = "./data/list_faces.csv"
simpath = "./data/list_cartoon.csv"
savedir = "/edward-slow-vol/cycleGAN/"

dataset = CustomDataset(datapath, simpath, size)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers)


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

netG = Generator().to(device)
# netG.apply(weights_init)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)

netD = Discriminator().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)

criterion = nn.MSELoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

G_losses = []
D_losses = []

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader

    start = time.time()
    for i, data in enumerate(dataloader):
        i = i*batch_size
        simdata, simpath, data, path = data
        b_size,channels,h,w = data.shape

        real_A = Variable(input_A.copy_(data)) # cartoon
        real_B = Variable(input_B.copy_(simdata)) # faces

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()

        b_size,channels,h,w = real_cpu.shape

        label = torch.full((b_size, 3,14,14), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D

        top = np.random.randint(0,patch)
        left = np.random.randint(0,patch)
        cropped_real_A = F.crop(real_A, top, left, patch, patch)
        output = netD(cropped_real_A)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch

        # Generate fake image batch with G
        fake = netG(real_B)
        label.fill_(fake_label)

        # Classify all fake batch with D
        top = np.random.randint(0,patch)
        left = np.random.randint(0,patch)
        cropped_fake = F.crop(fake.detach(), top, left, patch, patch)
        output = netD(cropped_fake)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D

        top = np.random.randint(0,patch)
        left = np.random.randint(0,patch)
        cropped_fake = F.crop(fake, top, left, patch, patch)
        output = netD(cropped_fake)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()


    # Output training stats
    if i % 1000 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(dataloader),
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        print(time.time()-start)
        start = time.time()

    if i % 1000 == 0 or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        with torch.no_grad():
            fake = netG(real_B).detach().cpu().numpy()

        img = ((fake[0]*0.5)+0.5)*255.
        img = img.transpose(1,2,0)
        img = Image.fromarray(img.astype(np.uint8),'RGB')
        img.save(savedir+'fake'+str(epoch)+'_'+str(i)+'.png')
        print(simpath[0], path[0])

    # Save Losses for plotting later
    G_losses.append(errG.item())
    D_losses.append(errD.item())

# loss graph
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(G_GAN_losses,label="G_GAN")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.ylim(0,20)
plt.savefig(savedir + 'lossgraph.png')
