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

from nets.utils import ReplayBuffer, weights_init
from nets.discriminator import PatchGAN as Discriminator
# from nets.generator import MiniUnet as Generator
from nets.newgenerator import ResnetGenerator as Generator
from custom_dataset import CustomDataset

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

netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG_A2B = nn.DataParallel(netG_A2B, list(range(ngpu)))
    netG_B2A = nn.DataParallel(netG_B2A, list(range(ngpu)))

netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD_A = nn.DataParallel(netD_A, list(range(ngpu)))
    netD_B = nn.DataParallel(netD_B, list(range(ngpu)))

netD_A.apply(weights_init)
netD_B.apply(weights_init)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))


# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(batch_size, 3, size, size)
input_B = Tensor(batch_size, 3, size, size)
target_real = torch.full((batch_size,3,14,14), real_label, dtype=torch.float, device=device) # 30 x 30 because crop size was 256 with PatchGAN
target_fake = torch.full((batch_size,3,14,14), fake_label, dtype=torch.float, device=device) # should be 14 x 14 for patch 128

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
G_losses = []
G_identity_losses = []
G_GAN_losses = []
G_cycle_losses = []
D_losses = []
iters = 0

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

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0

        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        top = np.random.randint(0,patch)
        left = np.random.randint(0,patch)
        cropped_fake_B = F.crop(fake_B, top, left, patch, patch)
        pred_fake = netD_B(cropped_fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        top = np.random.randint(0,patch)
        left = np.random.randint(0,patch)
        cropped_fake_A = F.crop(fake_A, top, left, patch, patch)
        pred_fake = netD_A(cropped_fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        top = np.random.randint(0,patch)
        left = np.random.randint(0,patch)
        cropped_real_A = F.crop(real_A, top, left, patch, patch)
        pred_real = netD_A(cropped_real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        top = np.random.randint(0,patch)
        left = np.random.randint(0,patch)
        cropped_fake_A = F.crop(fake_A.detach(), top, left, patch, patch)
        cropped_fake_A = fake_A_buffer.push_and_pop(cropped_fake_A)
        pred_fake = netD_A(cropped_fake_A)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        top = np.random.randint(0,patch)
        left = np.random.randint(0,patch)
        cropped_real_B = F.crop(real_B, top, left, patch, patch)
        pred_real = netD_B(cropped_real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        top = np.random.randint(0,patch)
        left = np.random.randint(0,patch)
        cropped_fake_B = F.crop(fake_B.detach(), top, left, patch, patch)
        cropped_fake_B = fake_B_buffer.push_and_pop(cropped_fake_B)
        pred_fake = netD_B(cropped_fake_B)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()


        if i % 1000 == 0:
            print("====== ", i, len(dataloader), epoch)
            print('loss_G: '+ str(loss_G.item()) + ' loss_G_identity: ' + str((loss_identity_A + loss_identity_B).item()) +  ' loss_G_GAN: ' + str((loss_GAN_A2B + loss_GAN_B2A).item()) + ' loss_G_cycle: ' +  str((loss_cycle_ABA + loss_cycle_BAB).item()))
            print('loss_D: ' + str((loss_D_A + loss_D_B).item()))
            print(time.time()-start)
            start = time.time()

        if i % 1000 == 0 or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_A = netG_B2A(real_B).detach().cpu().numpy()
                fake_B = netG_A2B(real_A).detach().cpu().numpy()

            img = ((real_A[0]*0.5)+0.5)*255.
            img = img.detach().cpu().numpy()
            img = img.transpose(1,2,0)
            img = Image.fromarray(img.astype(np.uint8),'RGB')
            img.save(savedir + 'real'+str(epoch)+'_'+str(i)+'.png')

            img = ((real_B[0]*0.5)+0.5)*255.
            img = img.detach().cpu().numpy()
            img = img.transpose(1,2,0)
            img = Image.fromarray(img.astype(np.uint8),'RGB')
            img.save(savedir + 'sim'+str(epoch)+'_'+str(i)+'.png')

            img = ((fake_A[0]*0.5)+0.5)*255.
            img = img.transpose(1,2,0)
            img = Image.fromarray(img.astype(np.uint8),'RGB')
            img.save(savedir + 'fakeReal'+str(epoch)+'_'+str(i)+'.png')

            img = ((fake_B[0]*0.5)+0.5)*255.
            img = img.transpose(1,2,0)
            img = Image.fromarray(img.astype(np.uint8),'RGB')
            img.save(savedir + 'fakeSim'+str(epoch)+'_'+str(i)+'.png')

            print(simpath[0], path[0])

        G_losses.append(loss_G.item())
        D_losses.append((loss_D_A + loss_D_B).item())
        G_GAN_losses.append((loss_GAN_A2B + loss_GAN_B2A).item())


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
