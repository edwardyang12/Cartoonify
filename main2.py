import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F

from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

from nets.utils import weights_init
from nets.discriminator import PatchGAN, MiniDiscriminator
from nets.generator import OrigGenerator as Generator
from custom_dataset1 import CustomDataset

lr = 0.0002
num_epochs = 3000
batch_size = 25
beta1 = 0.5
num_workers = 0
ngpu = 1
patch = 64 # patch size
size = 128

datapath = "./data/list_real.csv"

dataset = CustomDataset(datapath)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers)


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

netG = Generator().to(device)
# netG.apply(weights_init)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)

netD1 = PatchGAN().to(device)
netD2 = MiniDiscriminator().to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD1 = nn.DataParallel(netD1, list(range(ngpu)))
    netD2 = nn.DataParallel(netD2, list(range(ngpu)))
netD1.apply(weights_init)
netD2.apply(weights_init)

criterionMSE = nn.MSELoss()
criterionBCE = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD1 = optim.Adam(netD1.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD2 = optim.Adam(netD2.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

G_losses = []
DPatch_losses = []
DMini_losses = []

start = time.time()
fixed_noise = torch.randn(32, 3, 1, 1, device=device)

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader

    for i, data in enumerate(dataloader):
        
        data, path = data

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD1.zero_grad()
        netD2.zero_grad()

        real_cpu = data.to(device)
       
        b_size,channels,h,w = real_cpu.shape

        labelPatch = torch.full((b_size, 3,14,14), real_label, dtype=torch.float, device=device)
        labelMini = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        # Forward pass real batch through D
        outputPatch = netD1(real_cpu)
        # Calculate loss on all-real batch
        errD_real_Patch = criterionMSE(outputPatch, labelPatch)
        # Calculate gradients for D in backward pass
        errD_real_Patch.backward()

        top = np.random.randint(25,55)
        left = np.random.randint(25,40)
        cropped_real = F.crop(real_cpu, top, left, patch, patch)
        outputMini = netD2(cropped_real).view(-1)
        errD_real_Mini = criterionBCE(outputMini, labelMini)
        errD_real_Mini.backward()
        
        D_x = outputPatch.mean().item() + outputMini.mean().item()

        ## Train with all-fake batch

        # Generate fake image batch with G
        noise = torch.randn(b_size, 3, 1, 1, device=device)
        fake = netG(noise)
        labelPatch.fill_(fake_label)
        labelMini.fill_(fake_label)

        # Classify all fake batch with D
        outputPatch = netD1(fake.detach())
        # Calculate D's loss on the all-fake batch
        errD_fake_Patch = criterionMSE(outputPatch, labelPatch)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake_Patch.backward()

        top = np.random.randint(25,55)
        left = np.random.randint(25,40)
        cropped_fake = F.crop(fake.detach(), top, left, patch, patch)
        outputMini = netD2(cropped_fake).view(-1)
        errD_fake_Mini = criterionBCE(outputMini, labelMini)
        errD_fake_Mini.backward()
        
        D_G_z1 = outputPatch.mean().item() + outputMini.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real_Patch + errD_real_Mini + errD_fake_Patch + errD_fake_Mini

        # Update D
        optimizerD1.step()
        optimizerD2.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelPatch.fill_(real_label)
        labelMini.fill_(real_label)  # fake labels are real for generator cost
        
        # Since we just updated D, perform another forward pass of all-fake batch through D

        outputPatch = netD1(fake)

        top = np.random.randint(25,55)
        left = np.random.randint(25,40)
        cropped_fake = F.crop(fake, top, left, patch, patch)
        outputMini = netD2(cropped_fake).view(-1)
        # Calculate G's loss based on this output
        errGPatch = criterionMSE(outputPatch, labelPatch)
        errGMini = criterionBCE(outputMini, labelMini)

        # Calculate gradients for G
        errG = errGPatch + errGMini
        
        errGPatch.backward(retain_graph=True)
        errGMini.backward()
        
        D_G_z2 = outputPatch.mean().item() + outputMini.mean().item()
        # Update G
        optimizerG.step()


    # Output training stats
    if epoch % 25 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch, num_epochs, i, len(dataloader),
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        
        # Save Losses for plotting later
        G_losses.append(errG.item())
        DPatch_losses.append(errD_real_Patch.item() + errD_fake_Patch.item())
        DMini_losses.append(errD_real_Mini.item() + errD_fake_Mini.item())
        
        print(time.time()-start)
        start = time.time()

    if (epoch%100 ==0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu().numpy()

        img = ((fake[0]*0.5)+0.5)*255.
        img = img.transpose(1,2,0)
        img = Image.fromarray(img.astype(np.uint8),'RGB')
        img.save('fake'+str(epoch)+'_'+str(i)+'.png')
        
    if (epoch%200 ==0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        filename = 'carolGAN' + str(epoch) + '.pth'
        state = {'state_dict': netG.state_dict()}
        torch.save(state, filename)
        print('saved')

# loss graph
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(DPatch_losses,label="DPatch")
plt.plot(DMini_losses,label="DMini")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.ylim(0,20)
plt.savefig('lossgraph.png')
