import torch.nn as nn
from utils import MiniBatchDiscrimination

# final layer is one hot prediction
class OrigDiscriminator(nn.Module):
    def __init__(self, channels=3, feat_map=64, batch=50):
        super(OrigDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (channels) x 512 x 512
            # input is actually (chennels) x 64 x 64 patches
            nn.Conv2d(channels, feat_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map, feat_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 2, feat_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 4, feat_map * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 8, feat_map * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 16),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (feat_map*8) x 4 x 4
            nn.Conv2d(feat_map * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() # no sigmoid when running cycle GAN
        )

    def forward(self, input):
        return self.main(input)

# final layer is patch prediction
class PatchGAN(nn.Module):
    def __init__(self, channels=3, feat_map=64, batch=50):
        super(PatchGAN, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(channels, feat_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map, feat_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map*2, feat_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (feat_map*8) x 4 x 4
            nn.Conv2d(feat_map * 4, feat_map*8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map * 8, 3, 4, 1, 1, bias=False),
        )

    def forward(self, input):
        return self.main(input)

# https://github.com/sanghviyashiitb/GANS-VanillaAndMinibatchDiscrimination/blob/master/minibatch_discrimination.py
# final layer is patch prediction
class PatchMiniBatch(nn.Module):
    def __init__(self, channels=3, feat_map=64, batch=50):
        super(PatchGAN, self).__init__()
        self.mbd1 = MiniBatchDiscrimination(feat_map*4, feat_map*4, feat_map, batch) # insert before last layer
        self.main = nn.Sequential(
            # input is actually (channels) x 64 x 64 patches
            nn.Conv2d(channels, feat_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map, feat_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_map*2, feat_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (feat_map*8) x 4 x 4
            nn.Conv2d(feat_map * 4, feat_map*4, 4, 1, 1, bias=False),
            nn.BatchNorm2d(feat_map * 4),
            nn.LeakyReLU(0.2, inplace=True),

        )
        self.out = nn.Conv2d(feat_map * 8, 3, 4, 1, 1, bias=False),

    def forward(self, input):
        x = self.main(input)
        return self.out(torch.cat((x,self.mbd1(x)),dim=1))
