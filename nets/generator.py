import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1, kernel_size=3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding =1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class ConvTranspose(nn.Module):
        def __init__(self, in_channels, out_channels,stride=2, kernel_size=4):
            super().__init__()
            self.double_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding =1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.double_conv(x)

class MiniUnet(nn.Module):
    def __init__(self):
        super(MiniUnet,self).__init__()
        self.input = ConvBlock(3,64)
        self.sInput = ConvBlock(64,64, stride=2)
        self.down_1 = ConvBlock(64,128)
        self.sDown_1 = ConvBlock(128,128, stride=2)
        self.down_2 = ConvBlock(128,256)

        self.up_sample_2 = ConvTranspose(256,128)
        self.up_2 = ConvBlock(256,128)

        self.up_sample_3 = ConvTranspose(128,64)
        self.up_3 = ConvBlock(128,64)

        self.temp_4 = nn.Conv2d(64,3, kernel_size=3, padding =1)


    def forward(self,x):
        x1 = self.input(x)
        x2 = self.sInput(x1)
        x3 = self.down_1(x2)
        x4 = self.sDown_1(x3)
        x5 = self.down_2(x4)

        x5 = self.up_sample_2(x5)

        x5 = torch.cat((x3,x5),dim=1)
        x5 = self.up_2(x5)
        x6 = self.up_sample_3(x5)

        x6 = torch.cat((x1,x6),dim=1)
        x6 = self.up_3(x6)
        x7 = nn.Tanh()(self.temp_4(x6))

        return x7

class OrigGenerator(nn.Module):
    def __init__(self, channels=3, ngf=64):
        super(OrigGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(channels, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf*16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
