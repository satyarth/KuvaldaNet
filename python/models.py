import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class downstream_block(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_dim=3, padding=1, dropout=False):
        super(downstream_block, self).__init__()
        
        block = [
            nn.Conv3d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_dim, padding=padding),
            nn.BatchNorm3d(out_chan),
            nn.LeakyReLU()
        ]
        
        if dropout:
            block += [nn.Dropout3d()]
        
        block += [
            nn.Conv3d(in_channels=out_chan, out_channels=out_chan, kernel_size=kernel_dim, padding=padding),
            nn.BatchNorm3d(out_chan),
            nn.LeakyReLU(),
            nn.Conv3d(out_chan, out_chan, (1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        ]
        
        self.block = nn.Sequential(*block)
    
    def forward(self, x):
        return self.block(x)
        
class upstream_block(nn.Module):
    def __init__(self, dim_x, dim_y, out_dim, kernel_dim=3, padding=1, dropout=False):
        super(upstream_block, self).__init__()
        
        self.up = nn.ConvTranspose3d(in_channels=dim_x, out_channels=dim_x, kernel_size=(1, 4, 4), padding=(0, 1, 1), stride=(1, 2, 2))
        
        block = [
            nn.Conv3d(in_channels=dim_x+dim_y, out_channels=out_dim, kernel_size=kernel_dim, padding=padding),
            nn.BatchNorm3d(out_dim),
            nn.LeakyReLU(),
        ]
        
        if dropout:
            block += [nn.Dropout3d()]

        
        block += [
            nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_dim, padding=padding),
            nn.BatchNorm3d(out_dim),
            nn.LeakyReLU(),
        ]
        
        self.block = nn.Sequential(*block)
    
    def forward(self, x, y):
        x = self.up(x)
        return self.block(torch.cat([x,y],1))
        
class UNet3D(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=48, dropout=False):
        super(UNet3D, self).__init__()
        
        self.inconv = nn.Sequential(
            nn.Conv3d(in_channels=input_nc, out_channels=ngf, kernel_size=3, padding=1),
            nn.BatchNorm3d(ngf),
            nn.LeakyReLU(),
            
            nn.Conv3d(in_channels=ngf, out_channels=ngf, kernel_size=3, padding=1),
            nn.BatchNorm3d(ngf),
            nn.LeakyReLU()
        )
        
        self.down1 = downstream_block(in_chan=ngf, out_chan=ngf*2, dropout=dropout)
        self.down2 = downstream_block(in_chan=ngf*2, out_chan=ngf*4, dropout=dropout)
        self.down3 = downstream_block(in_chan=ngf*4, out_chan=ngf*8, dropout=dropout)
        self.down4 = downstream_block(in_chan=ngf*8, out_chan=ngf*16, dropout=dropout)
        
        self.up1 = upstream_block(ngf*16, ngf*8, ngf*8, dropout=dropout)
        self.up2 = upstream_block(ngf*8, ngf*4, ngf*4, dropout=dropout)
        self.up3 = upstream_block(ngf*4, ngf*2, ngf*2, dropout=dropout)
        self.up4 = upstream_block(ngf*2, ngf, ngf, dropout=dropout)
        
        self.outconv = nn.Sequential(
            nn.Conv3d(in_channels=ngf, out_channels=ngf, kernel_size=3, padding=1),
            nn.BatchNorm3d(ngf),
            nn.LeakyReLU(),
            
            nn.Conv3d(in_channels=ngf, out_channels=ngf, kernel_size=3, padding=1),
            nn.BatchNorm3d(ngf),
            nn.LeakyReLU(),
            
            nn.Conv3d(in_channels=ngf, out_channels=output_nc, kernel_size=3, padding=1),
            #nn.Sigmoid()
        )
        
    def forward(self, x):
            
        x_in = self.inconv(x)
#         print(x_in.shape)
        x1 = self.down1(x_in)
#         print(x1.shape)
        
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
#         print(x4.shape)
#         print(x3.shape)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x_in)
        
        return self.outconv(x)
    
class downstream_block_2d(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_dim=3, padding=1, dropout=False):
        super(downstream_block_2d, self).__init__()
        
        block = [
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=kernel_dim, padding=padding),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU()
        ]
        
        if dropout:
            block += [nn.Dropout2d()]
        
        block += [
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=kernel_dim, padding=padding),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=4, stride=2, padding=1)
        ]
        
        self.block = nn.Sequential(*block)
    
    def forward(self, x):
        return self.block(x)
    

class upstream_block_2d(nn.Module):
    def __init__(self, dim_x, dim_y, out_dim, kernel_dim=3, padding=1, dropout=False):
        super(upstream_block_2d, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels=dim_x, out_channels=dim_x, kernel_size=4, stride=2, padding=1)
        
        block = [
            nn.Conv2d(in_channels=dim_x+dim_y, out_channels=out_dim, kernel_size=kernel_dim, padding=padding),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(),
        ]
        
        if dropout:
            block += [nn.Dropout2d()]

        
        block += [
            nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=kernel_dim, padding=padding),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(),
        ]
        
        self.block = nn.Sequential(*block)
    
    def forward(self, x, y):
        x = self.up(x)
        return self.block(torch.cat([x,y],1))
        
class UNet2D(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=48, dropout=False):
        super(UNet2D, self).__init__()
        
        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels=input_nc, out_channels=ngf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU()
        )
        
        self.down1 = downstream_block_2d(in_chan=ngf, out_chan=ngf*2, dropout=dropout)
        self.down2 = downstream_block_2d(in_chan=ngf*2, out_chan=ngf*4, dropout=dropout)
        self.down3 = downstream_block_2d(in_chan=ngf*4, out_chan=ngf*8, dropout=dropout)
        self.down4 = downstream_block_2d(in_chan=ngf*8, out_chan=ngf*16, dropout=dropout)
        
        self.up1 = upstream_block_2d(ngf*16, ngf*8, ngf*8, dropout=dropout)
        self.up2 = upstream_block_2d(ngf*8, ngf*4, ngf*4, dropout=dropout)
        self.up3 = upstream_block_2d(ngf*4, ngf*2, ngf*2, dropout=dropout)
        self.up4 = upstream_block_2d(ngf*2, ngf, ngf, dropout=dropout)
        
        self.outconv = nn.Sequential(
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=ngf, out_channels=output_nc, kernel_size=3, padding=1),
            #nn.Sigmoid()
        )
        
    def forward(self, x):
            
        x_in = self.inconv(x)
        #print(x_in.shape)
        x1 = self.down1(x_in)
        
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x_in)
        
        return self.outconv(x)
    
    
class UNet2DAuto(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=48, dropout=False):
        super(UNet2DAuto, self).__init__()
        
        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels=input_nc, out_channels=ngf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU()
        )
        
        self.down1 = downstream_block_2d(in_chan=ngf, out_chan=ngf*2, dropout=dropout)
        self.down2 = downstream_block_2d(in_chan=ngf*2, out_chan=ngf*4, dropout=dropout)
        self.down3 = downstream_block_2d(in_chan=ngf*4, out_chan=ngf*8, dropout=dropout)
        self.down4 = downstream_block_2d(in_chan=ngf*8, out_chan=ngf*16, dropout=dropout)
        
        self.up1 = upstream_block_2d(ngf*16, ngf*8, ngf*8, dropout=dropout)
        self.up2 = upstream_block_2d(ngf*8, ngf*4, ngf*4, dropout=dropout)
        self.up3 = upstream_block_2d(ngf*4, ngf*2, ngf*2, dropout=dropout)
        self.up4 = upstream_block_2d(ngf*2, ngf, ngf, dropout=dropout)
        
        self.outconv = nn.Sequential(
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(),
            
            nn.Conv2d(in_channels=ngf, out_channels=input_nc, kernel_size=3, padding=1),
            #nn.Sigmoid()
        )
        
    def forward(self, x):
            
        x_in = self.inconv(x)
        #print(x_in.shape)
        x1 = self.down1(x_in)
        
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, torch.zeros_like(x3))
        x = self.up2(x, torch.zeros_like(x2))
        x = self.up3(x, torch.zeros_like(x1))
        x = self.up4(x, torch.zeros_like(x_in))
        
        return self.outconv(x)