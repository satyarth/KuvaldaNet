import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Conv2D1(nn.Module):
    def __init__(self, in_slices=24, in_channels=1, out_channels=1):
        super(Conv2D1, self).__init__()
        self.in_slices = in_slices
        self.in_channels = in_channels
        
        self.conv_set = [nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=(4,4), 
                             stride=2, padding=1).cuda() for _ in range(self.in_slices)]
                
    def forward(self, x):
        x_slice = x[:,:,0,...][:,None,...].squeeze(1)
        
        
        downsampled = self.conv_set[0](x_slice).unsqueeze(2)
        
        for i in range(1,self.in_slices):
            x_slice = x[:,:,i,...][:,None,...].squeeze(1)
            downsampled_curr = self.conv_set[i](x_slice).unsqueeze(2)
            downsampled = torch.cat([downsampled, downsampled_curr],2)
            
        return downsampled
    
class ConvTranspose2D1(nn.Module):
    def __init__(self, in_slices=24, in_channels=1, out_channels=1):
        super(ConvTranspose2D1, self).__init__()
        self.in_chanels = in_slices
        
        convs2d = [nn.ConvTranspose2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=(4,4), 
                             stride=2, padding=1).cuda() for _ in range(self.in_chanels)]
        
        self.conv_set = convs2d
        
    def forward(self, x):
        
        x_slice = x[:,:,0,...][:,None,...].squeeze(1)
        downsampled = self.conv_set[0](x_slice).unsqueeze(2)
        
        for i in range(1,self.in_chanels):
            x_slice = x[:,:,i,...][:,None,...].squeeze(1)
            downsampled_curr = self.conv_set[i](x_slice).unsqueeze(2)
            downsampled = torch.cat([downsampled, downsampled_curr],2)
        return downsampled