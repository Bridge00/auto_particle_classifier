#https://stackoverflow.com/questions/65154182/implement-separableconv2d-in-pytorch
import torch
import torch.nn as nn
from torch.nn import init

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding = padding, stride = stride,
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    

class SeparableConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, bias=False):
        super(SeparableConvTranspose2d, self).__init__()
        self.depthwise = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, padding=padding, stride = stride)
        self.pointwise = nn.ConvTranspose2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out