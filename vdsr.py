import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F

def binarize(x, scale = True):
    replace = x.clamp(-1, 1)
    mean = replace.abs().mean(-1, keepdim = True).mean(-2, keepdim = True).mean(-3, keepdim = True)

    return ((torch.sign(x) - replace).detach() + replace).mul(mean)


class BConv(nn.Module):
    def __init__(self, 
            in_channels = 64,
            out_channels = 64,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            dilation = 1,
            groups = 1,
            bias = False
            ):
        super(BConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        assert in_channels % self.groups == 0, "The number of input channels can not be splited uniformly by number {}".format(self.groups)

        self.weight = nn.Parameter(torch.randn(out_channels, int(in_channels//groups), kernel_size, kernel_size)*1e-2)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1)) if bias else None
        
        self.weight_init()

    def weight_init(self):
        nn.init.kaiming_normal_(self.weight, mode = 'fan_out', nonlinearity = 'relu')
        
    def forward(self, x):
        weight = binarize(self.weight)

        return F.conv2d(
                    x,
                    weight,
                    bias = self.bias,
                    padding = self.padding,
                    dilation = self.dilation,
                    groups = self.groups
                )

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class BConv_ReLU_Block(nn.Module):
    def __init__(self):
        super(BConv_ReLU_Block, self).__init__()
        self.conv = BConv(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding =1, bias = False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_layer = self.make_layer(BConv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out
 
