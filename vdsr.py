import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F

def binarize(x, temperature=1.0, progressive=True, scale=False):
    replace = x.clamp(-1, 1)
    if scale:
        mean = abs(x).mean(-1, keepdim=True).mean(-2, keepdim=True).mean(-3, keepdim=True)
    else:
        mean = 1.0
    with torch.no_grad():
        binary = F.hardtanh(x/max(temperature, 1e-8))
        if not progressive:
            binary = binary.sign()
    return ((binary - replace).detach() + replace).mul(mean)

class BConv2d(nn.Module):
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 wb=True,
                 ab=True):
        super(BConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.wb = wb
        self.ab = ab
        self.register_buffer("temperature", torch.Tensor([1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.weight = nn.Parameter(torch.randn(out_channels, int(in_channels // groups), kernel_size, kernel_size))
        self.weight_init()
        if self.padding > 0:
            self.replicatepad = nn.ReplicationPad2d(padding=(padding, padding, padding, padding))
        else:
            self.replicatepad = lambda x: x
    def weight_init(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def update_temperature(self):
            self.temperature *= 0.965

    def forward(self, x):
        if self.wb:
            weight = binarize(self.weight, self.temperature, progressive=self.training, scale=False)
        x = x.view(x.size(0), -1, x.size(-2), x.size(-1))
        x = self.replicatepad(x)    # 对外围填充
        out = F.conv2d(input=x,
                       weight=weight,
                       bias=self.bias,
                       stride=self.stride,
                       padding=0,
                       dilation=self.dilation,
                       groups=self.groups)
        return out


# 神经网络结构块
class BConv_ReLU_Block(nn.Module):
    def __init__(self):
        super(BConv_ReLU_Block, self).__init__()
        self.conv = BConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


# 主要网络结构
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
        out = torch.add(out, residual)
        return out
