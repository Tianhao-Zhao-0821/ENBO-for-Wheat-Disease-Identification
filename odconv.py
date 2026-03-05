import torch.nn as nn



class ODConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True):
        super(ODConv2d, self).__init__()

        # 直接使用普通卷积，但保持接口一致
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        # 为了保持参数名称一致
        self.weight = self.conv.weight
        if bias:
            self.bias = self.conv.bias
        else:
            self.bias = None

    def forward(self, x):
        return self.conv(x)