import torch
from torch import nn


class ResnetBlock(nn.Module):

    def __init__(self, dim, levels=0, bn=False,down=True):
        super(ResnetBlock, self).__init__()
        blocks = []
        for i in range(levels):
            blocks.append(Block(dim=dim, bn=bn))
        self.res = nn.Sequential(
            *blocks
        )

        self.downsample_layer = nn.Sequential(
            # nn.InstanceNorm2d(dim, eps=1e-6) if not bn else nn.BatchNorm2d(dim, eps=1e-6),
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2),
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim * 2) if not bn else nn.BatchNorm2d(dim * 2, eps=1e-6),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1) ,
            nn.InstanceNorm2d(dim * 2) if not bn else nn.BatchNorm2d(dim * 2, eps=1e-6),
            nn.LeakyReLU(0.2)
     
        ) if down else None


    def forward(self, x):
        out = x + self.res(x)
        if self.downsample_layer is not None:
            out = self.downsample_layer(out)
        return out





class Block(nn.Module):

    def __init__(self, dim, bn=False):
        super(Block, self).__init__()

        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                       nn.InstanceNorm2d(dim) if not bn else nn.BatchNorm2d(dim, eps=1e-6),
                       nn.LeakyReLU(0.2)
                    
                      ]

        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                       nn.InstanceNorm2d(dim) if not bn else nn.BatchNorm2d(dim, eps=1e-6)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class PALayer(nn.Module):

    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y




class CALayer(nn.Module):

    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


def padding_image(image, h, w):
    assert h >= image.size(2)
    assert w >= image.size(3)
    padding_top = (h - image.size(2)) // 2
    padding_down = h - image.size(2) - padding_top
    padding_left = (w - image.size(3)) // 2
    padding_right = w - image.size(3) - padding_left
    out = torch.nn.functional.pad(image, (padding_left, padding_right, padding_top, padding_down), mode='reflect')
    return out, padding_left, padding_left + image.size(3), padding_top, padding_top + image.size(2)

