import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


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

class ResnetGlobalAttention(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ResnetGlobalAttention, self).__init__()

        self.feature_channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv_end = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.soft = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        # 添加全局信息
        zx = y.squeeze(-1)
        zy = zx.permute(0, 2, 1)
        zg = torch.matmul(zy, zx)
        zg = zg / self.feature_channel

        batch = zg.shape[0]
        v = zg.squeeze(-1).permute(1, 0).expand((self.feature_channel, batch))
        v = v.unsqueeze_(-1).permute(1, 2, 0)

        # 全局局部信息融合
        atten = self.conv(y.squeeze(-1).transpose(-1, -2))
        atten = atten + v
        atten = self.conv_end(atten)
        atten = atten.permute(0, 2, 1).unsqueeze(-1)

        atten_score = self.soft(atten)

        return x * atten_score

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0

        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]


        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(DehazeBlock, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

#######################################主网络######################################################
class Base_Model(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, r=8,use_dropout=False, padding_type='reflect',
                 n_blocks=6):
        super(Base_Model, self).__init__()
        # down sampling
        self.down_pt1 = nn.Sequential(nn.ReflectionPad2d(3),
                                      nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                      nn.InstanceNorm2d(ngf),
                                      nn.ReLU(True))
        self.down_pt2 = nn.Sequential(nn.ReflectionPad2d(3),
                                      nn.Conv2d(ngf, input_nc, kernel_size=7, padding=0),
                                      nn.InstanceNorm2d(input_nc),
                                      nn.ReLU(True))
        self.down_resize = nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                                         nn.InstanceNorm2d(ngf),
                                         nn.ReLU(True))
        self.down_pt11 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                        nn.InstanceNorm2d(ngf * 2),
                                        nn.ReLU(True))

        self.down_pt21 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                       nn.InstanceNorm2d(ngf * 4),
                                       nn.ReLU(True))

        # block
        self.block1 = DehazeBlock(default_conv, ngf, 3)
        self.block2 = DehazeBlock(default_conv, ngf, 3)
        self.block3 = DehazeBlock(default_conv, ngf, 3)
        self.block4 = DehazeBlock(default_conv, ngf, 3)
        self.block5 = DehazeBlock(default_conv, ngf, 3)
        self.block6 = DehazeBlock(default_conv, ngf, 3)

        norm_layer = nn.InstanceNorm2d
        activation = nn.ReLU(True)
        model_res1 = []
        for i in range(n_blocks):
            model_res1 += [
                ResnetBlock(ngf * 4, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.model_res1 = nn.Sequential(*model_res1)

        # up sampling
        self.up1 = nn.Sequential(
                                nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                nn.InstanceNorm2d(ngf * 2),
                                nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.InstanceNorm2d(ngf),
                                 nn.ReLU(True))

        self.pa1 = PALayer(ngf*4)
        self.pa2 = PALayer(ngf*2)
        self.pa3 = PALayer(ngf)

        self.ca1 = ResnetGlobalAttention(ngf*4)
        self.ca2 = ResnetGlobalAttention(ngf*2)
        self.ca3 = ResnetGlobalAttention(ngf)



        self.conv = nn.Sequential( nn.Conv2d(ngf,ngf,kernel_size=3,stride=1,padding=1),
                                   nn.InstanceNorm2d(ngf),
                                   nn.ReLU(),
                                   nn.ReflectionPad2d(3),
                                   nn.Conv2d(ngf, output_nc,kernel_size=7,padding=0),
                                   nn.Tanh())

    def forward(self, input):

        #***************************************U-Net*************************************************
        x_down0 = self.down_resize(input)  # ngf
        x_down1 = self.down_pt11(x_down0)  # ngf*2
        x_down2 = self.down_pt21(x_down1)  # ngf*4
        # 最底层的残差块
        x2 = self.model_res1(x_down2)
        x2 = self.ca1(x2)
        x2 = self.pa1(x2)
        # 第二层的下采样，与对应解码层的进行融合
        x32 =x_down2 + x2
        # 融合完接着上采样
        x21 = self.up1(x32)
        x21 = self.ca2(x21)
        x21 = self.pa2(x21)
        # 第一层的下采样，与对应解码层进行融合
        x1 = x_down1+x21
        x10 = self.up2(x1)
        x10 = self.ca3(x10)
        x10 = self.pa3(x10)

        x_U = x_down0+x10
        #***************************************U-Net*************************************************


        #**************************************下分支直筒结构********************************************
        x_down11 = self.down_pt1(input)
        # 经过残差块
        # x_pt = self.model_res2(x_down11)
        x_pt = self.block1(x_down11)
        x_pt = self.block2(x_pt)
        x_pt = self.block3(x_pt)
        x_pt = self.block4(x_pt)
        x_pt = self.block5(x_pt)
        x_L = self.block6(x_pt)
        #********************************************************************************************


        #**************************************两个分支融合********************************************
        x_out = x_U+ x_L
        #********************************************************************************************
        out = self.conv(x_out)
        return out

class Discriminator(nn.Module):
    def __init__(self, bn=False, ngf=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf * 4) if not bn else nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf * 4) if not bn else nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, padding=0),
            nn.BatchNorm2d(ngf * 8) if bn else nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(ngf * 8) if bn else nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf * 16, 1, kernel_size=1)

        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


if __name__ == '__main__':
    x = torch.randn((4, 3, 64, 64),)
    #y = torch.ones(4, 3, 16,16)*0.5
    net = Base_Model(3, 3)
    print(net)
    out = net(x)
