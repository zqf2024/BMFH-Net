import torch
import torch.nn as nn
from Model_util import PALayer, ResnetBlock, CALayer
from einops import rearrange
from ckt import CKTModule


# 233 26.59
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width], device=x.device).float()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, ll, hl, lh, hh):
        x = torch.cat((ll, hl, lh, hh), 1)
        return iwt_init(x)



class Channel_Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super(Channel_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(TransformerBlock, self).__init__()

        self.norm = nn.BatchNorm2d(dim)
        hidden = dim // 2
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn = Channel_Attention(dim, num_heads)
       # self.crb = CRB(dim)
        self.ckt = CKTModule(dim,dim,hidden)

    def forward(self,x):
        x = self.norm(x)
        xb = x + self.attn(x)
        xh = x + self.conv(x)
        xb_modulator,xh_modulator = self.ckt(xb,xh)

        return xb_modulator,xh_modulator,xb_modulator + xh_modulator


class high_frequency_attention(nn.Module):
    def __init__(self, dim):
        super(high_frequency_attention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3,stride=1, padding=1, groups=dim)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, high_freq_subband, low_freq_subband):
        x = torch.cat([high_freq_subband, low_freq_subband], dim=1)
        x = self.relu1(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        weighted_high_freq = high_freq_subband * x
        return weighted_high_freq

class High_frequency_enhance(nn.Module):
    def __init__(self, dim):
        super(High_frequency_enhance, self).__init__()
        reduce_dim = dim // 4
        self.num = 3
        # 使用nn.ModuleList来存储卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(dim, reduce_dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(reduce_dim, reduce_dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(reduce_dim, dim, 3, padding=2, dilation=2),
            nn.LeakyReLU(0.2),
        )
        self.high_attention = high_frequency_attention(dim)

    def forward(self, hl, lh, hh,low_frequency_layer):
        hl_res = hl
        lh_res = lh
        hh_res = hh
        for i in range(self.num):
            hl = self.conv(hl)
            high_frequency_attention_hl = self.high_attention(hl,low_frequency_layer[i])
            hl = hl + high_frequency_attention_hl
            lh = self.conv(lh)
            high_frequency_attention_lh = self.high_attention(lh, low_frequency_layer[i])
            lh = lh + high_frequency_attention_lh
            hh = self.conv(hh)
            high_frequency_attention_hh = self.high_attention(hh, low_frequency_layer[i])
            lh = lh + high_frequency_attention_hh

        return hl+hl_res, lh+lh_res, hh+hh_res

class low_frequency_preservation(nn.Module):
    def __init__(self, dim):
        super(low_frequency_preservation, self).__init__()
        reduce_dim = dim // 4
        self.num = 3
        # 使用nn.ModuleList来存储卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(dim, reduce_dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(reduce_dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self,ll):
        low_frequency_layer = []
        for i in range(self.num):
            ll = ll + self.conv(ll)
            low_frequency_layer.append(ll)


        return ll,low_frequency_layer

class ConvWeightFusion(nn.Module):
    def __init__(self, channels):
        super(ConvWeightFusion, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)  # 1x1卷积来学习权重
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feature1, feature2):
        # 计算各自特征的权重
        weight1 = self.sigmoid(self.conv1(feature1))
        weight2 = self.sigmoid(self.conv2(feature2))

        # 使用权重融合特征
        fused_feature = weight1 * feature1 + weight2 * feature2
        return fused_feature

class SwinTransformer(nn.Module):

    def __init__(self, in_chans=3, dim=32, **kwargs):
        super().__init__()
        bias = False

        num_blocks = [1, 1, 1, 1]
        heads = [1, 1, 1, 1]
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], bias=bias,
                             ) for i in range(num_blocks[0])]
                                            )

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1],
                             bias=bias) for i in range(num_blocks[1])])

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2],
                             bias=bias) for i in range(num_blocks[2])])

        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3],
                             bias=bias) for i in range(num_blocks[3])])

        self.down_resize = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-6),
        )

        self.down_pt1 = ResnetBlock(dim, bn=True)

        self.down_pt2 = ResnetBlock(dim * 2, levels=2, bn=True)
        self.down_pt3 = ResnetBlock(dim * 4, levels=3, bn=True)
        self.res = nn.Sequential(
            ResnetBlock(dim * 8, levels=6, down=False, bn=True)
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 16, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2))
        self.up2 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2))

        self.conv = nn.Sequential(
            nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2)

        )

        self.fusion1 = ConvWeightFusion(dim)
        self.fusion2 = ConvWeightFusion(dim * 2)
        self.fusion3 = ConvWeightFusion(dim * 4)
        self.reduce_channel3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1)
        self.reduce_channel2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.reduce_channel1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.High_frequency_layer1 = High_frequency_enhance(dim)
        self.High_frequency_layer2 = High_frequency_enhance(dim * 2)
        self.High_frequency_layer3 = High_frequency_enhance(dim * 4)
        self.low_frequency_layer1 = low_frequency_preservation(dim)
        self.low_frequency_layer2 = low_frequency_preservation(dim * 2)
        self.low_frequency_layer3 = low_frequency_preservation(dim * 4)
        self.pa1 = PALayer(dim * 8)
        self.pa2 = PALayer(dim * 8)
        self.pa3 = PALayer(dim * 4)
        self.pa4 = PALayer(dim * 2)

        self.ca1 = CALayer(dim * 8)
        self.ca2 = CALayer(dim * 8)
        self.ca3 = CALayer(dim * 4)
        self.ca4 = CALayer(dim * 2)
        self.DWT = DWT()
        self.IWT = IWT()

    def forward(self, x):
        CNN_feature = []
        tf_feature = []
        # encodering layer1
        feature = self.down_resize(x)  # [4,32,256,256]
        tf_feature1,CNN_feature1,attn1 = self.encoder_level1(feature)
        CNN_feature.append(CNN_feature1)
        tf_feature.append(tf_feature1)
        ll_layer1_1, hl_layer1_1, lh_layer1_1, hh_layer1_1 = self.DWT(feature)
        ll_layer1_1,low_frequency_layer1_1 = self.low_frequency_layer1(ll_layer1_1)
        hl_layer1_1, lh_layer1_1, hh_layer1_1 = self.High_frequency_layer1(hl_layer1_1, lh_layer1_1, hh_layer1_1,low_frequency_layer1_1)
        feature_wavlet_layer1_1 = self.IWT(ll_layer1_1, hl_layer1_1, lh_layer1_1, hh_layer1_1)
        frequency_spatial_fusion1 = self.fusion1(attn1,feature_wavlet_layer1_1)



        # encodering layer2

        down1 = self.down_pt1(attn1)
        attn2 = down1
        process_CNN = []
        process_tf = []
        for en in self.encoder_level2:
            tf_feature2,CNN_feature2,attn2 = en(attn2)
        CNN_feature.append(tf_feature2)
        tf_feature.append(CNN_feature2)
        ll_layer2_1, hl_layer2_1, lh_layer2_1, hh_layer2_1 = self.DWT(down1)
        ll_layer2_1,low_frequency_layer2_1 = self.low_frequency_layer2(ll_layer2_1)
        hl_layer2_1, lh_layer2_1, hh_layer2_1 = self.High_frequency_layer2(hl_layer2_1, lh_layer2_1, hh_layer2_1,low_frequency_layer2_1)

        feature_wavlet_layer2_1 = self.IWT(ll_layer2_1, hl_layer2_1, lh_layer2_1, hh_layer2_1)
        frequency_spatial_fusion2 = self.fusion2(attn2,feature_wavlet_layer2_1)

        # encodering layer3
        down2 = self.down_pt2(attn2)
        attn3 = down2
        process_CNN.clear()
        process_tf.clear()
        for en in self.encoder_level3:
            tf_feature3, CNN_feature3, attn3 = en(attn3)

        CNN_feature.append(CNN_feature3)
        tf_feature.append(tf_feature3)
        ll_layer3, hl_layer3, lh_layer3, hh_layer3 = self.DWT(down2)
        ll_layer3,low_frequency_layer3 = self.low_frequency_layer3(ll_layer3)
        hl_layer3, lh_layer3, hh_layer3 = self.High_frequency_layer3(hl_layer3, lh_layer3, hh_layer3,low_frequency_layer3)
        feature_wavlet_layer3_1 = self.IWT(ll_layer3, hl_layer3, lh_layer3, hh_layer3)
        frequency_spatial_fusion3 = self.fusion3(attn3,feature_wavlet_layer3_1)

        # encodering layer3
        down3 = self.down_pt3(attn3)
        tf_feature4, CNN_feature4,att4 = self.encoder_level4(down3)
        CNN_feature.append(CNN_feature4)
        tf_feature.append(tf_feature4)
        x1 = self.res(att4)
        x1 = self.pa1(self.ca1(x1))

        x2 = self.up1(x1)
        x2 = torch.cat([x2, frequency_spatial_fusion3], dim=1)
        x2 = self.pa2(self.ca2(x2))
        x2 = self.reduce_channel3(x2)
        x3 = self.up2(x2)


        x3 = torch.cat([x3, frequency_spatial_fusion2], dim=1)
        x3 = self.pa3(self.ca3(x3))
        x3 = self.reduce_channel2(x3)
        x4 = self.up3(x3)


        x4 = torch.cat([x4, frequency_spatial_fusion1], dim=1)
        x4 = self.pa4(self.ca4(x4))
        x4 = self.reduce_channel1(x4)
        res = self.conv(x4)

        return CNN_feature,tf_feature,res


if __name__ == '__main__':
    x = torch.randn((4, 3, 256, 256))
    net = SwinTransformer()

    from thop import profile, clever_format

    flops, params = profile(net, (x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
