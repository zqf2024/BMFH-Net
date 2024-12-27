import torch.nn.functional as F
import torch
import torch.nn as nn
from Model_util import PALayer, ResnetBlock, CALayer
from einops import rearrange
import torchvision.ops as ops
import torch.nn.init as init
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


class WinvIWT(nn.Module):
    def __init__(self):
        super(WinvIWT, self).__init__()
        self.requires_grad = False

    def forward(self, LL, HH):
        return iwt_init(torch.cat([LL, HH], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return F.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size % 2 == 1, 'Kernel size must be odd'

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return F.sigmoid(x)


class GateNetwork(nn.Module):
    def __init__(self, input_size, num_experts=6, top_k=3):
        super(GateNetwork, self).__init__()
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.fc0 = nn.Linear(input_size, num_experts)
        self.fc1 = nn.Linear(input_size, num_experts)
        self.relu1 = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        init.zeros_(self.fc1.weight)
        self.sp = nn.Softplus()

    def forward(self, x):
        # Flatten the input tensor
        x = self.gap(x) + self.gap2(x)
        x = x.view(-1, self.input_size)
        inp = x
        x = self.fc1(x)
        x = self.relu1(x)
        noise = self.sp(self.fc0(inp))
        noise_mean = torch.mean(noise, dim=1)
        noise_mean = noise_mean.view(-1, 1)
        std = torch.std(noise, dim=1)
        std = std.view(-1, 1)
        noram_noise = (noise - noise_mean) / std
        topk_values, topk_indices = torch.topk(x + noram_noise, k=self.top_k, dim=1)

        mask = torch.zeros_like(x).scatter_(dim=1, index=topk_indices, value=1.0)
        x[~mask.bool()] = float('-inf')

        gating_coeffs = self.softmax(x)

        return gating_coeffs

class MoFE(nn.Module):
    def __init__(self, dim,flag,num_experts=6, k=3):
        super(MoFE, self).__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.k = k

        self.gate = GateNetwork(dim, self.num_experts, self.k)
        if flag == 'C':
            self.expert_networks = nn.ModuleList([
                nn.Sequential(*[nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                            nn.ReLU(),
                            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)]) for i in range(self.num_experts)]
        )
        else:
            self.expert_networks = nn.ModuleList([
                nn.Sequential(*[nn.Conv2d(dim, dim, 3, 1, padding=2, groups=dim, dilation=2),
                                nn.ReLU(),
                                nn.Conv2d(dim, dim, 3, 1, padding=2, groups=dim, dilation=2)]) for i in range(self.num_experts)]
        )

    def forward(self, x):
        cof = self.gate(x)

        out = torch.zeros_like(x).to(x.device)
        for idx in range(self.num_experts):
            if torch.all(cof[:, idx] == 0):
                continue
            mask = torch.where(cof[:, idx] > 0)[0]
            expert_layer = self.expert_networks[idx]
            expert_out = expert_layer(x[mask])
            cof_k = cof[mask, idx].view(-1, 1, 1, 1)
            out[mask] += expert_out * cof_k

        return out

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




class DGBM(nn.Module):
    def __init__(self, dim, num_heads, num_experts=6,k=3):
        super(DGBM, self).__init__()

        self.norm = nn.BatchNorm2d(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn = Channel_Attention(dim, num_heads)
        self.conv0 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.in_size = dim
        self.out_size = dim
        self.num_experts = num_experts
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.k = k

        if self.in_size != self.out_size:
            self.identity = nn.Conv2d(dim, dim, 1, 1, 0)

        self.gate = GateNetwork(dim, self.num_experts, self.k)

        self.gate1 = GateNetwork(dim)
        self.gate2 = GateNetwork(dim)
        self.DE_C = MoFE(dim,'C',num_experts,k)
        self.DE_T = MoFE(dim,'T',num_experts, k)
        self.MLP = nn.Sequential(
            nn.Linear(dim,2 * dim),
            nn.ReLU(True),
            nn.Linear(2 * dim,dim)
        )
        self.MLP1 = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(True),
            nn.Linear(2 * dim, dim),

        )

        self.Unet = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.Conv2d(dim // 2,dim // 4,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(dim // 4, dim // 8, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(dim // 8, dim // 4, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(dim // 4, dim // 2, 3, 1, 1),
            nn.Conv2d(dim // 2, dim, 1),
            nn.Sigmoid()
        )
        self.Unet1 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.Conv2d(dim // 2, dim // 4, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(dim // 4, dim // 8, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(dim // 8, dim // 4, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(dim // 4, dim // 2, 3, 1, 1),
            nn.Conv2d(dim // 2, dim, 1),
            nn.Sigmoid()
        )
        self.softmax = nn.Sigmoid()
        self.X = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.Y = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.M = nn.Conv2d(dim, dim, 3, 1, 1,groups=dim)
        self.N = nn.Conv2d(dim, dim, 3, 1, 1,groups=dim)


    def forward(self, x):
        B,C,H,W = x.shape
        if self.in_size != self.out_size:
            x = self.identity(x)
        x = self.norm(x)
        xb = x + self.attn(x)
        xh = x + self.conv(x)
        diff_C = xb - xh
        diff_T = xh - xb
        Expert_C = self.DE_C(diff_C)
        Expert_T = self.DE_T(diff_T)
        Expert_C_A = self.softmax(self.MLP(self.gap(Expert_C).view(B,-1)).view(B,C,1,1))
        Expert_C_T = self.Unet(Expert_C)
        Expert_T_A = self.softmax(self.MLP1(self.gap(Expert_T).view(B,-1)).view(B,C,1,1))
        Expert_T_T = self.Unet1(Expert_T)
        Expert_C_phsical = Expert_C * Expert_C_T + (1-Expert_C_T) * Expert_C_A
        Expert_T_phsical = Expert_T * Expert_T_T + (1 - Expert_T_T) * Expert_T_A
        X,Y = self.X(Expert_C_phsical),self.Y(Expert_C_phsical)
        M,N = self.M(Expert_T_phsical),self.N(Expert_T_phsical)
        T_res = X * xb + Y
        C_res = M * xh + N
        L = T_res + C_res




        return L


class high_Enhancement(nn.Module):
    def __init__(self, dim):
        super().__init__()


        self.offset_conv_3x3 = nn.Conv2d(dim, 18, kernel_size=3, padding=1)
        self.offset_conv_5x5 = nn.Conv2d(dim, 50, kernel_size=3, padding=1)
        self.offset_conv_7x7 = nn.Conv2d(dim, 98, kernel_size=3, padding=1)
        self.deform_conv_3x3 = ops.DeformConv2d(dim, dim, kernel_size=3, padding=1)
        self.deform_conv_5x5 = ops.DeformConv2d(dim, dim, kernel_size=5, padding=2)
        self.deform_conv_7x7 = ops.DeformConv2d(dim, dim, kernel_size=7, padding=3)

        self.spatial = SpatialAttention()
        self.channel = ChannelAttention(dim * 3)
        self.sig = nn.Sigmoid()


    def forward(self, x):

        offset_3x3 = self.offset_conv_3x3(x)
        offset_5x5 = self.offset_conv_5x5(x)
        offset_7x7 = self.offset_conv_7x7(x)
        attn1 = self.deform_conv_3x3(x,offset_3x3)
        attn2 = self.deform_conv_5x5(x,offset_5x5)
        attn3 = self.deform_conv_7x7(x,offset_7x7)
        attn_res = torch.cat([attn1, attn2, attn3], dim=1)
        attn_spatial = self.spatial(attn_res).expand_as(attn_res)
        attn_channel = self.channel(attn_res).expand_as(attn_res)
        result = attn_spatial + attn_channel
        x1, x2, x3 = result.chunk(3, dim=1)

        attn1_res = attn1 * x1
        attn2_res = attn2 * x2
        attn3_res = attn3 * x3

        attn = attn1_res + attn2_res + attn3_res
        return attn




class WFHE(nn.Module):
    def __init__(self, dim):
        super(WFHE, self).__init__()

        self.conv = nn.Conv2d(dim, dim, 1)
        self.sig = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.de_conv = nn.Conv2d(dim, 1, 1)
        self.anmap = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim // 2, dim, 1),

        )
        self.MLP = nn.Sequential(
            nn.Linear(dim,2 * dim),
            nn.ReLU(True),
            nn.Linear(2 * dim,dim)

        )
        self.resblock = nn.Sequential(
            nn.Conv2d(1,1,7,1,3),
            nn.ReLU(True),
            nn.Conv2d(1, 1, 7, 1,3),
            nn.ReLU(True)

        )


        self.ca = CALayer(dim)
        self.pa = PALayer(dim)
        self.high_Enhancement = high_Enhancement(dim * 3)

    def forward(self, hl, lh, hh, low_frequency_layer):
        B, C, _, _ = hl.shape
        high_information = torch.cat([hl,lh,hh],dim=1)
        high_res = self.high_Enhancement(high_information)
        fre_hazy = torch.fft.rfft2(low_frequency_layer, norm='backward')
        origin_map = torch.abs(fre_hazy)
        mag_hazy = self.anmap(origin_map)
        mag_res = mag_hazy - origin_map
        hazy_weight_distribution_channel = self.sig(self.MLP(self.gap(mag_res).view(B,-1)).view(B,C,1,1))
        mag_attention_avg = torch.mean(mag_res,dim=1,keepdim=True)
        hazy_weight_distribution_spatial = self.sig(mag_attention_avg + self.resblock(mag_attention_avg))
        pha_hazy = torch.angle(fre_hazy)
        pha_hazy_res = pha_hazy * hazy_weight_distribution_channel
        pha_hazy_res = hazy_weight_distribution_spatial * self.conv(pha_hazy_res)
        real_hazy = mag_hazy * torch.cos(pha_hazy_res)
        imag_hazy = mag_hazy * torch.sin(pha_hazy_res)
        fre_out = torch.complex(real_hazy, imag_hazy)
        low_res = low_frequency_layer + torch.fft.irfft2(fre_out, s=(H, W), norm='backward')
        return high_res, low_res


class BMFT(nn.Module):

    def __init__(self, in_chans=3, dim=32, **kwargs):
        super().__init__()

        num_blocks = [1, 1, 1]
        heads = [1, 1, 1]
        self.DGBM1 = nn.Sequential(*[
            DGBM(dim=dim, num_heads=heads[0]
                             ) for i in range(num_blocks[0])]
                                            )

        self.DGBM2 = nn.Sequential(*[
            DGBM(dim=int(dim * 2 ** 1), num_heads=heads[1]
                             ) for i in range(num_blocks[1])])

        self.DGBM3 = nn.Sequential(*[
            DGBM(dim=int(dim * 2 ** 2), num_heads=heads[2]
                            ) for i in range(num_blocks[2])])

        self.down_resize = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-6),
        )

        self.down_pt1 = ResnetBlock(dim, bn=True)
        self.down_pt2 = ResnetBlock(dim * 2, levels=0, bn=True)
        self.down_pt3 = ResnetBlock(dim * 4, levels=0, bn=True)

        self.up1 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2))
        self.up2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.reduce_channel2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.reduce_channel1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.WFHE1 = WFHE(dim)
        self.WFHE2 = WFHE(dim * 2)
        self.WFHE3 = WFHE(dim * 4)
        self.pa1 = PALayer(dim * 4)
        self.pa2 = PALayer(dim * 4)
        self.pa3 = PALayer(dim * 2)
        self.ca1 = CALayer(dim * 4)
        self.ca2 = CALayer(dim * 4)
        self.ca3 = CALayer(dim * 2)
        self.DWT = DWT()
        self.IWT = WinvIWT()


    def forward(self, x):
        # encodering layer1
        feature = self.down_resize(x)  # [4,32,256,256]

        attn1 = self.DGBM1(feature)

        ll_layer1_1, hl_layer1_1, lh_layer1_1, hh_layer1_1 = self.DWT(attn1)
        high_res_1, low_res_1 = self.WFHE1(hl_layer1_1, lh_layer1_1,hh_layer1_1,ll_layer1_1)
        feature_wavlet_layer1_1 = self.IWT(low_res_1, high_res_1)

        down1 = self.down_pt1(feature_wavlet_layer1_1)
        attn2 = self.DGBM2(down1)

        ll_layer2_1, hl_layer2_1, lh_layer2_1, hh_layer2_1 = self.DWT(attn2)
        high_res_2, low_res_2 = self.WFHE2(hl_layer2_1, lh_layer2_1,hh_layer2_1,ll_layer2_1)
        feature_wavlet_layer2_1 = self.IWT(low_res_2, high_res_2)

        # encodering layer3
        down2 = self.down_pt2(feature_wavlet_layer2_1)

        attn3 = self.DGBM3(down2)
        ll_layer3, hl_layer3, lh_layer3, hh_layer3 = self.DWT(attn3)
        high_res_3, low_res_3 = self.WFHE3(hl_layer3, lh_layer3, hh_layer3,ll_layer3)
        feature_wavlet_layer3_1 = self.IWT(low_res_3, high_res_3)


        x1 = self.pa1(self.ca1(feature_wavlet_layer3_1))
        x2 = self.up1(x1)
        x2 = self.DGBM2(x2)
        D_ll_layer2_1, D_hl_layer2_1, D_lh_layer2_1, D_hh_layer2_1 = self.DWT(x2)
        D_high_res_2, D_low_res_2 = self.WFHE2(D_hl_layer2_1, D_lh_layer2_1,D_hh_layer2_1,D_ll_layer2_1)
        D_feature_wavlet_layer2_1 = self.IWT(D_low_res_2, D_high_res_2)

        x3 = torch.cat([D_feature_wavlet_layer2_1, feature_wavlet_layer2_1], dim=1)
        x3 = self.pa2(self.ca2(x3))
        x3 = self.reduce_channel2(x3)
        x4 = self.up2(x3)
        x4 = self.DGBM1(x4)
        D_ll_layer1_1, D_hl_layer1_1, D_lh_layer1_1, D_hh_layer1_1 = self.DWT(x4)
        D_high_res_1, D_low_res_1 = self.WFHE1(D_hl_layer1_1, D_lh_layer1_1, D_hh_layer1_1,D_ll_layer1_1)
        D_feature_wavlet_layer1_1 = self.IWT(D_low_res_1, D_high_res_1)

        x4 = torch.cat([D_feature_wavlet_layer1_1, feature_wavlet_layer1_1], dim=1)
        x4 = self.pa3(self.ca3(x4))
        x4 = self.reduce_channel1(x4)
        res = self.conv(x4)

        return res


if __name__ == '__main__':
    x = torch.randn((4, 3, 256, 256))
    net = BMFT()

    from thop import profile, clever_format

    flops, params = profile(net, (x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)




