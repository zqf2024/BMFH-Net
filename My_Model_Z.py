# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint
# import numpy as np
# from typing import Optional

# from Model_util import PALayer, ResnetBlock, CALayer


# def drop_path_f(x, drop_prob: float = 0., training: bool = False):
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output


# class DropPath(nn.Module):

#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         return drop_path_f(x, self.drop_prob, self.training)


# def window_partition(x, window_size: int):

#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
#     # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows


# def window_reverse(windows, window_size: int, H: int, W: int):

#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
#     # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x


# class PatchEmbed(nn.Module):

#     def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         patch_size = (patch_size, patch_size)
#         self.patch_size = patch_size
#         self.in_chans = in_c
#         self.embed_dim = embed_dim
#         self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x):
#         _, _, H, W = x.shape

#         # padding
#         # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
#         pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
#         if pad_input:
#             # to pad the last 3 dimensions,
#             # (W_left, W_right, H_top,H_bottom, C_front, C_back)
#             x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
#                           0, self.patch_size[0] - H % self.patch_size[0],
#                           0, 0))

#         # 下采样patch_size倍
#         x = self.proj(x)
#         _, _, H, W = x.shape
#         # flatten: [B, C, H, W] -> [B, C, HW]
#         # transpose: [B, C, HW] -> [B, HW, C]
#         x = x.flatten(2).transpose(1, 2)
#         x = self.norm(x)
#         return x, H, W


# class PatchMerging(nn.Module):

#     def __init__(self, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
#         self.norm = norm_layer(4 * dim)

#     def forward(self, x, H, W):

#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"

#         x = x.view(B, H, W, C)

#         # padding
#         # 如果输入feature map的H，W不是2的整数倍，需要进行padding
#         pad_input = (H % 2 == 1) or (W % 2 == 1)
#         if pad_input:
#             # to pad the last 3 dimensions, starting from the last dimension and moving forward.
#             # (C_front, C_back, W_left, W_right, H_top, H_bottom)
#             # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
#             x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

#         x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
#         x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
#         x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
#         x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
#         x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
#         x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

#         x = self.norm(x)
#         x = self.reduction(x)  # [B, H/2*W/2, 2*C]

#         return x


# class Mlp(nn.Module):
#     """ MLP as used in Vision Transformer, MLP-Mixer and related networks
#     """
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features

#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.drop1 = nn.Dropout(drop)
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop2 = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x


# class WindowAttention(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.

#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """

#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # [Mh, Mw]
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5

#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Mh, Mw]
#         coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
#         # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, mask: Optional[torch.Tensor] = None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, Mh*Mw, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         # [batch_size*num_windows, Mh*Mw, total_embed_dim]
#         B_, N, C = x.shape
#         # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
#         # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
#         # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
#         q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

#         # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
#         # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))

#         # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
#         attn = attn + relative_position_bias.unsqueeze(0)

#         if mask is not None:
#             # mask: [nW, Mh*Mw, Mh*Mw]
#             nW = mask.shape[0]  # num_windows
#             # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
#             # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)

#         attn = self.attn_drop(attn)

#         # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
#         # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
#         # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class SwinTransformerBlock(nn.Module):

#     def __init__(self, dim, num_heads, window_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention(
#             dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
#             attn_drop=attn_drop, proj_drop=drop)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#     def forward(self, x, attn_mask):
#         H, W = self.H, self.W
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"

#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, H, W, C)

#         # pad feature maps to multiples of window size
#         # 把feature map给pad到window size的整数倍
#         pad_l = pad_t = 0
#         pad_r = (self.window_size - W % self.window_size) % self.window_size
#         pad_b = (self.window_size - H % self.window_size) % self.window_size
#         x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
#         _, Hp, Wp, _ = x.shape

#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#         else:
#             shifted_x = x
#             attn_mask = None

#         # partition windows
#         x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
#         shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             x = shifted_x

#         if pad_r > 0 or pad_b > 0:
#             # 把前面pad的数据移除掉
#             x = x[:, :H, :W, :].contiguous()

#         x = x.view(B, H * W, C)

#         # FFN
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))

#         return x


# class BasicLayer(nn.Module):

#     def __init__(self, dim, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
#         super().__init__()
#         self.dim = dim
#         self.depth = depth
#         self.window_size = window_size
#         self.use_checkpoint = use_checkpoint
#         self.shift_size = window_size // 2

#         # build blocks
#         self.blocks = nn.ModuleList([
#             SwinTransformerBlock(
#                 dim=dim,
#                 num_heads=num_heads,
#                 window_size=window_size,
#                 shift_size=0 if (i % 2 == 0) else self.shift_size,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 drop=drop,
#                 attn_drop=attn_drop,
#                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                 norm_layer=norm_layer)
#             for i in range(depth)])

#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None

#     def create_mask(self, x, H, W):
#         # calculate attention mask for SW-MSA
#         # 保证Hp和Wp是window_size的整数倍
#         Hp = int(np.ceil(H / self.window_size)) * self.window_size
#         Wp = int(np.ceil(W / self.window_size)) * self.window_size
#         # 拥有和feature map一样的通道排列顺序，方便后续window_partition
#         img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
#         h_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         w_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         cnt = 0
#         for h in h_slices:
#             for w in w_slices:
#                 img_mask[:, h, w, :] = cnt
#                 cnt += 1

#         mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
#         mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
#         attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
#         # [nW, Mh*Mw, Mh*Mw]
#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#         return attn_mask

#     def forward(self, x, H, W):
#         attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
#         for blk in self.blocks:
#             blk.H, blk.W = H, W
#             if not torch.jit.is_scripting() and self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x, attn_mask)
#             else:
#                 x = blk(x, attn_mask)
#         if self.downsample is not None:
#             x = self.downsample(x, H, W)
#             H, W = (H + 1) // 2, (W + 1) // 2

#         return x, H, W


# class FCUUp(nn.Module):
#     """ Transformer patch embeddings -> CNN feature maps
#     """

#     def __init__(self, inplanes, outplanes):
#         super(FCUUp, self).__init__()

#         self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
#         self.bn = nn.BatchNorm2d(outplanes)
#         #self.act = nn.ReLU(True)
#         self.act =  nn.LeakyReLU(0.2)

#     def forward(self, x, x_t):
#         _, _,H,W  = x.shape
#         # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
#         x_r =self.conv_project(x_t)
#         #x_r = self.act(self.bn(self.conv_project(x)))  #[4,64,14,14]
#         x_r = F.interpolate(x_r, size=(H,W))
#         res = x + x_r
#         res = self.act(self.bn(res))

#         return res



# class SwinTransformer(nn.Module):

#     def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
#                  embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
#                  window_size=7, mlp_ratio=4., qkv_bias=True,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, patch_norm=True,
#                  use_checkpoint=False, **kwargs):
#         super().__init__()
#         self.ngf = 32
#         self.num_classes = num_classes
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.patch_norm = patch_norm
#         # stage4输出特征矩阵的channels
#         self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
#         self.mlp_ratio = mlp_ratio

#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

#         # build layers
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             # 注意这里构建的stage和论文图中有些差异
#             # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
#             layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
#                                 depth=depths[i_layer],
#                                 num_heads=num_heads[i_layer],
#                                 window_size=window_size,
#                                 mlp_ratio=self.mlp_ratio,
#                                 qkv_bias=qkv_bias,
#                                 drop=drop_rate,
#                                 attn_drop=attn_drop_rate,
#                                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                                 norm_layer=norm_layer,
#                                 downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
#                                 use_checkpoint=use_checkpoint)

#             self.layers.append(layers)

#         self.expand_block1 = FCUUp(inplanes=embed_dim * 2, outplanes=self.ngf * 2)
#         self.expand_block2 = FCUUp(inplanes=embed_dim * 4, outplanes=self.ngf * 4)
#         self.expand_block3 = FCUUp(inplanes=embed_dim * 8, outplanes=self.ngf * 8)
#         self.expand_block4 = FCUUp(inplanes=embed_dim * 8, outplanes=self.ngf * 16)
#        # ---------------------------------------------CNN分支-----------------------------------

#         self.down_resize = nn.Sequential(
#                                          nn.ReflectionPad2d(3),
#                                          nn.Conv2d(in_chans, self.ngf, kernel_size=7,stride=1,padding=0,bias=False),
#                                          nn.InstanceNorm2d(self.ngf),
#                                         )

#         self.down_pt1 = ResnetBlock(self.ngf,levels=2)
#         self.down_pt2 = ResnetBlock(self.ngf * 2,levels=3)
#         self.down_pt3 = ResnetBlock(self.ngf * 4, levels=4,bn=True)
#         self.down_pt4 = ResnetBlock(self.ngf * 8, levels=5, bn=True)


#         self.up1 = nn.Sequential(
#             nn.ConvTranspose2d(self.ngf * 16,self.ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(self.ngf * 8),
#             )
#         self.up2 = nn.Sequential(
#                                  nn.LeakyReLU(0.2),
#                                  nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
#                                  nn.BatchNorm2d(self.ngf * 4),
#                                  )
#         self.up3 = nn.Sequential(
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm2d(self.ngf * 2),
#             )
#         self.up4 = nn.Sequential(
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(self.ngf * 2, self.ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm2d(self.ngf),
#             )
#         self.conv = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(self.ngf, 3, kernel_size=7, padding=0),
#             nn.Tanh())




#         self.pa1 = PALayer(self.ngf * 16)
#         self.pa2 = PALayer(self.ngf * 8)
#         self.pa3 = PALayer(self.ngf * 4)
#         self.pa4 = PALayer(self.ngf * 2)
#         self.pa5 = PALayer(self.ngf)

#         self.ca1 = CALayer(self.ngf * 16)
#         self.ca2 = CALayer(self.ngf * 8)
#         self.ca3 = CALayer(self.ngf * 4)
#         self.ca4 = CALayer(self.ngf * 2)
#         self.ca5 = CALayer(self.ngf )


#     def forward(self, x):
#         # x: [B, L, C]
#         h = x

#         B,C,_,_ = x.shape
#         x, H, W = self.patch_embed(x)
#         x = self.pos_drop(x)
#         Trans_features = []
#         for layer in self.layers:
#             x, H, W = layer(x, H, W)
#             Trans_x = x.permute(0, 2, 1)
#             Trans_x = Trans_x.contiguous().view(B, -1, H, W)
#             Trans_features.append(Trans_x)



#         #x = self.norm(x)  # [B, L, C]
#         # x = x.transpose(1, 2).reshape(B, -1, H,W)
#         # x = F.interpolate(x, size=h.shape[2:4], mode='bilinear', align_corners=False)
#         # x= self.conv1(x)                   #swin transformer的输出


#         #----------------------------CNN----------------------------
#         down0 = self.down_resize(h)
#         i = 0
#         CNN_features = []
#         down1 = self.down_pt1(down0)
#         down1 = self.expand_block1(down1,Trans_features[0])
#         down2 = self.down_pt2(down1)
#         down2 = self.expand_block2(down2,Trans_features[1])
#         down3 = self.down_pt3(down2)
#         down3 = self.expand_block3(down3,Trans_features[2])
#         down4 = self.down_pt4(down3)
#         down4 = self.expand_block4(down4,Trans_features[3])
#         #x_pt = self.model_res1(down4)
#        #--------------------上采样------------------------
#              #经过一系列残差块
#         x1 = self.ca1(down4)
#         x1 = self.pa1(x1)
#         x1_add = x1 + down4
#         up1 = self.up1(x1_add)
#         up1 = self.ca2(up1)
#         up1 = self.pa2(up1)
#         x2_add = up1 + down3
#         up2 = self.up2(x2_add)
#         up2 = self.ca3(up2)
#         up2 = self.pa3(up2)
#         x3_add = up2 + down2
#         up3 = self.up3(x3_add)
#         up3 = self.ca4(up3)
#         up3 = self.pa4(up3)
#         x4_add = up3 + down1
#         up4 = self.up4(x4_add)
#         up4 = self.ca5(up4)
#         up4 = self.pa5(up4)
#         x5_add = up4 + down0
#         res = self.conv(x5_add)  # [4,3,256,256]

#         return res



# def swin_tiny_patch4_window8_256(num_classes: int = 1000, **kwargs):
#     # trained ImageNet-1K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=8,
#                             embed_dim=96,
#                             depths=(2, 2, 6, 2),
#                             num_heads=(3, 6, 12, 24),
#                             num_classes=1000,)
#     return model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

from Model_util import PALayer, ResnetBlock, CALayer
from einops import rearrange

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output



class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        # Trans_x = x.permute(0, 2, 1)
        # Trans_x = Trans_x.contiguous().view(4, -1, H, W)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes):
        super(FCUUp, self).__init__()

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(outplanes)
        #self.act = nn.ReLU(True)
        self.act =  nn.LeakyReLU(0.2)
        # self.structure_gamma = nn.Parameter(torch.zeros(1))
        # self.texture_gamma = nn.Parameter(torch.zeros(1))
        # self.structure_gate = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
        #               padding=1),
        #     nn.Sigmoid()
        # )
        # self.texture_gate = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
        #               padding=1),
        #     nn.Sigmoid()
        #)
    def forward(self, x, x_t):
        _, _,H,W  = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r =self.conv_project(x_t)
        #x_r = self.act(self.bn(self.conv_project(x)))  #[4,64,14,14]
        x_r = F.interpolate(x_r, size=(H,W))   #x_r和x的尺度一样了
        # energy = torch.cat((x, x_r), dim=1)
        # gate_structure_to_texture = self.structure_gate(energy)
        # gate_texture_to_structure = self.texture_gate(energy)
        # texture_feature = texture_feature + self.texture_gamma * (gate_structure_to_texture * structure_feature)
        # structure_feature = structure_feature + self.structure_gamma * (gate_texture_to_structure * texture_feature)
        res = x * x_r
        res = self.act(self.bn(res))

        return res





class SwinTransformer(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.ngf = 32
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.num_heads_Unet = [2,4,2,1]
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)

            self.layers.append(layers)

        self.expand_block1 = FCUUp(inplanes=embed_dim *2, outplanes=self.ngf)
        self.expand_block2 = FCUUp(inplanes=embed_dim * 4 , outplanes=self.ngf * 2)
        self.expand_block3 = FCUUp(inplanes=embed_dim * 8, outplanes=self.ngf * 4)
        self.expand_block4 = FCUUp(inplanes=embed_dim * 8, outplanes=self.ngf * 8)
       # ---------------------------------------------CNN分支-----------------------------------

        self.down_resize = nn.Sequential(
                                         nn.ReflectionPad2d(3),
                                         nn.Conv2d(in_chans, self.ngf, kernel_size=7,stride=1,padding=0,bias=False),
                                         nn.InstanceNorm2d(self.ngf),
                                        )

        self.down_pt1 = ResnetBlock(self.ngf,levels=2)
        self.down_pt2 = ResnetBlock(self.ngf * 2,levels=3)
        self.down_pt3 = ResnetBlock(self.ngf * 4, levels=4,bn=True)
        #self.down_pt4 = ResnetBlock(self.ngf * 8, levels=5, bn=True)
        # self.attn1 = Attention(self.ngf,self.num_heads_Unet[0])
        # self.attn2 = Attention(self.ngf * 2,self.num_heads_Unet[1])
        # self.attn3 = Attention(self.ngf * 4,self.num_heads_Unet[2])
        # self.attn4 = Attention(self.ngf * 8,self.num_heads_Unet[3])


        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 8,self.ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.ngf * 4),
            )
        self.up2 = nn.Sequential(
                                 nn.LeakyReLU(0.2),
                                 nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.BatchNorm2d(self.ngf * 2),
                                 )
        self.up3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(self.ngf),
            )

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.ngf, 3, kernel_size=7, padding=0),
            nn.Tanh())




        self.pa1 = PALayer(self.ngf * 8)
        self.pa2 = PALayer(self.ngf * 4)
        self.pa3 = PALayer(self.ngf * 2)
        self.pa4 = PALayer(self.ngf)

        self.ca1 = CALayer(self.ngf * 8)
        self.ca2 = CALayer(self.ngf * 4)
        self.ca3 = CALayer(self.ngf * 2)
        self.ca4 = CALayer(self.ngf)



    def forward(self, x):
        # x: [B, L, C]
        h = x
        B,C,_,_ = x.shape
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        Trans_features = []
        for layer in self.layers:
            x, H, W = layer(x, H, W)
            Trans_x = x.permute(0, 2, 1)
            Trans_x = Trans_x.contiguous().view(B, -1, H, W)
            Trans_features.append(Trans_x)

        #x = self.norm(x)  # [B, L, C]
        # x = x.transpose(1, 2).reshape(B, -1, H,W)
        # x = F.interpolate(x, size=h.shape[2:4], mode='bilinear', align_corners=False)
        # x= self.conv1(x)                   #swin transformer的输出


        #----------------------------CNN----------------------------
        down1 = self.down_resize(h)            #feature map[4,32,256,256]
        down1_attn1 = self.expand_block1(down1,Trans_features[0])
        down2 = self.down_pt1(down1_attn1)       #[4,64,128,128]
        down2_attn2 = self.expand_block2(down2,Trans_features[1])
        down3 = self.down_pt2(down2_attn2)        #[4,128,64,64]
        down3_attn3 = self.expand_block3(down3,Trans_features[2])
        down4 = self.down_pt3(down3_attn3)          #[4,256,32,32]
        down4_attn4 = self.expand_block4(down4,Trans_features[3])
        x1 = self.pa1(self.ca1(down4_attn4))
        x1_add = x1 + down4_attn4
        up1 = self.up1(x1_add)               #[4,128,64,64]
        up1 = self.pa2(self.ca2(up1))
        x2_add = up1 + down3_attn3
        up2 = self.up2(x2_add)                 #[4,64,128,128]
        up2 = self.pa3(self.ca3(up2))
        x3_add =up2 + down2_attn2
        up3 = self.up3(x3_add)                   #[4,32,256,256]
        up3 = self.pa4(self.ca4(up3))
        x4_add = up3 + down1_attn1
        res = self.conv(x4_add)
     

        return res



def swin_tiny_patch4_window8_256(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=8,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=1000,)
    return model

if __name__ == '__main__':
    x = torch.randn((4, 3, 256, 256),)
    y = torch.randn((4, 3, 256, 256), )
    #y = torch.ones(4, 3, 16,16)*0.5
    net = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=8,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=1000,
                            )
    print('Discriminator parameters:', sum(param.numel() for param in net.parameters()))
    out = net(x)
    cet = net(y)

