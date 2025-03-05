'''
CorrMLP

Original code retrieved from:
https://github.com/MungoMeng/Registration-CorrMLP

Original paper:
Meng M, Fulham M, Bi L, et al.
Advancing Deformable Medical Image Registration with Multi-axis Cross-covariance Attention[J].
arXiv preprint arXiv:2412.18545, 2024.

Modified and tested by:
Haiqiao Wang
2110246069@email.szu.edu.cn
Shenzhen University
'''
import einops
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal


########################################################
# Networks
########################################################

class CorrXCA(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 enc_channels: int = 12,
                 dec_channels: int = 24,
                 num_heads=[16, 8, 4, 2],
                 region_size=6,
                 use_checkpoint: bool = True):
        super().__init__()

        self.Encoder = Conv_encoder(in_channels=in_channels,
                                    channel_num=enc_channels,
                                    use_checkpoint=use_checkpoint)
        self.Decoder = XCA_decoder(in_channels=enc_channels,
                                   channel_num=dec_channels,
                                   num_heads=num_heads,
                                   region_size=region_size,
                                   use_checkpoint=use_checkpoint)

        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

    def forward(self, fixed, moving):
        x_fix = self.Encoder(fixed)
        x_mov = self.Encoder(moving)
        flow = self.Decoder(x_fix, x_mov)
        warped = self.SpatialTransformer(moving, flow)

        return warped, flow

    ########################################################


# Encoder/Decoder
########################################################

class Conv_encoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channel_num: int,
                 use_checkpoint: bool = False):
        super().__init__()

        self.Convblock_1 = Conv_block(in_channels, channel_num, use_checkpoint)
        self.Convblock_2 = Conv_block(channel_num, channel_num * 2, use_checkpoint)
        self.Convblock_3 = Conv_block(channel_num * 2, channel_num * 4, use_checkpoint)
        self.Convblock_4 = Conv_block(channel_num * 4, channel_num * 8, use_checkpoint)
        self.downsample = nn.AvgPool3d(2, stride=2)

    def forward(self, x_in):
        x_1 = self.Convblock_1(x_in)
        x = self.downsample(x_1)
        x_2 = self.Convblock_2(x)
        x = self.downsample(x_2)
        x_3 = self.Convblock_3(x)
        x = self.downsample(x_3)
        x_4 = self.Convblock_4(x)

        return [x_1, x_2, x_3, x_4]


class XCA_decoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 channel_num: int,
                 num_heads=[16, 8, 4, 2],
                 region_size=6,
                 use_checkpoint: bool = False):
        super().__init__()

        self.mlp_11 = CMAXCA_block(in_channels, channel_num, num_heads[3], region_size, use_corr=True,
                                   use_checkpoint=use_checkpoint)
        self.mlp_12 = CMAXCA_block(in_channels * 2, channel_num * 2, num_heads[2], region_size, use_corr=True,
                                   use_checkpoint=use_checkpoint)
        self.mlp_13 = CMAXCA_block(in_channels * 4, channel_num * 4, num_heads[1], region_size, use_corr=True,
                                   use_checkpoint=use_checkpoint)
        self.mlp_14 = CMAXCA_block(in_channels * 8, channel_num * 8, num_heads[0], region_size, use_corr=True,
                                   use_checkpoint=use_checkpoint)

        self.mlp_21 = CMAXCA_block(channel_num, channel_num, num_heads[2], region_size, use_corr=True,
                                   use_checkpoint=use_checkpoint)
        self.mlp_22 = CMAXCA_block(channel_num * 2, channel_num * 2, num_heads[1], region_size, use_corr=True,
                                   use_checkpoint=use_checkpoint)
        self.mlp_23 = CMAXCA_block(channel_num * 4, channel_num * 4, num_heads[0], region_size, use_corr=True,
                                   use_checkpoint=use_checkpoint)

        self.upsample_1 = PatchExpanding_block(embed_dim=channel_num * 2)
        self.upsample_2 = PatchExpanding_block(embed_dim=channel_num * 4)
        self.upsample_3 = PatchExpanding_block(embed_dim=channel_num * 8)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

        self.reghead_1 = RegHead_block(channel_num, use_checkpoint)
        self.reghead_2 = RegHead_block(channel_num * 2, use_checkpoint)
        self.reghead_3 = RegHead_block(channel_num * 4, use_checkpoint)
        self.reghead_4 = RegHead_block(channel_num * 8, use_checkpoint)

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3, x_fix_4 = x_fix
        x_mov_1, x_mov_2, x_mov_3, x_mov_4 = x_mov

        # Step 1
        x_4 = self.mlp_14(x_fix_4, x_mov_4)
        flow_4 = self.reghead_4(x_4)

        # Step 2
        flow_4_up = self.ResizeTransformer(flow_4)
        x_mov_3 = self.SpatialTransformer(x_mov_3, flow_4_up)

        x = self.mlp_13(x_fix_3, x_mov_3)
        x_3 = self.mlp_23(x, self.upsample_3(x_4))

        x = self.reghead_3(x_3)
        flow_3 = x + flow_4_up

        # Step 3
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2 = self.SpatialTransformer(x_mov_2, flow_3_up)

        x = self.mlp_12(x_fix_2, x_mov_2)
        x_2 = self.mlp_22(x, self.upsample_2(x_3))

        x = self.reghead_2(x_2)
        flow_2 = x + flow_3_up

        # Step 4
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1 = self.SpatialTransformer(x_mov_1, flow_2_up)

        x = self.mlp_11(x_fix_1, x_mov_1)
        x_1 = self.mlp_21(x, self.upsample_1(x_2))

        x = self.reghead_1(x_1)
        flow_1 = x + flow_2_up

        return flow_1


########################################################
# Blocks
########################################################

class SpatialTransformer_block(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ResizeTransformer_block(nn.Module):

    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x


class Conv_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.Conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_1 = nn.InstanceNorm3d(out_channels)

        self.Conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.norm_2 = nn.InstanceNorm3d(out_channels)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def Conv_forward(self, x_in):

        x = self.Conv_1(x_in)
        x = self.LeakyReLU(x)
        x = self.norm_1(x)

        x = self.Conv_2(x)
        x = self.LeakyReLU(x)
        x_out = self.norm_2(x)

        return x_out

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.Conv_forward, x_in)
        else:
            x_out = self.Conv_forward(x_in)

        return x_out


class RegHead_block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.reg_head = nn.Conv3d(in_channels, 3, kernel_size=3, stride=1, padding='same')
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.reg_head, x_in)
        else:
            x_out = self.reg_head(x_in)

        return x_out


class PatchExpanding_block(nn.Module):

    def __init__(self, embed_dim: int):
        super().__init__()

        self.up_conv = nn.ConvTranspose3d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(embed_dim // 2)

    def forward(self, x_in):
        x = self.up_conv(x_in)
        x = einops.rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        x_out = einops.rearrange(x, 'b d h w c -> b c d h w')

        return x_out


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    refer to：
    Ali A, Touvron H, Caron M, et al.
    Xcit: Cross-covariance image transformers[J].
    Advances in neural information processing systems, 2021, 34: 20014-20027.
    """

    def __init__(self, dim, num_heads=8, qk_scale=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qk_scale = qk_scale
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = C // 3
        qkv = x.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1) * self.qk_scale
        # k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WinLGXCA(nn.Module):  # Window-based Local and Global XCA
    def __init__(self, num_channels, num_heads, region_size=6, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.patch_size = to_tuple3d(region_size)
        self.N = region_size ** 3
        self.fh = region_size
        self.fw = region_size
        self.fd = region_size

        C = num_channels

        self.in_norm = nn.LayerNorm(C)
        self.in_project = nn.Linear(C, 2 * C)

        # Local branch
        self.l_norm = nn.LayerNorm(C)
        self.l_project = nn.Conv3d(C, 3 * C, 3, 1, 1)
        self.l_attn = XCA(C, num_heads)

        # Global Branch
        self.g_norm = nn.LayerNorm(C)
        self.g_project = nn.Conv3d(C, 3 * C, 3, 1, 1)
        self.g_attn = XCA(C, num_heads)

        self.out_project = nn.Linear(2 * C, C)

    def forward_run(self, x_in):

        B, h, w, d, c = x_in.shape
        x = self.in_norm(x_in)
        x = self.in_project(x)
        Fl, Fg = torch.chunk(x, chunks=2, dim=-1)

        # Local branch
        x = self.l_norm(Fl)
        x = einops.rearrange(x, 'b d h w c -> b c d h w')
        x = self.l_project(x)
        x = einops.rearrange(x, 'b c d h w -> b d h w c')

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.fh - h % self.fh) % self.fh
        pad_b = (self.fw - w % self.fw) % self.fw
        pad_r = (self.fd - d % self.fd) % self.fd
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        gh, gw, gd = x.shape[1] // self.fh, x.shape[2] // self.fw, x.shape[3] // self.fd
        nW = gh * gw * gd

        x = split_images(x, self.patch_size).reshape(B * nW, self.N, 3 * c)  # B, num_windows, patch_size, C

        x = self.l_attn(x)

        x = unsplit_images(x.view(B, -1, self.N, c), grid_size=(gh, gw, gd), patch_size=(self.fh, self.fw, self.fd))
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :d, :].contiguous()

        Fl = Fl + x

        # Global Branch
        x = self.l_norm(Fg)
        x = einops.rearrange(x, 'b d h w c -> b c d h w')
        x = self.l_project(x)
        x = einops.rearrange(x, 'b c d h w -> b d h w c')

        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        x = nonlocal_split_images(x, self.patch_size).reshape(B * self.N, nW, 3 * c)  # B, num_windows, patch_size, C

        x = self.g_attn(x)

        x = nonlocal_unsplit_images(x.view(B, self.N, nW, c), grid_size=(gh, gw, gd),
                                    patch_size=(self.fh, self.fw, self.fd))
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :d, :].contiguous()

        Fg = Fg + x

        # output project
        x = torch.cat([Fl, Fg], dim=-1)
        x = self.out_project(x)

        return x + x_in

    def forward(self, x_in):

        if self.use_checkpoint and x_in.requires_grad:
            x_out = checkpoint.checkpoint(self.forward_run, x_in)
        else:
            x_out = self.forward_run(x_in)

        return x_out


class CMAXCA_block(nn.Module):  # input shape: n, c, h, w, d
    """Correlation with Multi-axis Cross-covariance Attention (CMAXCA) block."""

    def __init__(self, in_channels, num_channels, num_heads, region_size=6, use_corr=True, use_checkpoint=False):
        super().__init__()
        self.use_corr = use_corr
        if use_corr:
            self.Corr = Correlation(max_disp=1, use_checkpoint=use_checkpoint)
            self.Conv = nn.Conv3d(in_channels * 2 + 27, num_channels, kernel_size=3, stride=1, padding='same')
        else:
            self.Conv = nn.Conv3d(in_channels * 2, num_channels, kernel_size=3, stride=1, padding='same')

        self.xcaLayer = WinLGXCA(num_channels, num_heads, region_size, use_checkpoint=use_checkpoint)
        self.channel_attention_block = RCAB(num_channels, use_checkpoint=use_checkpoint)

    def forward(self, x_1, x_2):

        if self.use_corr:
            x_corr = self.Corr(x_1, x_2)
            x = torch.cat([x_1, x_corr, x_2], dim=1)
            x = self.Conv(x)
        else:
            x = torch.cat([x_1, x_2], dim=1)
            x = self.Conv(x)

        x = x.permute(0, 2, 3, 4, 1)  # n,h,w,d,c
        shortcut = x
        x = self.xcaLayer(x)
        x = x + shortcut
        x = self.channel_attention_block(x)
        x_out = x.permute(0, 4, 1, 2, 3)  # n,c,h,w,d

        return x_out


class RCAB(nn.Module):  # input shape: n, h, w, d, c
    """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""

    def __init__(self, num_channels, reduction=4, lrelu_slope=0.2, use_bias=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.LayerNorm = nn.LayerNorm(num_channels)
        self.conv1 = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        self.leaky_relu = nn.LeakyReLU(negative_slope=lrelu_slope)
        self.conv2 = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, bias=use_bias, padding='same')
        self.channel_attention = CALayer(num_channels=num_channels, reduction=reduction)

    def forward_run(self, x):

        shortcut = x
        x = self.LayerNorm(x)

        x = x.permute(0, 4, 1, 2, 3)  # n,c,h,w,d
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 4, 1)  # n,h,w,d,c

        x = self.channel_attention(x)
        x_out = x + shortcut

        return x_out

    def forward(self, x):

        if self.use_checkpoint and x.requires_grad:
            x = checkpoint.checkpoint(self.forward_run, x)
        else:
            x = self.forward_run(x)
        return x


class CALayer(nn.Module):  # input shape: n, h, w, c
    """Squeeze-and-excitation block for channel attention."""

    def __init__(self, num_channels, reduction=4, use_bias=True):
        super().__init__()

        self.Conv_0 = nn.Conv3d(num_channels, num_channels // reduction, kernel_size=1, stride=1, bias=use_bias)
        self.relu = nn.ReLU()
        self.Conv_1 = nn.Conv3d(num_channels // reduction, num_channels, kernel_size=1, stride=1, bias=use_bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in):
        x = x_in.permute(0, 4, 1, 2, 3)  # n,c,h,w,d
        x = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        x = self.Conv_0(x)
        x = self.relu(x)
        x = self.Conv_1(x)
        w = self.sigmoid(x)
        w = w.permute(0, 2, 3, 4, 1)  # n,h,w,d,c

        x_out = x_in * w
        return x_out


class Correlation(nn.Module):
    def __init__(self, max_disp=1, kernel_size=1, stride=1, use_checkpoint=False):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert stride == 1, "stride other than 1 is not implemented"
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.max_disp = max_disp
        self.padlayer = nn.ConstantPad3d(max_disp, 0)

    def forward_run(self, x_1, x_2):

        x_2 = self.padlayer(x_2)
        offsetx, offsety, offsetz = torch.meshgrid([torch.arange(0, 2 * self.max_disp + 1),
                                                    torch.arange(0, 2 * self.max_disp + 1),
                                                    torch.arange(0, 2 * self.max_disp + 1)], indexing='ij')

        w, h, d = x_1.shape[2], x_1.shape[3], x_1.shape[4]
        x_out = torch.cat([torch.mean(x_1 * x_2[:, :, dx:dx + w, dy:dy + h, dz:dz + d], 1, keepdim=True)
                           for dx, dy, dz in zip(offsetx.reshape(-1), offsety.reshape(-1), offsetz.reshape(-1))], 1)
        return x_out

    def forward(self, x_1, x_2):

        if self.use_checkpoint and x_1.requires_grad and x_2.requires_grad:
            x = checkpoint.checkpoint(self.forward_run, x_1, x_2)
        else:
            x = self.forward_run(x_1, x_2)
        return x


########################################################
# Functions
########################################################

def split_images(x, patch_size):  # n, h, w, d, c
    """Image to patches."""

    batch, height, width, depth, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    grid_depth = depth // patch_size[2]

    x = einops.rearrange(
        x, "n (gh fh) (gw fw) (gd fd) c -> n (gh gw gd) (fh fw fd) c",
        gh=grid_height, gw=grid_width, gd=grid_depth, fh=patch_size[0], fw=patch_size[1], fd=patch_size[2])
    return x


def unsplit_images(x, grid_size, patch_size):
    """patches to images."""

    x = einops.rearrange(
        x, "n (gh gw gd) (fh fw fd) c -> n (gh fh) (gw fw) (gd fd) c",
        gh=grid_size[0], gw=grid_size[1], gd=grid_size[2], fh=patch_size[0], fw=patch_size[1], fd=patch_size[2])
    return x


def nonlocal_split_images(x, patch_size):  # n, h, w, d, c
    """Image to patches."""

    batch, height, width, depth, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    grid_depth = depth // patch_size[2]

    x = einops.rearrange(
        x, "n (gh fh) (gw fw) (gd fd) c -> n (fh fw fd) (gh gw gd)  c",
        gh=grid_height, gw=grid_width, gd=grid_depth, fh=patch_size[0], fw=patch_size[1], fd=patch_size[2])
    return x


def nonlocal_unsplit_images(x, grid_size, patch_size):
    """patches to images."""

    x = einops.rearrange(
        x, "n (fh fw fd) (gh gw gd)  c -> n (gh fh) (gw fw) (gd fd) c",
        gh=grid_size[0], gw=grid_size[1], gd=grid_size[2], fh=patch_size[0], fw=patch_size[1], fd=patch_size[2])
    return x


def to_tuple3d(a):
    return (a, a, a)


if __name__ == '__main__':
    model = CorrXCA(in_channels=1, enc_channels=2, dec_channels=4).cuda()
    A = torch.ones(2, 1, 80, 96, 80).cuda()
    B = torch.ones(2, 1, 80, 96, 80).cuda()
    output = model(B, A)
    for s in output:
        print(s.shape)
