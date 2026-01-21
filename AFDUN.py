
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        return x + self.res(x)


class Block(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.init_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)

        self.res_blocks = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),

        )

        self.final_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        x1 = x
        x = self.init_conv(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)

        x = x + x1
        return x


def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0, stride=32, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    temp = torch.nn.PixelShuffle(32)(temp)
    return temp


class AttBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, bias=True, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.scale = dim ** -0.5
        self.block_size = 16

    def forward(self, x):
        B, C, H, W = x.shape
        bs = self.block_size
        x = self.conv1(x)
        x_blocks = x.reshape(B, C, H // bs, bs, W // bs, bs).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, bs, bs)
        q = self.conv2(x_blocks).reshape(-1, C, bs * bs)
        k = self.conv3(x_blocks).reshape(-1, C, bs * bs).transpose(1, 2)
        att = (k @ q) * self.scale
        att = att.softmax(dim=-1)
        x_attn = (x_blocks.reshape(-1, C, bs * bs) @ att).reshape(-1, C, bs, bs)
        x_out = self.conv4(x_attn)
        x_out = x_out.reshape(B, H // bs, W // bs, C, bs, bs).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return x_out


class ACBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.res1 = ResBlock(dim)
        self.res2 = ResBlock(dim)
        self.attn_branch = AttBlock(dim)
        self.fusion = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        input_x = x
        cnn_out = self.res1(x)
        attn_out = self.attn_branch(x)
        fused = torch.cat([cnn_out, attn_out], dim=1)
        fused_out = self.fusion(fused)
        fused_out = self.res2(fused_out)
        output = input_x + fused_out
        return output



class Basic_Block(nn.Module):
    def __init__(self, dim):
        super(Basic_Block, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([0.5]))
        self.block = Block(dim)
        self.ACblock = ACBlock(dim)


        self.conv1 = nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, padding=0, bias=True)
        self.cat = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)


        self.fuse_momentum = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x, x_prev, t, t_prev, z, Phi, PhiT, PhiTb):

        momentum = self.fuse_momentum(torch.cat([x, x_prev], dim=1))
        v = x + ((t_prev - 1) / t) * momentum


        x_prev = x
        t_prev = t
        t = (1 + (1 + 4 * t_prev ** 2) ** 0.5) / 2


        grad_step_out = v - self.alpha * PhiTPhi_fun(v, Phi, PhiT)
        grad_step_out = grad_step_out + self.alpha * PhiTb


        prox_in = grad_step_out
        prox_feat = self.conv1(prox_in)
        prox_feat = self.cat(torch.cat((prox_feat, z), dim=1))
        prox_feat = self.block(prox_feat)
        prox_feat = self.ACblock(prox_feat)

        z_next = prox_feat
        x_next = self.conv2(z_next)

        return x_next, x_prev, t, t_prev, z_next




class Net(nn.Module):
    def __init__(self, sensing_rate, LayerNo, channel_number):
        super(Net, self).__init__()
        self.LayerNo = LayerNo
        self.measurement = int(sensing_rate * 1024)
        self.base = channel_number
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.measurement, 1024)))

        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.base, 1, kernel_size=1, padding=0, bias=True)
        self.cab = Block(self.base)
        self.ACblock = ACBlock(self.base)
        onelayer = []
        for i in range(self.LayerNo):
            onelayer.append(Basic_Block(self.base))
        self.RND = nn.ModuleList(onelayer)

    def forward(self, x_input_original):

        Phi = self.Phi.contiguous().view(self.measurement, 1, 32, 32)
        PhiT = self.Phi.t().contiguous().view(1024, self.measurement, 1, 1)

        y = F.conv2d(x_input_original, Phi, padding=0, stride=32, bias=None)

        PhiTb = F.conv2d(y, PhiT, padding=0, bias=None)
        PhiTb = nn.PixelShuffle(32)(PhiTb)

        x = PhiTb
        z = self.conv1(x)

        x_prev = x
        t = torch.tensor(1.0, device=x.device)
        t_prev = torch.tensor(1.0, device=x.device)


        for i in range(self.LayerNo):
            x, x_prev, t, t_prev, z = self.RND[i](x, x_prev, t, t_prev, z, Phi, PhiT, PhiTb)

        return x

