
import os
from model_repo.S83TOA import *
from einops import rearrange
import torch.nn as nn
import torch
import torch.nn.functional as F

def flunce_pad(x):
    _,_,h,w = x.shape
    size = 128
    res = [0, 0]
    if h % size != 0:
        res[0] = (size - h % size)
    if w % size != 0:
        res[1] = (size - w % size)
    x = F.pad(x, [int(res[1]/2), res[1]-int(res[1]/2), int(res[0]/2), res[0]-int(res[0]/2)], mode='reflect')
    return x

def de_pad(x, nh, nw):
    _,_,h,w = x.shape
    # print(x.shape)
    pad = (h - nh), (w - nw)
    # print(pad)
    return x[:,:,int(pad[0]/2):h-(pad[0]-int(pad[0]/2)),int(pad[1]/2):w-(pad[1]-int(pad[1]/2))].sigmoid()

class ResidualConv(nn.Module):
    def __init__(self, inC, outC):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inC, inC, 3, 1, 1, groups=inC),
            nn.GroupNorm(outC, outC),
            nn.GELU(),
            nn.Conv2d(inC, outC, 1, 1, 0),
        )
        self.norm = nn.GroupNorm(inC, inC)
    def forward(self, x):
        shortcut = x
        return self.norm(shortcut + self.conv(x))

class SPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            ResidualConv(32, 32),
            ResidualConv(32, 32),
        )
        self.encoder2 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            # ResidualConv(32, 64),
            nn.Conv2d(32, 64, 3, 1, 1),
            ResidualConv(64, 64),
            ResidualConv(64, 64),
        )
        self.encoder3 = nn.Sequential(
            nn.AvgPool2d(2, 2),
            # ResidualConv(64, 128),
            nn.Conv2d(64, 128, 3, 1, 1),
            ResidualConv(128, 128),
            ResidualConv(128, 128),
        )

    def forward(self, x):
        end_1 = self.encoder1(x)
        end_2 = self.encoder2(end_1)
        end_3 = self.encoder3(end_2)
        return [end_1, end_2, end_3]


class Rearange(nn.Module):
    def __init__(self, op):
        self.op = op
        super().__init__()
    def forward(self, x):
        return rearrange(x, self.op)

class PartProject(nn.Module):
    def __init__(self, dimlist):
        super().__init__()
        self.conv1 = nn.Sequential(
            ImageMerging(dimlist[0], 4)
        )
        self.conv2 = nn.Sequential(
            ImageMerging(dimlist[1], 8),
        )
        self.conv3 = nn.Sequential(
            ImageMerging(dimlist[2], 16),
        )
    def forward(self, x):
        y = [self.conv1(x[0]).sigmoid(), self.conv2(x[1]).sigmoid(), self.conv3(x[2]).sigmoid()]
        for i in range(len(y)):
            y[i] = y[i].unsqueeze(-1).repeat(1, 1, 1, 1, 32).flatten(3)
        return y


class ImageMerging(nn.Module):
    def __init__(self, dim, out, norm_layer=nn.GroupNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(dim, out, 4, 4, 0, bias=False)
        # self.norm = nn.BatchNorm2d(out)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.reduction(x)
        x = rearrange(x, 'B C H W -> B H W C', H=H//4, W=W//4)

        return x

class Net(nn.Module):
    def __init__(self, pretrain=None):
        super(Net, self).__init__()
        self.name = os.path.basename(__file__).split('.')[0]
        self.FPN = swin_base_patch4_window7_224(pretrained=pretrain, embed_dim=128, num_heads=(4, 8, 16),
                                     patch_size=4, window_sizes=[8, 8, 8], depths=(2, 2, 18))
        self.SPN = SPN()
        self.SPAM = PartProject([32, 64, 128])

        self.deseq = embeding_to_image_none
        self.PFMpre = new_Emb2Img()
        self.decode = decode()
        self.size = 256
    def forward(self, x):
        _,_,h,w = x.shape
        # x = Ft.resize(x, (self.size, self.size))
        x = flunce_pad(x)
        part_points = self.SPN(x)
        SPAs = self.SPAM(part_points)

        end_points = self.FPN(x, SPAs)
        end_points = self.PFMpre(self.deseq(end_points))

        x = self.decode([*part_points, *end_points])
        x = de_pad(x, h, w)
        # x = Ft.resize(x, (h, w), PIL.Image.NEAREST)
        return x


def embeding_to_image_none(x):
    for i in range(len(x)):
        H, W = x[i].resolution
        x[i] = rearrange(x[i], 'b (h w) c -> b c h w ', h=H, w=W)
    return x

class C_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C_Layer, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                                    nn.GroupNorm(out_channels, out_channels),
                                    nn.GELU()])
    def forward(self, x):
        x = self.conv(x)
        return x

class FFM(nn.Module):
    def __init__(self, in_channel,
                 pre_conv=C_Layer,
                 ):
        super(FFM, self).__init__()
        self.pre_conv1 = pre_conv(in_channel[0], in_channel[0])  # pre kernal_size=1
        self.pre_conv2 = pre_conv(in_channel[1], in_channel[1])
        self.upsample2 = Up_sample_wN(in_channel[1], rate=2, factor=2)
        self.norm = nn.GroupNorm((in_channel[1] * 2) // (2 ** 2), (in_channel[1] * 2) // (2 ** 2))
        in_chan = int(in_channel[0] + in_channel[1] // 2)
        self.forway = C_Layer(in_chan, in_channel[0])

    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        x2 = self.norm((self.upsample2(x2)))
        return self.forway(torch.cat([x1, x2], dim=1))
        # return x1 + x2

class PFM(nn.Module):
    def __init__(self, in_channel,
                 pre_conv=C_Layer,
                 ):
        super(PFM, self).__init__()
        self.pre_conv1 = pre_conv(in_channel[0], in_channel[0])  # pre kernal_size=1
        self.pre_conv2 = pre_conv(in_channel[1], in_channel[1])
        in_chan = int(in_channel[0] + in_channel[1])
        self.forway = C_Layer(in_chan, in_channel[0])
    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        return self.forway(torch.cat([x1, x2], dim=1))

class Up_sample_wN(nn.Module):
    def __init__(self, in_channel, rate=0, factor=0, kernel_size=3, padding=1):
        super(Up_sample_wN, self).__init__()
        self.convhigh_subConv = nn.Sequential(
                *[nn.Conv2d(in_channel, in_channel * rate, kernel_size=kernel_size, padding=padding),
                  nn.GELU()])
        self.convhigh_up = nn.PixelShuffle(factor)
    def forward(self, input):
        convhigh_subConv = self.convhigh_subConv(input)
        up_sample = self.convhigh_up(convhigh_subConv)
        return up_sample

class embeding_to_image(nn.Module):
    """ 一个pach 扩充 4 倍"""
    def __init__(self, in_channel=0, rate=0, factor=0, G=1):
        super(embeding_to_image, self).__init__()
        self.upsample1 = Up_sample_wN(in_channel, rate, factor)
        self.norm = nn.GroupNorm((in_channel * rate) // (2 ** factor), (in_channel * rate) // (2 ** factor))
    def forward(self, x):
        x = self.upsample1(x)
        return self.norm(x)

class new_Emb2Img(nn.Module):
    def __init__(self):
        super(new_Emb2Img, self).__init__()
        ges = [[64, 32],
               [128, 256],
               [256, 128],
               ]
        self.s1 = nn.Sequential(
            *[
                embeding_to_image(in_channel=128, rate=2, factor=2, G=ges[0][0]),
                embeding_to_image(in_channel=128 // 2, rate=2, factor=2, G=ges[0][1])
            ])
        self.s2 = nn.Sequential(
            *[
                embeding_to_image(in_channel=256, rate=2, factor=2,  G=ges[1][0]),
                embeding_to_image(in_channel=256 // 2, rate=2, factor=2,  G=ges[1][1]),
            ])
        self.s3 = nn.Sequential(
            *[
                embeding_to_image(in_channel=512, rate=2, factor=2, G=ges[2][0]),
                embeding_to_image(in_channel=512 // 2, rate=2, factor=2, G=ges[2][1]),
            ])

    def forward(self, x):
        s1 = self.s1(x[0])
        s2 = self.s2(x[1])
        s3 = self.s3(x[2])
        return [s1, s2, s3]


class decode(nn.Module):  # 解码网络的反卷积过程
    def __init__(self):
        super(decode, self).__init__()
        inchanle = 32
        self.share_part = None
        self.prefuse1 = PFM((inchanle, inchanle))
        self.prefuse2 = PFM((inchanle * 2, inchanle * 2))
        self.prefuse3 = PFM((inchanle * 4, inchanle * 4))

        self.leve1_1 = FFM((inchanle * 1, inchanle * 2))
        self.leve1_2 = FFM((inchanle * 2, inchanle * 4))
        self.leve2_1 = FFM((inchanle * 1, inchanle * 2))
        self.out = nn.Conv2d(inchanle, 1, kernel_size=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 1e-2)  # 正态分布

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, t_o):

        t_o[0] = self.prefuse1(t_o[0], t_o[3])
        t_o[1] = self.prefuse2(t_o[1], t_o[4])
        t_o[2] = self.prefuse3(t_o[2], t_o[5])

        leve1_1 = self.leve1_1(t_o[0], t_o[1])
        leve1_2 = self.leve1_2(t_o[1], t_o[2])
        leve2_1 = self.leve2_1(leve1_1, leve1_2)
        t_o = self.out(leve2_1)
        return t_o


if __name__ == '__main__':
        m = Net(r'none')
        x = torch.rand([1, 3, 128, 128])
        y = m(x)
        for i in y:
            print(i.shape)
