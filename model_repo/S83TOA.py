
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import logging
import math
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg
from timm.models.layers import PatchEmbed, Mlp, DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import checkpoint_filter_fn, _init_vit_weights
from einops import rearrange
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'swin_base_patch4_window7_224': _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth',
    )
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):

    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows.view(-1, window_size[0] * window_size[1], C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])   # [0-6]
        coords_w = torch.arange(self.window_size[1])   # [0-6]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)


    def get_position(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask, stagein):

        B_, N, C = x.shape

        if stagein is not None:
            x = x * stagein
        else:
            x = x


        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # num B head L D
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn + self.get_position() + mask

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads,
                 input_resolution = None, shift_size=0,  # 废物参数
                  window_size=7, roll=False,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 whichone=0):
        super().__init__()
        self.whichone = whichone
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = [window_size, window_size]
        self.mlp_ratio = mlp_ratio
        self.shift_size = [shift_size, shift_size]

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.roll = roll


    def getmask(self, window_size, shift_size, H, W, device):
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))

        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, window_size[0] * window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask.unsqueeze(1)

    def forward(self, x,  stagein = None):
        H, W = x.resolution
        if self.roll == True:
            mask = self.getmask(self.window_size, self.shift_size, H, W, device=x.device)
        else:
            mask = 0

        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        window_size = self.window_size

        if max(self.shift_size) > 0:
            may_shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            may_shifted_x = x  # B, H, W, C
        x_windows = window_partition(may_shifted_x, window_size)

        # if stagein is not None and self.roll == False:
        #     stagein_windows = window_partition(stagein, window_size)
        # else:
        #     stagein_windows = None
        # print(self.whichone)
        if self.whichone in STDLIST and stagein is not None:
            # print(self.whichone)
            stagein_windows = window_partition(stagein, window_size)
        else:
            stagein_windows = None

        attn_windows = self.attn(x_windows, mask, stagein_windows)
        shifted_x = window_reverse(attn_windows, window_size, H, W)  # B H' W' C

        if max(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x.resolution = H, W
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = x.resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0,3,1,2)
        x = x.permute(0, 2, 3, 1)
        # x = rearrange(x, 'B C H W -> B H W C')
        H, W = x.shape[1], x.shape[2]
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x.resolution = (H//2, W//2)
        return x

class ImageMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.pool = nn.AvgPool2d(2, 2)
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 4 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # x = self.pool(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'B C H W -> B H W C')

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = rearrange(x, 'B (H W) C -> B H W C', H=H//2, W=W//2)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 form_num = 0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim,
                                 input_resolution=input_resolution,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 roll=False if (i % 2 == 0) else True,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 whichone=form_num + i+1)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, out_feature, stagein = None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)  # 一种不保存中间激活的运算，而是在反向传播的时候重新计算
            else:
                x = blk(x, stagein)
        #下采样的时候要调用下采样方法，这里这个方法是patch merging。
        if self.downsample is not None:
            out_feature.append(x)
            x = self.downsample(x)
        return x



class SwinTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_sizes=None, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        form_num = 0

        for i_layer in range(self.num_layers):

            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_sizes[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               form_num = form_num)
            form_num += depths[i_layer]
            self.layers.append(layer)

        self.downsample1 = ImageMerging(None, 32)
        self.downsample2 = ImageMerging(None, 64)
        self.downsample3 = ImageMerging(None, 128)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, stage):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        #主要定义在layer中

        feature = []
        for index in range(len(self.layers)):
            layer = self.layers[index]
            # if index in stagein:
                # print(index)
            x = layer(x, feature, stage[index])
            # else:
            #     x = layer(x, feature)
        feature.append(x)

        return feature

    def forward(self, x, partpoints):
        x = self.forward_features(x, partpoints)
        return x


def _create_swin_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        SwinTransformer, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        # pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_filter_fn=fiter_pretrain,
        pretrained_strict=False,
        **kwargs)

    return model

def fiter_pretrain(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    pop_dict = [
        'layers.0.blocks.1.attn_mask',
                'layers.1.blocks.1.attn_mask',
                'layers.2.blocks.1.attn_mask',
                'layers.2.blocks.3.attn_mask',
                'layers.2.blocks.5.attn_mask',
                'layers.2.blocks.7.attn_mask',
                'layers.2.blocks.9.attn_mask',
                'layers.2.blocks.11.attn_mask',
                'layers.2.blocks.13.attn_mask',
                'layers.2.blocks.15.attn_mask',
                'layers.2.blocks.17.attn_mask',
                'norm.weight',
                'norm.bias',
                'head.weight',
                'head.bias'
                ]
    # pop index
    blocks = 4
    layers = [2, 2, 18, 2]
    for i in range(blocks):
        for j in range(layers[i]):
            temp = 'layers.%s.blocks.%s.attn.relative_position_index'%(i, j)
            pop_dict.append(temp)

    for i in pop_dict:
        state_dict['model'].pop(i)

    out_dict = {}
    if 'model' in state_dict:
        state_dict = state_dict['model']

    index = 0
    num_block = 0


    for k, v in state_dict.items():

        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = adapt_input_conv(I, v)

        if 'attn.relative_position_bias_table' in k:
            name = 'layers.%s.blocks.%s.attn.relative_position_bias_table'%(num_block, index)
            index += 1
            if index == layers[num_block]:
                num_block += 1
                index = 0
            if name in dict(model.named_parameters()).keys():
                v = resize_pos_embed(v, dict(model.named_parameters())[name])
        out_dict[k] = v

    add_dict = (
        'layers.%s.blocks.%s.norm1.weight',
        'layers.%s.blocks.%s.norm1.bias',
        'layers.%s.blocks.%s.attn.relative_position_bias_table',
        'layers.%s.blocks.%s.attn.qkv.weight',
        'layers.%s.blocks.%s.attn.qkv.bias',
        'layers.%s.blocks.%s.attn.proj.weight',
        'layers.%s.blocks.%s.attn.proj.bias',
        'layers.%s.blocks.%s.norm2.weight',
        'layers.%s.blocks.%s.norm2.bias',
        'layers.%s.blocks.%s.mlp.fc1.weight',
        'layers.%s.blocks.%s.mlp.fc1.bias',
        'layers.%s.blocks.%s.mlp.fc2.weight',
        'layers.%s.blocks.%s.mlp.fc2.bias',
    )

    for k, v in model.named_parameters():
        if k not in state_dict.keys():
            k_list = k.split('.')
            if k_list[0] == 'layers' and k_list[2] == 'blocks':
                # print(k_list[1], k_list[3])
                for i in add_dict:
                    state_dict[i % (int(k_list[1]), int(k_list[3]))] = state_dict[
                        i % (int(k_list[1]), int(k_list[3]) - 2)]
                    # print('add:'+i % (int(k_list[1]), int(k_list[3])))

    return out_dict


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans != 3:
        print('resize: patch_embed.proj.weight')
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:

            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


_logger = logging.getLogger(__name__)
def resize_pos_embed(posemb, posemb_new):
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[0]
    posemb_grid = posemb[:, :]
    gs_old = int(math.sqrt(len(posemb_grid[:, 0])))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    # posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    posemb = posemb_grid.squeeze(0)
    return posemb

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # preconv = self.preconv(x)
        x = self.proj(x)  # B Ph*Pw C
        res = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        x.resolution =res
        return x

@register_model
def swin_base_patch4_window7_224(pretrained=False, **kwargs):
    """ Swin-B @ 224x224, pretrained ImageNet-22k, fine tune 1k
    """
    # model_kwargs = dict(
    #     patch_size=4, window_size=None, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    # return _create_swin_transformer('swin_base_patch4_window7_224', pretrained=pretrained, **model_kwargs)
    model_kwargs = dict(
         **kwargs)
    return _create_swin_transformer('swin_base_patch4_window7_224', pretrained=pretrained, **model_kwargs)


STDLIST = [1, 3, 5]
if __name__ == '__main__':
    m = swin_base_patch4_window7_224(pretrained=True, embed_dim=128, num_heads=(4, 8, 16),
                                     patch_size=4, window_sizes=[8, 8, 8], depths=(2, 2, 18))
    x = torch.randn([1, 3, 128, 128])
    part_list = [
        torch.randn([1, 32, 32, 128]),
        torch.randn([1, 16, 16, 256]),
        torch.randn([1, 8, 8, 512]),
    ]
    y = m(x, part_list)
    for i in y:
        print(i.shape)

