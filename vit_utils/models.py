# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class RegRPEAttention(Attention):
    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=64, **kwargs):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, **kwargs)

        self.num_patches = num_patches
        self.learnable = nn.Parameter(torch.randn(num_heads, 2 * num_patches - 1))

        pos_mat = torch.arange(num_patches).unsqueeze(1) - torch.arange(num_patches).unsqueeze(0) + num_patches - 1
        self.pos_mat = pos_mat.unsqueeze(0).expand(num_heads, -1, -1)  # shape: (num_heads, num_patches, num_patches)

        self.head_indices = torch.arange(num_heads).view(num_heads, 1, 1).expand(-1, num_patches, num_patches)

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.num_patches + 1
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        rpe = self.learnable[self.head_indices, self.pos_mat]  # shape: (num_heads, num_patches, num_patches)
        rpe = nn.functional.pad(rpe, pad=(1, 0, 1, 0), mode='constant', value=0)  # Add pad to cover the class token, new shape is (num_heads, N, N)

        attn = (q @ k.transpose(-2, -1))
        attn = attn + rpe.unsqueeze(0)  # add RPE to attention matrix
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PolyRPEAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., grid_size=(8, 8), poly_deg=3, **kwargs):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, **kwargs)
        
        self.poly_deg = poly_deg
        self.poly_coeffs = nn.Parameter(torch.randn(num_heads, poly_deg + 1))

        self.grid_size = grid_size
        # generate patch coordinates from (0, 0) to (height - 1, width - 1), shape is (H*W, 2)
        coords = torch.stack(torch.meshgrid(torch.arange(self.grid_size[0]), torch.arange(self.grid_size[1]), indexing='ij'), dim=-1).reshape(-1, 2)
        # calculate L1-distance between each pair of patches, shape is (H*W, H*W)
        dists = torch.cdist(coords.float(), coords.float(), p=1)
        # polynomial power terms, shape is (poly_deg+1, H*W, H*W)
        self.powers = torch.stack([dists ** i for i in range(poly_deg + 1)], dim=0)
    
    def forward(self, x):
        B, N, C = x.shape
        (H, W) = self.grid_size
        assert N == H * W + 1  # Sequence length is height * width + 1 extra class token
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        
        powers = self.powers.to("cuda")
        rpe = torch.einsum("hd, dxy -> hxy", self.poly_coeffs, powers)  # RPE shape is (num_heads, H*W, H*W)
        rpe = nn.functional.pad(rpe, pad=(1, 0, 1, 0), mode='constant', value=0)  # Add pad to cover the class token, new shape is (num_heads, N, N)

        attn = (q @ k.transpose(-2, -1))
        attn = attn + rpe.unsqueeze(0)  # add RPE to attention matrix
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4, num_patches=None, grid_size=None, poly_deg=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            num_patches=num_patches, grid_size=grid_size, poly_deg=poly_deg)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=8, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = Layer_scale_init_Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,
                dpr_constant=True,init_scale=1e-4,
                mlp_ratio_clstk = 4.0, poly_deg=None, **kwargs):
        super().__init__()
        
        self.dropout_rate = drop_rate

            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        grid_size = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale,
                num_patches=num_patches, grid_size=grid_size, poly_deg=poly_deg)
            for i in range(depth)])
        

        
            
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
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
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x + self.pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
            
        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = nn.functional.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        return x


@register_model  # ViT model with APE only
def vit_ape(img_size=32, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    return model

@register_model  # ViT model with APE and regular additive RPE
def vit_ape_reg_rpe(img_size=32, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, 
        Attention_block=RegRPEAttention, **kwargs)
    return model

@register_model  # ViT model with APE and polynomial additive RPE
def vit_ape_poly_rpe(img_size=32, **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block,
        Attention_block=PolyRPEAttention, poly_deg=3, **kwargs)
    return model