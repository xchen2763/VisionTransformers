"""
This code was originally obtained from:
https://github.com/naver-ai/rope-vit
and
https://github.com/google-research/google-research/tree/master/performer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Tuple

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from models_v2 import vit_models, Layer_scale_init_Block, Attention

def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def compute_mixed_cis(freqs, t_x, t_y, num_heads):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)

    return freqs_cis


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

def relu_kernel_transformation(x, projection_matrix=None, numerical_stabilizer=0.001):
    """
    Computes features for the ReLU-kernel.

    Computes random features for the ReLU kernel from
    https://arxiv.org/pdf/2009.14794.pdf.

    Args:
        data: input data tensor of the shape [B, L, H, D], where: B - batch
        dimension, L - attention dimensions, H - heads, D - features.
        is_query: indicates whether input data is a query oor key tensor.
        projection_matrix: random Gaussian matrix of shape [M, D], where M stands
        for the number of random features and each D x D sub-block has pairwise
        orthogonal rows.
        numerical_stabilizer: small positive constant for numerical stability.

    Returns:
        Corresponding kernel feature map.
    """
    if projection_matrix is None:
        return torch.relu(x) + numerical_stabilizer
    else:
        ratio = 1.0 / torch.sqrt(float(projection_matrix.shape[0]))
        data_dash = ratio * torch.einsum("blhd,md->blhm", x, projection_matrix)
        return torch.relu(data_dash) + numerical_stabilizer

def noncausal_numerator(qs, ks, vs):
    """Computes not-normalized FAVOR noncausal attention AV.

    Args:
        qs: query_prime tensor of the shape [L,B,H,M].
        ks: key_prime tensor of the shape [L,B,H,M].
        vs: value tensor of the shape [L,B,H,D].

    Returns:
        Not-normalized FAVOR noncausal attention AV.
    """
    kvs = torch.einsum("lbhm,lbhd->bhmd", ks, vs)
    return torch.einsum("lbhm,bhmd->lbhd", qs, kvs)

def noncausal_denominator(qs, ks):
    """Computes FAVOR normalizer in noncausal attention.

    Args:
        qs: query_prime tensor of the shape [L,B,H,M].
        ks: key_prime tensor of the shape [L,B,H,M].

    Returns:
        FAVOR normalizer in noncausal attention.
    """
    all_ones = torch.ones([ks.shape[0]], device='cuda:0')
    ks_sum = torch.einsum("lbhm,l->bhm", ks, all_ones)
    return torch.einsum("lbhm,bhm->lbh", qs, ks_sum)

def favor_attention(q, k, v, kernel_transformation=relu_kernel_transformation,
                    projection_matrix=None):
    """Computes FAVOR normalized attention.

    Args:
        query: query tensor.
        key: key tensor.
        value: value tensor.
        kernel_transformation: transformation used to get finite kernel features.
        projection_matrix: projection matrix to be used.

    Returns:
        FAVOR normalized attention.
    """
    q = q.permute(0, 2, 1, 3)  # [B,L,H,D]
    k = k.permute(0, 2, 1, 3)  # [B,L,H,D]
    v = v.permute(0, 2, 1, 3)  # [B,L,H,D]

    q_prime = kernel_transformation(q, projection_matrix)  # Both q_prime and k_prime have size [B,L,H,D] after、
    k_prime = kernel_transformation(v, projection_matrix)  # relu kernel transformation

    q_prime = q_prime.permute(1, 0, 2, 3)  # [L,B,H,D]
    k_prime = k_prime.permute(1, 0, 2, 3)  # [L,B,H,D]
    v = v.permute(1, 0, 2, 3)  # [L,B,H,D]

    # non-causal nominator and denominator
    av_attention = noncausal_numerator(q_prime, k_prime, v)  # [L,B,H,D]
    attention_normalizer = noncausal_denominator(q_prime, k_prime)  # [L,B,H]

    av_attention = av_attention.permute(1, 0, 2, 3)  # [B,L,H,D]
    attention_normalizer = attention_normalizer.permute(1, 0, 2)  # [B,L,H]
    attention_normalizer = torch.unsqueeze(attention_normalizer,
                                           len(attention_normalizer.shape))  # [B,L,H,1]
    
    return av_attention / attention_normalizer  # [B,L,H,D]


class PerformerRoPEAttention(Attention):
    """Multi-head Attention block with relative position embeddings and performer mechanism."""
    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,L,D]
        
        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)

        attn = favor_attention(q, k, v, kernel_transformation=relu_kernel_transformation,
                               projection_matrix=None)

        x = attn.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
class Performer_RoPE_Layer_scale_init_Block(Layer_scale_init_Block):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = PerformerRoPEAttention
        super().__init__(*args, **kwargs)

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), freqs_cis=freqs_cis))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        
        return x

class performer_rope_vit_models(vit_models):
    def __init__(self, rope_theta=100.0, rope_mixed=False, use_ape=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        img_size = kwargs['img_size'] if 'img_size' in kwargs else 224
        patch_size = kwargs['patch_size'] if 'patch_size' in kwargs else 16
        num_heads = kwargs['num_heads'] if 'num_heads' in kwargs else 12
        embed_dim = kwargs['embed_dim'] if 'embed_dim' in kwargs else 768
        mlp_ratio = kwargs['mlp_ratio'] if 'mlp_ratio' in kwargs else 4.
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        
        self.use_ape = use_ape
        if not self.use_ape:
            self.pos_embed = None            
        
        self.rope_mixed = rope_mixed
        self.num_heads = num_heads
        self.patch_size = patch_size
        
        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)
            
            freqs = []
            for i, _ in enumerate(self.blocks):
                freqs.append(
                    init_random_2d_freqs(dim=embed_dim // num_heads, num_heads=num_heads, theta=rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(self.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
            
            t_x, t_y = init_t_xy(end_x = img_size // patch_size, end_y = img_size // patch_size)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        else:
            self.compute_cis = partial(compute_axial_cis, dim=embed_dim//num_heads, theta=rope_theta)
            
            freqs_cis = self.compute_cis(end_x = img_size // patch_size, end_y = img_size // patch_size)
            self.freqs_cis = freqs_cis
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'freqs'}
        
    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_ape:
            pos_embed = self.pos_embed
            if pos_embed.shape[-2] != x.shape[-2]:
                img_size = self.patch_embed.img_size
                patch_size = self.patch_embed.patch_size
                pos_embed = pos_embed.view(
                    1, (img_size[1] // patch_size[1]), (img_size[0] // patch_size[0]), self.embed_dim
                ).permute(0, 3, 1, 2)
                pos_embed = F.interpolate(
                    pos_embed, size=(H // patch_size[1], W // patch_size[0]), mode='bicubic', align_corners=False
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            x = x + pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.rope_mixed:
            if self.freqs_t_x.shape[0] != x.shape[1] - 1:
                t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            
            for i , blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis[i])
        else:
            if self.freqs_cis.shape[0] != x.shape[1] - 1:
                freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size)
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(x.device)
            
            for i , blk in enumerate(self.blocks):
                x = blk(x, freqs_cis=freqs_cis)
                
        x = self.norm(x)
        x = x[:, 0]
        
        return x

# Performer RoPE-Axial
@register_model
def performer_rope_axial_small_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = performer_rope_vit_models(
        img_size = img_size, patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Performer_RoPE_Layer_scale_init_Block, Attention_block=PerformerRoPEAttention,
        rope_theta=100.0, rope_mixed=False, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def performer_rope_axial_base_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = performer_rope_vit_models(
        img_size = img_size, patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Performer_RoPE_Layer_scale_init_Block, Attention_block=PerformerRoPEAttention,
        rope_theta=100.0, rope_mixed=False, **kwargs)
    return model

# Performer RoPE-Mixed
@register_model
def performer_rope_mixed_small_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = performer_rope_vit_models(
        img_size = img_size, patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Performer_RoPE_Layer_scale_init_Block, Attention_block=PerformerRoPEAttention,
        rope_theta=10.0, rope_mixed=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def performer_rope_mixed_base_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = performer_rope_vit_models(
        img_size = img_size, patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Performer_RoPE_Layer_scale_init_Block, Attention_block=PerformerRoPEAttention,
        rope_theta=10.0, rope_mixed=True, **kwargs)
    return model

# Performer RoPE-Axial + APE
@register_model
def performer_rope_axial_ape_small_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = performer_rope_vit_models(
        img_size = img_size, patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Performer_RoPE_Layer_scale_init_Block, Attention_block=PerformerRoPEAttention,
        rope_theta=100.0, rope_mixed=False, use_ape=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def performer_rope_axial_ape_base_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = performer_rope_vit_models(
        img_size = img_size, patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Performer_RoPE_Layer_scale_init_Block, Attention_block=PerformerRoPEAttention,
        rope_theta=100.0, rope_mixed=False, use_ape=True, **kwargs)
    return model

# Performer RoPE-Mixed + APE
@register_model
def performer_rope_mixed_ape_small_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = performer_rope_vit_models(
        img_size = img_size, patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Performer_RoPE_Layer_scale_init_Block, Attention_block=PerformerRoPEAttention,
        rope_theta=10.0, rope_mixed=True, use_ape=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def performer_rope_mixed_ape_base_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = performer_rope_vit_models(
        img_size = img_size, patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Performer_RoPE_Layer_scale_init_Block, Attention_block=PerformerRoPEAttention,
        rope_theta=10.0, rope_mixed=True, use_ape=True, **kwargs)
    return model