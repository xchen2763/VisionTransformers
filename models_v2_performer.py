"""
This code was originally obtained from:
https://github.com/naver-ai/rope-vit
and
https://github.com/google-research/google-research/tree/master/performer
"""

import math

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from distutils.version import LooseVersion

TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')


class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dim_heads = head_dim
        self.a = None

    def orthogonal_matrix_chunk(self, cols, device=None):
        unstructured_block = torch.randn((cols, cols), device=device)
        if TORCH_GE_1_8_0:
            q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
        else:
            q, r = torch.qr(unstructured_block.cpu(), some=True)
        q, r = map(lambda t: t.to(device), (q, r))
        return q.t()

    def gaussian_orthogonal_random_matrix(self, nb_rows, nb_columns, scaling=0, device=None):
        nb_full_blocks = int(nb_rows / nb_columns)
        block_list = []
        for _ in range(nb_full_blocks):
            q = self.orthogonal_matrix_chunk(nb_columns, device=device)
            block_list.append(q)
        remaining_rows = nb_rows - nb_full_blocks * nb_columns
        if remaining_rows > 0:
            q = self.orthogonal_matrix_chunk(nb_columns, device=device)
            block_list.append(q[:remaining_rows])
        final_matrix = torch.cat(block_list)
        if scaling == 0:
            multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
        elif scaling == 1:
            multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
        else:
            raise ValueError(f'Invalid scaling {scaling}')

        return torch.diag(multiplier) @ final_matrix

    def projection_matrix(self, nb_rows=4, nb_columns=None, scaling=0, device=None):
        if nb_columns is None:
            nb_columns = self.dim_heads
        return self.gaussian_orthogonal_random_matrix(nb_rows=nb_rows, nb_columns=nb_columns, scaling=scaling,
                                                      device=device)

    def relu_kernel_transformation(self, x, projection_matrix=None, numerical_stabilizer=0.001, device=None):
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
            ratio = 1.0 / math.sqrt(float(projection_matrix.shape[0]))
            data_dash = ratio * torch.einsum("blhd,md->blhm", x, projection_matrix)
            return torch.relu(data_dash) + numerical_stabilizer

    def orf_kernel_transformation(x, projection_matrix=None, numerical_stabilizer=1e-4):
        """
        Orthogonal Random Features kernel transformation.

        If no projection_matrix is provided:
            Returns a trivial mapping similar to just applying cos to the input.
        If projection_matrix is provided (and is orthogonal [M, D]):
            1. Project x into M dimensions orthogonally.
            2. Apply a cosine transform.

        Args:
            x: [B, L, H, D]
            projection_matrix: [M, D] orthogonal projection matrix
            numerical_stabilizer: small constant added for stability
            device: optional device

        Returns:
            If projection_matrix is None: [B, L, H, D]
            If projection_matrix is provided: [B, L, H, M]
        """
        if projection_matrix is None:
            # Without projection, just return a trivial transform (e.g., cos) to differ from relu
            return torch.cos(x) + numerical_stabilizer
        else:
            B, L, H, D = x.shape
            M = projection_matrix.shape[0]
            ratio = 1.0 / math.sqrt(M)

            # Project: [B, L, H, D] x [M, D] -> [B, L, H, M]
            data_proj = torch.einsum("blhd,md->blhm", x, projection_matrix)

            # Apply cosine to get features
            # [B, L, H, M]
            features = ratio * torch.cos(data_proj)
            return features + numerical_stabilizer

    def porf_kernel_transformation(x, projection_matrix=None, numerical_stabilizer=1e-4):
        """
        Positive Orthogonal Random Features kernel transformation.

        If no projection_matrix is provided:
            Return a positive transform, e.g. exp(-x^2/2).
        If projection_matrix is provided (and is orthogonal [M, D]):
            1. Project x into M dimensions orthogonally.
            2. Compute exp(- (proj^2)/2) to ensure positivity and approximate an RBF kernel.

        Args:
            x: [B, L, H, D]
            projection_matrix: [M, D] orthogonal projection matrix
            numerical_stabilizer: small constant for stability
            device: optional device

        Returns:
            If projection_matrix is None: [B, L, H, D]
            If projection_matrix is provided: [B, L, H, M]
        """
        if projection_matrix is None:
            # Without projection, just ensure positivity:
            # Use an exponential transform to differ from relu or cos:
            # [B, L, H, D]
            return torch.exp(-(x ** 2) / 2) + numerical_stabilizer
        else:
            B, L, H, D = x.shape
            M = projection_matrix.shape[0]
            ratio = 1.0 / math.sqrt(M)

            # Project: [B, L, H, D] x [M, D] -> [B, L, H, M]
            data_proj = torch.einsum("blhd,md->blhm", x, projection_matrix)

            # Compute exp(-proj^2 / 2) for each projected feature, ensuring positivity
            # [B, L, H, M]
            features = ratio * torch.exp(-(data_proj ** 2) / 2)
            return features + numerical_stabilizer

    def softmax_kernel_transformation(self, x, projection_matrix=None, is_query=False, normalize_data=True, eps=1e-4,
                                      numerical_stabilizer=1e-4, device=None):
        """
        Approximates the softmax kernel using random features, similar in style to the
        relu_kernel_transformation. If no projection_matrix is provided, returns a trivial
        exponential-based transform.
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, H, D].
            projection_matrix (torch.Tensor, optional): Random projection matrix of shape [M, D].
            is_query (bool): Whether the input x represents queries (True) or keys (False).
            normalize_data (bool): Whether to apply data normalization as in Performers.
            eps (float): Epsilon for numerical stability in exponentiation.
            numerical_stabilizer (float): Added to final output for numerical stability.
        Returns:
            torch.Tensor: Transformed feature map.
              - If projection_matrix is None: shape is [B, L, H, D].
              - If projection_matrix is provided: shape is [B, L, H, M].
        """
        B, L, H, D = x.shape
        data_normalizer = (D ** -0.25) if normalize_data else 1.0

        if projection_matrix is None:
            # Without a projection matrix, return a trivial exponential transform.
            # Similar to relu_kernel_transformation returning relu(x).
            return torch.exp(data_normalizer * x) + numerical_stabilizer
        else:
            M = projection_matrix.shape[0]
            ratio = (M ** -0.5)  # scaling factor

            # Expand the projection matrix to match [B, H, M, D]
            projection = repeat(projection_matrix, 'j d -> b h j d', b=B, h=H)
            # Project data: [B, L, H, D] x [B, H, M, D] -> [B, L, H, M]
            # Using einsum: 'blhd,bhmd->blhm'
            data_dash = torch.einsum('blhd,bhmd->blhm', data_normalizer * x, projection)

            # Compute diagonal terms for exponent normalization
            diag_data = (x ** 2).sum(dim=-1)  # [B, L, H]
            diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
            diag_data = diag_data.unsqueeze(-1)  # [B, L, H, 1]

            if is_query:
                # For queries, we stabilize along the M dimension
                max_val = torch.amax(data_dash, dim=-1, keepdim=True).detach()  # [B, L, H, 1]
                data_dash = ratio * (torch.exp(data_dash - diag_data - max_val) + eps)
            else:
                # For keys, we stabilize along both L and M dimensions
                max_val = torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()  # [B, 1, 1, 1]
                data_dash = ratio * (torch.exp(data_dash - diag_data - max_val) + eps)

            return data_dash + numerical_stabilizer

    def exponential_kernel_transformation_X(x, projection_matrix=None, numerical_stabilizer=0.001):
        """
        Args:
            x (torch.Tensor): input tensor of shape [B, L, H, D].
            projection_matrix (torch.Tensor, optional): random matrix of shape [M, D].
            numerical_stabilizer (float): small constant for numerical stability.

        Returns:
            torch.Tensor: Transformed feature tensor.
        """
        if projection_matrix is None:
            # Directly apply exp element-wise, then add stabilizer
            # Output shape: [B, L, H, D]
            return torch.exp(x) + numerical_stabilizer
        else:
            # With a projection matrix:
            M = projection_matrix.shape[0]
            offsets = 2 * math.pi * torch.rand(M, device=x.device, dtype=x.dtype)
            norm_sq = (x ** 2).sum(dim=-1, keepdim=False)
            scale_factor = torch.exp(norm_sq / 2)
            projected = torch.einsum("blhd,md->blhm", x, projection_matrix)
            projected = projected + offsets[None, None, None, :]
            cos_features = torch.cos(projected)
            sin_features = torch.sin(projected)
            rff_features = (torch.cat([cos_features, sin_features], dim=-1) / math.sqrt(M))
            scale_factor = scale_factor.unsqueeze(-1)  # [B, L, H, 1]
            data_dash = scale_factor * rff_features  # [B, L, H, 2M]
            # 3. Apply exponential and add a stabilizer
            return data_dash + numerical_stabilizer  # Output shape: [B, L, H, M]

    def exponential_kernel_transformation(self, x, projection_matrix=None, numerical_stabilizer=0.001):
        """
        Args:
            x (torch.Tensor): input tensor of shape [B, L, H, D].
            projection_matrix (torch.Tensor, optional): random matrix of shape [M, D].
            numerical_stabilizer (float): small constant for numerical stability.

        Returns:
            torch.Tensor: Transformed feature tensor.
        """

        if projection_matrix is None:
            return torch.exp(x) + numerical_stabilizer
        else:
            # With a projection matrix:
            M = projection_matrix.shape[0]
            ratio = 1.0 / math.sqrt(M)
            data_dash = ratio * torch.einsum("blhd,md->blhm", x, projection_matrix)
            # Output shape: [B, L, H, M]
            return torch.exp(data_dash) + numerical_stabilizer

    import torch
    import math

    def gaussian_kernel_transformation(self, x, projection_matrix=None, offsets=None, sigma=1.0,
                                       numerical_stabilizer=1e-4):
        """
        Approximates the Gaussian (RBF) kernel using Random Fourier Features.

        If no projection_matrix is given:
            Returns a trivial exp(-x^2/(2sigma^2)) transform [B, L, H, D].
        If projection_matrix is given with shape [M, D]:
            - We sample offsets of shape [M].
            - For each input vector, we compute cos and sin of projections and combine them.
            - Resulting shape: [B, L, H, 2M].

        Args:
            x: input [B, L, H, D].
            projection_matrix: [M, D], random Gaussian directions. If None, no projection is applied.
            offsets: [M], random offsets from Uniform(0, 2*pi). If None, they are generated.
            sigma: Kernel bandwidth parameter.
            numerical_stabilizer: small positive constant.

        Returns:
            If no projection_matrix: [B, L, H, D]
            If projection_matrix is provided: [B, L, H, 2M]
        """
        if projection_matrix is None:
            # No projection, trivial Gaussian transform
            return torch.exp(-(x ** 2).sum(dim=-1, keepdim=True) / (2 * sigma ** 2)) + numerical_stabilizer
        else:
            B, L, H, D = x.shape
            M = projection_matrix.shape[0]

            # Generate offsets if not provided
            if offsets is None:
                offsets = 2 * math.pi * torch.rand(M, device=x.device, dtype=x.dtype)

            # Scale the projection matrix by 1/sigma so we approximate kernel with variance sigma^2
            scaled_projection = projection_matrix / sigma

            # Project: [B, L, H, D] x [M, D] -> [B, L, H, M]
            data_proj = torch.einsum('blhd,md->blhm', x, scaled_projection)

            # Add offsets (broadcast over B,L,H)
            data_proj = data_proj + offsets[None, None, None, :]

            # Compute cosine and sine features
            cos_feats = torch.cos(data_proj)
            sin_feats = torch.sin(data_proj)

            # Concatenate along the last dimension -> [B, L, H, 2M]
            features = torch.cat([cos_feats, sin_feats], dim=-1)

            # Normalize by sqrt(M) to ensure features have unit variance
            features = features / math.sqrt(M)

            return features + numerical_stabilizer

    def noncausal_numerator(self, qs, ks, vs):
        """Computes not-normalized FAVOR noncausal attention AV.

        Args:
            qs: query_prime tensor of the shape [L,B,H,M].
            ks: key_prime tensor of the shape [L,B,H,M].
            vs: value tensor of the shape [L,B,H,D].

        Returns:
            Not-normalized FAVOR noncausal attention AV.
        """
        # print(ks.shape, vs.shape)
        kvs = torch.einsum("lbhm,lbhd->bhmd", ks, vs)
        return torch.einsum("lbhm,bhmd->lbhd", qs, kvs)

    def noncausal_denominator(self, qs, ks):
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

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,L,M]

        q = q.permute(0, 2, 1, 3)  # [B,L,H,M]
        k = k.permute(0, 2, 1, 3)  # [B,L,H,M]
        v = v.permute(0, 2, 1, 3)  # [B,L,H,M]

        # self.nb_features = 4
        # projection_matrix = self.projection_matrix(nb_rows=self.nb_features, nb_columns=self.dim_heads, scaling=0,
        #                                            device=x.device)

        create_kernel = partial(self.relu_kernel_transformation, projection_matrix=None, device=x.device)
        # kernel transformation
        query_prime = create_kernel(q)  # [B,L,H,M]
        key_prime = create_kernel(k)  # [B,L,H,M]
        
        # query_prime = self.softmax_kernel_transformation(q,is_query=True)
        # key_prime = self.softmax_kernel_transformation(k,is_query=False)
        # query_prime = self.exponential_kernel_transformation(q, projection_matrix=None, numerical_stabilizer=0.001)
        # key_prime = self.exponential_kernel_transformation(k, projection_matrix=None, numerical_stabilizer=0.001)

        # query_prime = self.gaussian_kernel_transformation(q)
        # key_prime = self.gaussian_kernel_transformation(k)

        query_prime = query_prime.permute(1, 0, 2, 3)  # [L,B,H,M]
        key_prime = key_prime.permute(1, 0, 2, 3)  # [L,B,H,M]
        value = v.permute(1, 0, 2, 3)  # [L,B,H,M]

        # non-causal nominator and denominator
        av_attention = self.noncausal_numerator(query_prime, key_prime, value)  # [L,B,H,M]
        attention_normalizer = self.noncausal_denominator(query_prime, key_prime)  # [L,B,H]

        av_attention = av_attention.permute(1, 0, 2, 3)  # [B,L,H,M]
        attention_normalizer = attention_normalizer.permute(1, 0, 2)  # [B,L,H]
        attention_normalizer = torch.unsqueeze(attention_normalizer,
                                               len(attention_normalizer.shape))  # [B,L,H,1]

        x = av_attention / attention_normalizer  # [B,L,H,M]
        x = x.reshape(B, N, C)  # [B,L,H*M]

        # q = q * self.scale

        # attn = (q @ k.transpose(-2, -1))  # [B,H,L,M] @ [B,H,M,L] = [B,H,L,L]
        # attn = attn.softmax(dim=-1)  # [B,H,L,L]
        # attn = self.attn_drop(attn)  # [B,H,L,L]

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp
                 , init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp
                 , init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp
                 , init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_1_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x))) + self.drop_path(
            self.gamma_1_1 * self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + self.drop_path(
            self.gamma_2_1 * self.mlp1(self.norm21(x)))
        return x


class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp
                 , init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.mlp1(self.norm21(x)))
        return x


class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=nn.SyncBatchNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim // 4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim // 4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim // 4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                          ])

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers=Block,
                 Patch_layer=PatchEmbed, act_layer=nn.GELU,
                 Attention_block=Attention, Mlp_block=Mlp,
                 dpr_constant=True, init_scale=1e-4,
                 mlp_ratio_clstk=4.0, **kwargs):
        super().__init__()

        self.dropout_rate = drop_rate

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
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

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features(x)

        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)

        return x

@register_model
def performer_small_patch8_LS(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def performer_base_patch8_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    return model