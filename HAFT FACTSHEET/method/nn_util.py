import warnings
from typing import Tuple, Optional
import math
import torch
from timm.layers import to_2tuple
from torch import nn, Tensor, cat, sqrt, ones, zeros
from torch.autograd import Function

from torch.nn.init import trunc_normal_


class IdentityMod(nn.Module):
    def __init__(self):
        super(IdentityMod, self).__init__()

    def forward(self, x=None, *args, **kwargs) -> Tensor:
        return x

class ConcatTensors(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return cat((x, y), dim=1)

class SkipConnection(nn.Module):
    def __init__(self,):
        super(SkipConnection, self).__init__()

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return x + skip

class ApplyVectorWeights(nn.Module):
    def __init__(self):
        super(ApplyVectorWeights, self).__init__()

    def forward(self, x: Tensor, weights: Tensor) -> Tensor:
        return x * weights

class ChannelEmbeddingCompression(nn.Module):
    def __init__(self, embed_dim, embed_dim_next):
        super().__init__()
        self.patch_unembed = PatchUnEmbedIR(embed_dim=embed_dim)
        self.conv = nn.Conv2d(embed_dim, embed_dim_next, 1, 1, 0)
        self.patch_embed = PatchEmbedIR(embed_dim=embed_dim_next)

    def forward(self, x, x_size):
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x)
        return x

class InvertedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=bias)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, stride=1, groups=out_channels, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DWConv2d(nn.Module):

    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """
        x: (b h w c)
        """
        x = x.permute(0, 3, 1, 2)  # (b c h w)
        x = self.conv(x)  # (b c h w)
        x = x.permute(0, 2, 3, 1)  # (b h w c)
        return x

class ChannelAttention(nn.Module):
    """
    Channel attention module with optional attention weights.
    """
    def __init__(self, num_channel: int):
        """

        :param num_channel: Number of channels in the input tensor
        :param apply_att_weights: Should attention weights be applied in the forward pass?
        """
        super(ChannelAttention, self).__init__()

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=num_channel, out_channels=num_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_channel // 2, out_channels=num_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, num_channel, apply_att_weights=False):
        """

        :param num_channel: Number of channels in the input tensor
        :param apply_att_weights: Should attention weights be applied in the forward pass?
        """
        super(SimplifiedChannelAttention, self).__init__()

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.apply1 = ApplyVectorWeights() if apply_att_weights else IdentityMod()

    def forward(self, x: Tensor, att_weights: Tensor = None) -> Tensor:
        x = self.model(x)
        return self.apply1(x=x, weights=att_weights)

class ApertureAwareAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads # key dim is channel size of the key matrix for each head
        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)
        self.lepe = DWConv2d(embed_dim, 5, 1, 2) # LEPE is Local Context Enhancement in the paper

        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x: torch.Tensor, rel_pos):
        """Force float32 for attention to prevent FP16 overflow in softmax."""
        x = x.float()  # Cast input to float32
        bsz, h, w, _ = x.size()

        mask_h, mask_w = rel_pos
        mask_h = mask_h.float()
        mask_w = mask_w.float()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        lepe = self.lepe(v)

        k *= self.scaling
        qr = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        kr = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)

        qr_w = qr.transpose(1, 2) # (bsz, h, heads, w, dim)
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

        v_out_w = []
        chunk_size_h = 32
        mask_w_expand = mask_w.unsqueeze(1) # Broadcastable to (bsz, chunk, heads, w, w)
        for i in range(0, h, chunk_size_h):
            qk_mat_w = qr_w[:, i:i+chunk_size_h] @ kr_w[:, i:i+chunk_size_h].transpose(-1, -2)
            qk_mat_w = qk_mat_w + mask_w_expand
            qk_mat_w = torch.softmax(qk_mat_w, -1)
            v_chunk = qk_mat_w @ v[:, i:i+chunk_size_h]
            v_out_w.append(v_chunk)
            
        v = torch.cat(v_out_w, dim=1)

        qr_h = qr.permute(0, 3, 1, 2, 4) # (bsz, w, h, heads, dim)
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4) # (bsz, w, heads, h, val_dim)

        v_out_h = []
        chunk_size_w = 32
        mask_h_expand = mask_h.unsqueeze(1) # Broadcastable to (bsz, chunk, heads, h, h)
        for i in range(0, w, chunk_size_w):
            qk_mat_h = qr_h[:, i:i+chunk_size_w] @ kr_h[:, i:i+chunk_size_w].transpose(-1, -2)
            qk_mat_h = qk_mat_h + mask_h_expand
            qk_mat_h = torch.softmax(qk_mat_h, -1)
            v_chunk = qk_mat_h @ v[:, i:i+chunk_size_w]
            v_out_h.append(v_chunk)

        output = torch.cat(v_out_h, dim=1)

        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
        output = output + lepe
        output = self.out_proj(output)
        return output

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNorm2d(nn.GroupNorm):
    def __init__(self, channels, eps=1e-4):  # 1e-4 is safe for FP16 (1e-6 flushes to 0)
        # GroupNorm(1, channels) is exactly equivalent to LayerNorm over the channel dimension for 2D inputs,
        # but uses PyTorch's native, highly optimized, memory-efficient, and FP16-safe C++ implementation.
        super().__init__(1, channels, eps=eps)


class DynRelPos2d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        """
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        """
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        self.register_buffer('angle', angle)

    @torch.amp.autocast('cuda', enabled=False)
    def generate_1d_decay(self, l: int, range_factor: Tensor):
        """
        generate 1d decay mask, the result is l*l
        Force float32 to prevent FP16 overflow in log/exp.
        """
        range_factor = abs(range_factor).float()  # Cast to float32

        bs = range_factor.size(0)
        ###print(f"Generating 1D decay mask for batch size: {bs} and length: {l}")

        heads_ranges = self.heads_range * torch.arange(self.num_heads, dtype=torch.float) / self.num_heads
        heads_ranges = heads_ranges.to(range_factor.device)
        ###print("Heads Ranges:")
        ###print(heads_ranges)
        ###print(f"Heads Ranges shape: {heads_ranges.shape}")

        range_factor = torch.sqrt(torch.sqrt(range_factor))  # (n) # give extra weight to smaller values
        range_factor = range_factor[:, None]  # (n 1)
        ###print("Range Factor:")
        ###print(range_factor)
        ###print(f"Range Factor shape: {range_factor.shape}")

        ranges = (-self.initial_value - heads_ranges.repeat(bs, 1) * range_factor)
        ###print("Ranges:")
        decay = torch.log(1 - 2 ** ranges)  # (b n)
        ###print("Decay:")
        ###print(decay)
        ###print(f"Decay shape: {decay.shape}")

        index = torch.arange(l).to(decay)
        ###print("Index:")
        ###print(index)
        ###print(f"Index shape: {index.shape}")
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        # extend mask to batch size with one channel for each
        mask = mask[None, None, :, :]  # (1 1 l l)
        ###print("Mask before decay application:")
        ###print(mask)
        ###print(f"Mask shape : {mask.shape}")

        mask = mask * decay[:, :, None, None]  # (b n l l)
        ###print("Mask after decay application:")
        ###print(mask)
        ###print(f"Mask shape: {mask.shape}")

        return mask

    def forward(self, slen: Tuple[int], range_factor: Tensor):
        mask_h = self.generate_1d_decay(slen[0], range_factor=range_factor)  # x axis decay
        mask_w = self.generate_1d_decay(slen[1], range_factor=range_factor)  # y axis decay
        return mask_h, mask_w

class PatchEmbedIR(nn.Module):
    r""" Image to Patch Embedding

    Args:
        embed_dim (int): Number of linear projection output channels.
        norm_layer (nn.Module, optional): Normalization layer.
    """

    def __init__(self, embed_dim=96, norm_layer=None):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (b c h w) -> (b h w c)
        if self.norm is not None:
            # print("Using norm layer")
            x = self.norm(x)
        return x


class PatchUnEmbedIR(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        embed_dim (int): Number of linear projection output channels.
    """

    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = x.view(B, self.embed_dim, x_size[0], x_size[1]) # B Ph*Pw C
        return x

class ApertureEncoder(nn.Module):
    def __init__(self, d_embed=64, num_freqs=8):
        super().__init__()
        self.num_freqs = num_freqs
        
        # We need an even number of channels for the MLP input (sin + cos for each freq)
        # Input channel size = 2 * num_freqs
        in_dim = 2 * num_freqs
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_embed // 2),
            nn.GELU(),
            nn.Linear(d_embed // 2, d_embed)
        )
        
        # Log-linear frequencies from 2^0 to 2^(num_freqs-1)
        freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x):
        # x is (B, 1) f-stop values
        # Add frequency dimension: (B, num_freqs)
        x_freq = x * self.freq_bands[None, :] * math.pi
        
        # Compute sin and cos components: (B, 2 * num_freqs)
        fourier_features = torch.cat([torch.sin(x_freq), torch.cos(x_freq)], dim=-1)
        
        return self.mlp(fourier_features)

class FiLMLayer(nn.Module):
    def __init__(self, in_channels, d_embed=64):
        super().__init__()
        self.fc = nn.Linear(d_embed, in_channels * 2)
        # Zero-init so FiLM starts as identity (scale=0, shift=0)
        # This prevents destabilizing pretrained backbone features
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    def forward(self, x, condition):
        scale_shift = self.fc(condition) 
        scale, shift = scale_shift.chunk(2, dim=1) 
        if x.dim() == 4:
            if x.shape[1] == scale.shape[1]: 
                scale = scale.view(-1, scale.shape[1], 1, 1)
                shift = shift.view(-1, shift.shape[1], 1, 1)
            else: 
                scale = scale.view(-1, 1, 1, scale.shape[1])
                shift = shift.view(-1, 1, 1, shift.shape[1])
        return x * (1 + scale) + shift

class FocalPriorGenerator(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, depth, mask):
        focal_dist = (depth * mask).sum(dim=(2, 3), keepdim=True) / (mask.sum(dim=(2, 3), keepdim=True) + 1e-4)
        focal_map = torch.abs(depth - focal_dist)
        soft_mask = torch.exp(-focal_map)
        return focal_map, soft_mask

class FusionStem(nn.Module):
    def __init__(self, in_channels=7, d_embed=64): 
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, d_embed, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_embed, d_embed, kernel_size=3, padding=1),
        )
    def forward(self, rgb, depth, mask, focal_map, soft_mask=None):
        if soft_mask is not None:
            x = torch.cat([rgb, depth, mask, focal_map, soft_mask], dim=1)
        else:
            x = torch.cat([rgb, depth, mask, focal_map], dim=1)
        return self.proj(x)