"""UOViT-E optimisation components: lightweight CNN feature encoding,
multi-scale feature fusion, and token reduction.

These are written as standalone building blocks (no dependency on each
other) so `swin_transformer_uovit.py` can compose them independently per
ablation config. Each one is designed to be a shape-preserving addition
to the baseline Swin-Unet: nothing here changes the token count or
resolution the rest of the network expects, so PatchMerging, PatchExpand,
window_reverse and the skip-connection concatenation in
SwinTransformerSys.forward_up_features all keep working unmodified.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer_unet_skip_expand_decoder_sys import (
    SwinTransformerBlock,
    window_partition,
    window_reverse,
)


# ---------------------------------------------------------------------------
# 1. Lightweight CNN feature encoding ("Hybrid")
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class LightweightCNNStem(nn.Module):
    """Extracts local feature maps at the same 4 resolutions as the Swin
    encoder stages (H/4, H/8, H/16, H/32) directly from the input image, so
    they can be fused into the corresponding skip connections. Depthwise-
    separable convs keep it cheap relative to the Transformer branch.
    """

    def __init__(self, in_chans, stage_dims, stem_channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
        )  # input -> H/4, W/4
        self.stage0_proj = nn.Conv2d(stem_channels, stage_dims[0], 1, bias=False)
        self.down_blocks = nn.ModuleList([
            DepthwiseSeparableConv(stage_dims[i], stage_dims[i + 1], stride=2)
            for i in range(len(stage_dims) - 1)
        ])

    def forward(self, x):
        x = self.stem(x)
        feats = [self.stage0_proj(x)]
        for blk in self.down_blocks:
            feats.append(blk(feats[-1]))
        return feats  # [stage0 (H/4), stage1 (H/8), stage2 (H/16), stage3 (H/32)]


def tokens_to_spatial(x, H, W):
    B, L, C = x.shape
    return x.transpose(1, 2).reshape(B, C, H, W)


def spatial_to_tokens(x):
    B, C, H, W = x.shape
    return x.flatten(2).transpose(1, 2)


class SkipFusionGate(nn.Module):
    """Fuses a Swin skip-connection tensor with a CNN local-feature map at
    the same scale: concat -> 1x1 conv -> sigmoid-gated residual add. The
    gate is initialised close to 0 (sigmoid(-5) ~= 0.007) so training starts
    close to the unmodified baseline and the network learns how much to
    trust the CNN branch, rather than being forced to use it from step one.
    """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(dim * 2, dim, 1, bias=True)
        self.gate = nn.Parameter(torch.tensor(-5.0))

    def forward(self, swin_tokens, cnn_feat, H, W):
        swin_spatial = tokens_to_spatial(swin_tokens, H, W)
        fused = self.proj(torch.cat([swin_spatial, cnn_feat], dim=1))
        out = swin_spatial + torch.sigmoid(self.gate) * fused
        return spatial_to_tokens(out)


# ---------------------------------------------------------------------------
# 2. Multi-scale feature fusion ("Fusion")
# ---------------------------------------------------------------------------

class MultiScaleFusion(nn.Module):
    """Top-down (FPN-style) cross-scale fusion across the 4 CNN-branch
    feature maps, so each scale carries both its own local detail and
    context propagated down from coarser scales -- combining low-level
    boundary information with high-level semantic information, as called
    for by the proposal's multi-scale fusion component. Operates purely on
    the CNN branch's feature pyramid (stage_dims channels differ per
    scale, so fusion happens in a shared `fusion_dim` space and is
    projected back out); output shapes match the input feature list
    exactly via a residual connection, so it's a drop-in enhancement
    wherever the raw CNN stem features would otherwise be used.
    """

    def __init__(self, stage_dims, fusion_dim=None, num_out_scales=None):
        super().__init__()
        fusion_dim = fusion_dim or min(stage_dims)
        # every input scale needs a lateral (even one that's only a
        # top-down seed and not itself returned), but smooth/out_proj are
        # only built for scales the caller actually consumes downstream --
        # otherwise they'd be dead weight inflating params/GFLOPs for
        # nothing, which matters here since those are exactly the numbers
        # this study reports.
        self.num_out_scales = num_out_scales if num_out_scales is not None else len(stage_dims)
        self.in_proj = nn.ModuleList([nn.Conv2d(d, fusion_dim, 1) for d in stage_dims])
        self.smooth = nn.ModuleList([nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1)
                                      for _ in range(self.num_out_scales)])
        self.out_proj = nn.ModuleList([nn.Conv2d(fusion_dim, d, 1) for d in stage_dims[:self.num_out_scales]])

    def forward(self, feats):
        # feats: list ordered fine -> coarse (stage0 .. stage3)
        laterals = [proj(f) for proj, f in zip(self.in_proj, feats)]
        fused = [None] * len(laterals)
        fused[-1] = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(fused[i + 1], size=laterals[i].shape[-2:], mode="nearest")
            fused[i] = laterals[i] + up
        out = []
        for i in range(self.num_out_scales):
            s = self.smooth[i](fused[i])
            out.append(self.out_proj[i](s) + feats[i])
        return out


# ---------------------------------------------------------------------------
# 3. Token reduction ("Token")
# ---------------------------------------------------------------------------

def bipartite_soft_matching(metric, r):
    """Token Merging (ToMe) bipartite soft matching, following Bolya et al.,
    "Token Merging: Your ViT But Faster" (ICLR 2023). Splits tokens into two
    interleaved sets, matches each token in the smaller set to its most
    similar token in the other by cosine similarity, and merges the `r`
    highest-similarity pairs by averaging.

    Args:
        metric: (B, N, C) features used to compute similarity (usually the
            block's normalised input -- redundant/near-duplicate tokens
            have highly similar features here).
        r: number of token pairs to merge (reduces N by r).

    Returns:
        (merge, unmerge) callables. merge(x) -> (B, N - r, C).
        unmerge(y) with y of shape (B, N - r, C) -> (B, N, C), restoring
        every original position (merged positions receive the shared
        merged value).
    """
    B, N, _ = metric.shape
    r = max(0, min(r, N // 2 - 1)) if N // 2 > 1 else 0
    if r <= 0:
        return (lambda x: x), (lambda x: x)

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # unmerged tokens from set 'a'
        src_idx = edge_idx[..., :r, :]  # tokens from set 'a' being merged into 'b'
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    a_len = a.shape[-2]
    unm_len = a_len - r

    def merge(x, mode="mean"):
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, _, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, unm_len, c))
        src_m = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src_m, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x):
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros((n, N, c), device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return out

    return merge, unmerge


class TokenReducingSwinBlock(SwinTransformerBlock):
    """SwinTransformerBlock with ToMe-style token merging applied around
    the MLP sublayer only.

    Token reduction is deliberately NOT applied inside window attention:
    shifted-window blocks (SW-MSA) use a precomputed attention mask sized
    for the full window token count, and merging tokens before attention
    would require re-deriving that mask per merge on every forward pass --
    a real correctness risk that isn't worth it for the FLOPs the (small,
    window_size**2-token) attention op contributes. The MLP has no such
    constraint: it's a per-token pointwise operation, and with
    `mlp_ratio=4` it is typically the larger share of a block's compute.
    Merging is applied there instead, leaving attention and every
    downstream shape-sensitive operation (window_reverse, PatchMerging,
    skip connections) byte-for-byte identical to the baseline.

    Because merge/unmerge happens entirely inside the MLP branch, this
    block's output token count always equals its input token count -- it
    is a drop-in replacement for SwinTransformerBlock anywhere in the
    network, including inside BasicLayer/BasicLayer_up unchanged.
    """

    def __init__(self, *args, keep_ratio=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0. < keep_ratio <= 1.0, "keep_ratio must be in (0, 1]"
        self.keep_ratio = keep_ratio

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)

        y = self.norm2(x)
        r = L - max(1, int(round(L * self.keep_ratio)))
        merge, unmerge = bipartite_soft_matching(y, r)
        y = unmerge(self.mlp(merge(y)))
        x = x + self.drop_path(y)

        return x


class TokenReducingBasicLayer(nn.Module):
    """Mirrors Swin-Unet's BasicLayer (encoder stage container), but builds
    TokenReducingSwinBlock instead of SwinTransformerBlock. Duplicated
    rather than subclassed from BasicLayer because BasicLayer constructs
    its block list directly inside __init__ with no factory seam to hook
    into; this keeps every other hyperparameter (dims, resolution, heads,
    window size, drop path schedule, downsample) identical to the
    baseline's construction.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, keep_ratio=0.75):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            TokenReducingSwinBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, keep_ratio=keep_ratio)
            for i in range(depth)
        ])

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) \
            if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                import torch.utils.checkpoint as checkpoint
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
