"""UOViT-E: SwinTransformerSys extended with three independently toggleable
optimisation components, covering every ablation configuration in the
study design from one model class:

    Baseline       -> all three flags False (identical to SwinTransformerSys)
    Hybrid         -> use_cnn_hybrid=True
    Fusion         -> use_cnn_hybrid=True, use_multiscale_fusion=True
    Token          -> use_token_reduction=True
    Hybrid + Token -> use_cnn_hybrid=True, use_token_reduction=True
    UOViT-E (full) -> all three True

Keeping this as one class with flags (rather than 6 separate model files)
guarantees every ablation config shares identical baseline hyperparameters
and weight initialisation, which is what makes the ablation comparison
meaningful.
"""

import torch
import torch.nn as nn

from .swin_transformer_unet_skip_expand_decoder_sys import (
    SwinTransformerSys,
    PatchMerging,
)
from .hybrid_modules import (
    LightweightCNNStem,
    SkipFusionGate,
    MultiScaleFusion,
    TokenReducingBasicLayer,
)


class SwinTransformerSysUOViT(SwinTransformerSys):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first",
                 use_cnn_hybrid=False, use_multiscale_fusion=False,
                 use_token_reduction=False, token_keep_ratio=0.75,
                 cnn_stem_channels=32, **kwargs):

        if use_multiscale_fusion and not use_cnn_hybrid:
            raise ValueError("use_multiscale_fusion=True requires use_cnn_hybrid=True "
                              "(fusion operates on the CNN branch's feature pyramid)")

        super().__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
            embed_dim=embed_dim, depths=depths, depths_decoder=depths_decoder, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
            use_checkpoint=use_checkpoint, final_upsample=final_upsample, **kwargs)

        self.use_cnn_hybrid = use_cnn_hybrid
        self.use_multiscale_fusion = use_multiscale_fusion
        self.use_token_reduction = use_token_reduction

        stage_dims = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        if use_cnn_hybrid:
            # SwinTransformerSys.forward_up_features never reads
            # x_downsample[-1] (the bottleneck-level skip; the bottleneck
            # feature itself is the decoder's starting point, not a skip
            # target) -- only 3 of the 4 stored skips are ever consumed, so
            # skip_fusion only needs 3 gates. The CNN stem only needs to
            # produce a 4th (deepest) scale when use_multiscale_fusion is
            # on, since that's the only place it's used (as the top-down
            # pass's context seed) -- otherwise building/running it would
            # be dead compute inflating params/GFLOPs for nothing.
            num_used_skips = self.num_layers - 1
            num_cnn_scales = self.num_layers if use_multiscale_fusion else num_used_skips
            self.cnn_stem = LightweightCNNStem(in_chans, stage_dims[:num_cnn_scales],
                                                stem_channels=cnn_stem_channels)
            self.skip_fusion = nn.ModuleList([SkipFusionGate(d) for d in stage_dims[:num_used_skips]])

        if use_multiscale_fusion:
            self.multiscale_fusion = MultiScaleFusion(stage_dims, num_out_scales=self.num_layers - 1)

        if use_token_reduction:
            # Rebuild the encoder stages with TokenReducingBasicLayer.
            # Mirrors SwinTransformerSys.__init__'s encoder-building loop
            # exactly (same dims/resolutions/depths/heads/window
            # size/drop-path schedule) so the only difference from the
            # baseline is the block class used inside each stage.
            patches_resolution = self.patches_resolution
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                layer = TokenReducingBasicLayer(
                    dim=int(embed_dim * 2 ** i_layer),
                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                       patches_resolution[1] // (2 ** i_layer)),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    keep_ratio=token_keep_ratio,
                )
                self.layers.append(layer)
            self.apply(self._init_weights)

    def _fuse_skips(self, x_downsample, cnn_feats):
        # only the first (num_layers - 1) skips are ever read by
        # forward_up_features -- see the comment in __init__.
        fused = []
        for i in range(self.num_layers - 1):
            H = self.patches_resolution[0] // (2 ** i)
            W = self.patches_resolution[1] // (2 ** i)
            fused.append(self.skip_fusion[i](x_downsample[i], cnn_feats[i], H, W))
        fused.append(x_downsample[-1])
        return fused

    def forward(self, x):
        raw_input = x
        feats, x_downsample = self.forward_features(x)

        if self.use_cnn_hybrid:
            cnn_feats = self.cnn_stem(raw_input)
            if self.use_multiscale_fusion:
                cnn_feats = self.multiscale_fusion(cnn_feats)
            x_downsample = self._fuse_skips(x_downsample, cnn_feats)

        x = self.forward_up_features(feats, x_downsample)
        x = self.up_x4(x)
        return x
