"""SwinUnet-style wrapper around SwinTransformerSysUOViT, plus a plain-
kwargs factory for use outside the yacs config system (benchmarking,
unit tests, ad hoc scripts).
"""

import torch.nn as nn

from .vision_transformer import SwinUnet
from .swin_transformer_uovit import SwinTransformerSysUOViT


class VisionTransformerUOViT(SwinUnet):
    """Reuses SwinUnet's forward() (single-channel -> 3-channel repeat) and
    load_from() (ImageNet-pretrained Swin encoder loading) unchanged: the
    new CNN hybrid/fusion/token-reduction modules don't touch any
    parameter name the pretrained checkpoint maps onto (TokenReducingSwinBlock
    only overrides forward(), it doesn't add or rename parameters relative
    to SwinTransformerBlock), so pretrained loading still works as-is.
    Only __init__ is overridden, to build SwinTransformerSysUOViT instead
    of the baseline SwinTransformerSys.
    """

    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False,
                 use_cnn_hybrid=False, use_multiscale_fusion=False,
                 use_token_reduction=False, token_keep_ratio=0.75, cnn_stem_channels=32):
        nn.Module.__init__(self)
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSysUOViT(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=self.num_classes,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_cnn_hybrid=use_cnn_hybrid,
            use_multiscale_fusion=use_multiscale_fusion,
            use_token_reduction=use_token_reduction,
            token_keep_ratio=token_keep_ratio,
            cnn_stem_channels=cnn_stem_channels,
        )


# Ablation configs from the proposal's Section "Ablation Study", as flag
# combinations. Reuse these keys everywhere (benchmarking, training loop,
# results tables) so config naming stays consistent across the study.
ABLATION_CONFIGS = {
    "baseline": dict(use_cnn_hybrid=False, use_multiscale_fusion=False, use_token_reduction=False),
    "hybrid": dict(use_cnn_hybrid=True, use_multiscale_fusion=False, use_token_reduction=False),
    "fusion": dict(use_cnn_hybrid=True, use_multiscale_fusion=True, use_token_reduction=False),
    "token": dict(use_cnn_hybrid=False, use_multiscale_fusion=False, use_token_reduction=True),
    "hybrid_token": dict(use_cnn_hybrid=True, use_multiscale_fusion=False, use_token_reduction=True),
    "uovit_e": dict(use_cnn_hybrid=True, use_multiscale_fusion=True, use_token_reduction=True),
}


def build_uovit_e(img_size=224, in_chans=3, num_classes=9, patch_size=4,
                   embed_dim=96, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                   window_size=7, mlp_ratio=4.0, drop_rate=0.0, drop_path_rate=0.1,
                   token_keep_ratio=0.75, cnn_stem_channels=32,
                   config_name=None, **flags):
    """Builds SwinTransformerSysUOViT directly from explicit kwargs, without
    requiring the yacs config object SwinUnet/VisionTransformerUOViT expect.
    Handy for benchmarking ablation configs and for unit tests.

    Pass either `config_name` (one of ABLATION_CONFIGS' keys) or explicit
    `use_cnn_hybrid=`/`use_multiscale_fusion=`/`use_token_reduction=` flags.
    """
    if config_name is not None:
        if flags:
            raise ValueError("pass either config_name or explicit use_* flags, not both")
        flags = ABLATION_CONFIGS[config_name]

    return SwinTransformerSysUOViT(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans,
        num_classes=num_classes, embed_dim=embed_dim, depths=list(depths),
        num_heads=list(num_heads), window_size=window_size, mlp_ratio=mlp_ratio,
        drop_rate=drop_rate, drop_path_rate=drop_path_rate,
        token_keep_ratio=token_keep_ratio, cnn_stem_channels=cnn_stem_channels,
        **flags,
    )
