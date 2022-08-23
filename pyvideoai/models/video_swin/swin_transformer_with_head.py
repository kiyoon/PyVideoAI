from torch import nn
from .i3d_head import I3DHead
from .swin_transformer import SwinTransformer3D

class SwinTransformer3DWithHead(nn.Module):
    def __init__(self,
                 num_classes,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(4,4,4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False,
                 # Head
                 head_in_channels = 768,
                 head_spatial_type = 'avg',
                 head_dropout_ratio = 0.5,
                 ):
        super().__init__()
        self.backbone = SwinTransformer3D(
                                            pretrained=pretrained,
                                            pretrained2d=pretrained2d,
                                            patch_size=patch_size,
                                            in_chans=in_chans,
                                            embed_dim=embed_dim,
                                            depths=depths,
                                            num_heads=num_heads,
                                            window_size=window_size,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            drop_rate=drop_rate,
                                            attn_drop_rate=attn_drop_rate,
                                            drop_path_rate=drop_path_rate,
                                            norm_layer=norm_layer,
                                            patch_norm=patch_norm,
                                            frozen_stages=frozen_stages,
                                            use_checkpoint=use_checkpoint,
                                          )

        self.cls_head = I3DHead(
                            num_classes = num_classes,
                            in_channels = head_in_channels,
                            spatial_type = head_spatial_type,
                            dropout_ratio = head_dropout_ratio,
                           )


    def features(self, x):
        return self.backbone(x)


    def logits(self, features):
        return self.cls_head(features)


    def forward(self, x):
        return self.logits(self.features(x))
