from .swinir_basics import *

class SwinIR_model(nn.Module):
    def __init__(self, config):
        super(SwinIR_model, self).__init__()
        self.config = config
        self.swin_unet = SwinIR(img_size=config.MODEL.IMG_SIZE,
                               patch_size=config.MODEL.PATCH_SIZE,
                               in_chans=1,
                               out_chans=1,
                               embed_dim=config.MODEL.EMB_DIM,
                               depths=config.MODEL.DEPTH_EN,
                               num_heads=config.MODEL.HEAD_NUM,
                               window_size=config.MODEL.WIN_SIZE,
                               mlp_ratio=config.MODEL.MLP_RATIO,
                               qkv_bias=config.MODEL.QKV_BIAS,
                               qk_scale=config.MODEL.QK_SCALE,
                               drop_rate=config.MODEL.DROP_RATE,
                               drop_path_rate=config.MODEL.DROP_PATH_RATE,
                               ape=config.MODEL.APE,
                               patch_norm=config.MODEL.PATCH_NORM,
                               use_checkpoint=config.MODEL.USE_CHECKPOINTS)

    def forward(self, x):
        # if x.size()[1] == 1:
        #     x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits