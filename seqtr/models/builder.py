from mmcv.utils import Registry


VIS_ENCODERS = Registry('VIS_ENCS')
LAN_ENCODERS = Registry('LAN_ENCS')
MODELS = Registry('MODELS')
FUSIONS = Registry('FUSIONS')
HEADS = Registry('HEADS')


def build_vis_enc(cfg):
    """Build vis_enc."""
    return VIS_ENCODERS.build(cfg)


def build_lan_enc(cfg, default_args):
    """Build lan_enc."""
    return LAN_ENCODERS.build(cfg, default_args=default_args)


def build_fusion(cfg):
    """Build lad_conv_list."""
    return FUSIONS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_model(cfg, word_emb=None, num_token=-1):
    """Build model."""
    model = MODELS.build(cfg, default_args=dict(
        word_emb=word_emb, num_token=num_token))

    return model
