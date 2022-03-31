import torch
from mmcv.utils import Registry

OPTIMIZERS = Registry('OPTIMIZERS')


def build_optimizer(cfg, params):
    """Build optimizer."""
    return OPTIMIZERS.build(cfg, default_args=dict(params=params))


@OPTIMIZERS.register_module()
class SGD(torch.optim.SGD):
    def __init__(self,
                 params,
                 lr,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False):
        super(SGD, self).__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov
        )


@OPTIMIZERS.register_module()
class RMSProp(torch.optim.RMSprop):
    def __init__(self,
                 params,
                 lr=1e-2,
                 alpha=0.99,
                 eps=1e-8,
                 weight_decay=0,
                 momentum=0,
                 centered=False):
        super(RMSProp, self).__init__(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered
        )


@OPTIMIZERS.register_module()
class Adam(torch.optim.Adam):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False):
        super(Adam, self).__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )


@OPTIMIZERS.register_module()
class AdamW(torch.optim.AdamW):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=1e-2,
                 amsgrad=False):
        super(AdamW, self).__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
