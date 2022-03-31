from typing import Sequence
from mmcv.utils import Registry
import torch.optim.lr_scheduler as scheduler

SCHEDULERS = Registry('SCHEDULERS')


def build_scheduler(cfg, optimizer):
    """Build scheduler."""
    return SCHEDULERS.build(cfg, default_args=dict(optimizer=optimizer))


@SCHEDULERS.register_module()
class MultiStepLRWarmUp(scheduler.LambdaLR):
    def __init__(self,
                 optimizer,
                 warmup_epochs,
                 decay_steps=None,
                 decay_ratio=None,
                 max_epoch=-1,
                 verbose=False):
        assert max_epoch > 0

        def lr_lambda(epoch):  # start from 0
            if epoch <= warmup_epochs - 1:
                factor = float(epoch + 1) / float(warmup_epochs + 1)
            else:
                if isinstance(decay_steps, Sequence) and decay_ratio > 0.:
                    factor = 1.
                    for step in decay_steps:
                        if epoch + 1 < step:
                            break
                        factor *= decay_ratio
                elif decay_steps is None and type(decay_steps) == type(decay_ratio):
                    linear_decay_epochs = max_epoch - warmup_epochs
                    factor = (linear_decay_epochs -
                              (epoch - warmup_epochs)) / linear_decay_epochs
            return factor

        super(MultiStepLRWarmUp, self).__init__(
            optimizer,
            lr_lambda=lambda epoch: lr_lambda(epoch),
            verbose=verbose
        )


@SCHEDULERS.register_module()
class CosineAnnealingLR(scheduler.CosineAnnealingLR):
    def __init__(self,
                 optimizer,
                 T_max,
                 max_epoch=-1,
                 eta_min=0,
                 verbose=False):
        super(CosineAnnealingLR, self).__init__(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
            verbose=verbose
        )


@SCHEDULERS.register_module()
class CosineAnnealingLRWarmRestarts(scheduler.CosineAnnealingWarmRestarts):
    def __init__(self,
                 optimizer,
                 T_0,
                 max_epoch=-1,
                 T_mult=1,
                 eta_min=0,
                 verbose=False):
        super(CosineAnnealingLRWarmRestarts, self).__init__(
            optimizer,
            T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            verbose=verbose
        )
