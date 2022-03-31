from abc import ABCMeta
from mmcv.runner import BaseModule, auto_fp16


class BaseModel(BaseModule, metaclass=ABCMeta):
    """Base class for models"""

    def __init__(self):
        super(BaseModel, self).__init__()
        self.fp16_enabled = False

    def add_batch_input_shape(self, img, img_metas):
        batch_input_shape = tuple(img.size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, ref_expr_inds, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
            on whether ``return_loss`` is ``True``.
        """
        self.add_batch_input_shape(img, img_metas)

        if return_loss:
            return self.forward_train(img, ref_expr_inds, img_metas, **kwargs)
        else:
            return self.forward_test(img, ref_expr_inds, img_metas, **kwargs)
