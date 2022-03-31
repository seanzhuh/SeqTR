import collections
from ..builder import PIPELINES
from mmcv.utils import build_from_cfg


@PIPELINES.register_module()
class Compose(object):
    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, results):
        for transform in self.transforms:
            results = transform(results)
            if results is None:
                return None
        return results
