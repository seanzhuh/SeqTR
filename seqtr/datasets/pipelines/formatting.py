import numpy
from ..builder import PIPELINES
from mmcv.parallel import DataContainer
from mmdet.datasets.pipelines import Collect, to_tensor


@PIPELINES.register_module()
class CollectData(Collect):
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'expression', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg')):
        super(CollectData, self).__init__(keys=keys, meta_keys=meta_keys)


@PIPELINES.register_module()
class DefaultFormatBundle:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img", 
    "gt_bbox", "gt_mask". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - ref_expr_inds: (1)to tensor, (2) to DataContainer (stack=True)
    - gt_bbox: (1)to tensor, (2)to DataContainer
    - gt_mask: (1)to tensor, (2)to DataContainer (cpu_only=True)
    """

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=numpy.zeros(num_channels, dtype=numpy.float32),
                std=numpy.ones(num_channels, dtype=numpy.float32),
                to_rgb=False))
        return results

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = numpy.expand_dims(img, -1)
            img = numpy.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DataContainer(to_tensor(img), stack=True)

        if 'ref_expr_inds' in results:
            results['ref_expr_inds'] = DataContainer(
                to_tensor(results['ref_expr_inds']), stack=True, pad_dims=None)

        if results['with_bbox']:
            results['gt_bbox'] = DataContainer(to_tensor(results['gt_bbox']))

        if results['with_mask']:
            results['gt_mask'] = DataContainer(
                results['gt_mask'], cpu_only=True)
            if 'gt_mask_rle' in results:
                results['gt_mask_rle'] = DataContainer(
                    results['gt_mask_rle'], cpu_only=True, pad_dims=None)
            if 'is_crowd' in results:
                results['is_crowd'] = DataContainer(
                    results['is_crowd'], cpu_only=True, pad_dims=None)
            if 'gt_mask_vertices' in results:
                results['gt_mask_vertices'] = DataContainer(
                    to_tensor(results['gt_mask_vertices']), stack=True, pad_dims=None)
            if 'mass_center' in results:
                results['mass_center'] = DataContainer(
                    to_tensor(results['mass_center']), stack=True, pad_dims=None)

        return results
