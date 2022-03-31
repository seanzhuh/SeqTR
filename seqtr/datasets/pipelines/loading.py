import re
import mmcv
import numpy
import torch
import os.path as osp
from mmdet.core import BitmapMasks
import pycocotools.mask as maskUtils


from ..builder import PIPELINES


def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')


@PIPELINES.register_module()
class LoadImageAnnotationsFromFile(object):
    """Load an image, referring expression, gt_bbox, gt_mask from file.

    Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 dataset='RefCOCOUNC',
                 color_type='color',
                 backend=None,
                 file_client_cfg=dict(backend='disk'),
                 max_token=15,
                 with_bbox=False,
                 with_mask=False):
        self.color_type = color_type
        self.backend = backend
        self.file_client_cfg = file_client_cfg.copy()
        self.file_client = None
        self.max_token = max_token
        assert with_bbox or with_mask
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        assert dataset in ['RefCOCOUNC', 'RefCOCOGoogle', 'RefCOCOgUMD',
                           'RefCOCOgGoogle', 'RefCOCOPlusUNC', 'ReferItGameBerkeley', 'Flickr30k', 'PretrainingVG']
        self.dataset = dataset

    def _load_img(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_cfg)

        if "ReferItGame" in self.dataset or "Flickr30k" in self.dataset:
            filepath = osp.join(results['imgsfile'],
                                "%d.jpg" % results['ann']['image_id'])
        elif "RefCOCO" in self.dataset:
            filepath = osp.join(results['imgsfile'],
                                "COCO_train2014_%012d.jpg" % results['ann']['image_id'])
        elif "PretrainingVG" == self.dataset:
            data_source = results['ann']['data_source']
            img_name = "COCO_train2014_%012d.jpg" if "coco" in data_source else "%d.jpg"
            img_name = img_name % results['ann']['image_id']
            filepath = osp.join(results['imgsfile'][data_source], img_name)
        img_bytes = self.file_client.get(filepath)
        img = mmcv.imfrombytes(
            img_bytes,
            flag=self.color_type,
            backend=self.backend)

        results['filename'] = filepath
        results['img'] = img
        results['img_shape'] = img.shape  # (h, w, 3), rgb default
        results['ori_shape'] = img.shape
        return results

    def _load_expression(self, results):
        expressions = results['ann']['expressions']
        # choice always the same if 'val'/'test'/'testA'/'testB'
        expression = expressions[numpy.random.choice(len(expressions))]
        expression = clean_string(expression)

        ref_expr_inds = torch.zeros(self.max_token, dtype=torch.long)
        for idx, word in enumerate(expression.split()):
            if word in results['token2idx']:
                ref_expr_inds[idx] = results['token2idx'][word]
            else:
                ref_expr_inds[idx] = results['token2idx']['UNK']
            if idx + 1 == self.max_token:
                break

        results['ref_expr_inds'] = ref_expr_inds
        results['expression'] = expression
        results['max_token'] = self.max_token
        return results

    def _load_bbox(self, results):
        if self.with_bbox:
            gt_bbox = results['ann']['bbox']
            gt_bbox[2] = gt_bbox[0] + gt_bbox[2]
            gt_bbox[3] = gt_bbox[1] + gt_bbox[3]
            gt_bbox = numpy.array(
                gt_bbox, dtype=numpy.float64)  # x1, y1, x2, y2
            h, w = results['ori_shape'][:2]
            gt_bbox[0::2] = numpy.clip(gt_bbox[0::2], 0, w-1)
            gt_bbox[1::2] = numpy.clip(gt_bbox[1::2], 0, h-1)
            results['gt_bbox'] = gt_bbox
        results['with_bbox'] = self.with_bbox
        return results

    def _load_mask(self, results):
        if self.with_mask:
            mask = results['ann']['mask']
            h, w = results['ori_shape'][:2]

            is_crowd = 0
            if isinstance(mask, list):  # polygon
                rles = maskUtils.frPyObjects(mask, h, w)
                if len(rles) > 1:
                    is_crowd = 1
                # sometimes there are multiple binary map (corresponding to multiple segs)
                rle = maskUtils.merge(rles)
            else:
                rle = mask
            mask = maskUtils.decode(rle)
            mask = BitmapMasks(mask[None], h, w)
            results['gt_mask'] = mask
            results['gt_mask_rle'] = rle  # {'size':, 'counts'}
            results['is_crowd'] = is_crowd

        results['with_mask'] = self.with_mask
        return results

    def __call__(self, results):
        results = self._load_img(results)
        results = self._load_expression(results)
        results = self._load_bbox(results)
        results = self._load_mask(results)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_cfg}, '
                    f'max_token={self.max_token}, '
                    f'with_bbox={self.with_bbox}, '
                    f'with_mask={self.with_mask})')
        return repr_str
