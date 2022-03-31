import mmcv
import numpy
import random

from ..builder import PIPELINES
import pycocotools.mask as maskUtils


@PIPELINES.register_module()
class Resize(object):
    """Resize image & gt_bbox & gt_mask.

    This transform resizes the input image to some scale, gt_bbox and gt_mask are
    then resized with the same scale factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale), from which multi-scale mode randomly sample a scale.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
    """

    def __init__(self,
                 img_scale=None,
                 keep_ratio=True,
                 interpolation='bilinear',
                 backend='cv2'):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio

    @staticmethod
    def _random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple): Returns ``img_scale``, the selected image scale
        """
        assert mmcv.is_list_of(img_scales, tuple)
        return img_scales[numpy.random.randint(len(img_scales))]

    def _random_scale(self, results):
        if len(self.img_scale) == 1:
            scale = self.img_scale[0]
        else:
            scale = self._random_select(self.img_scale)

        results['scale'] = scale

    def _resize_img(self, results):
        if self.keep_ratio:
            img = mmcv.imrescale(
                results['img'],
                results['scale'],
                interpolation=self.interpolation,
                backend=self.backend
            )
            new_h, new_w = img.shape[:2]
            h, w = results['ori_shape'][:2]
            w_scale, h_scale = new_w / w, new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'],
                results['scale'],
                return_scale=True,
                interpolation=self.interpolation,
                backend=self.backend
            )
        scale_factor = numpy.array(
            [w_scale, h_scale, w_scale, h_scale], dtype=numpy.float32)

        results['img'] = img
        results['img_shape'] = img.shape
        # in case that there is no padding
        results['pad_shape'] = img.shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bbox(self, results):
        if results['with_bbox']:
            gt_bbox = results['gt_bbox'] * results['scale_factor']
            results['gt_bbox'] = gt_bbox

    def _resize_mask(self, results):
        if results['with_mask']:
            if self.keep_ratio:
                results['gt_mask'] = results['gt_mask'].rescale(
                    results['scale'])
            else:
                results['gt_mask'] = results['gt_mask'].resize(
                    results['img_shape'][:2])
            results['gt_mask_rle'] = maskUtils.encode(
                numpy.asfortranarray(results['gt_mask'].masks[0]))

    def __call__(self, results):
        """Call function to resize image, gt_bbox, gt_mask

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        self._random_scale(results)
        self._resize_img(results)
        self._resize_bbox(results)
        self._resize_mask(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class Normalize:
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = numpy.array(mean, dtype=numpy.float32)
        self.std = numpy.array(std, dtype=numpy.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from upstream pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class Pad:
    """Pad the image & gt_mask.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
           Currently only used for YOLO-series. Default: False.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_to_square_size=(640, 640),
                 pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square
        self.pad_to_square_size = pad_to_square_size

        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.pad_to_square:
            self.size = self.pad_to_square_size
        if self.size is not None:
            padded_img = mmcv.impad(
                results['img'], shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        if results['with_mask']:
            pad_shape = results['pad_shape'][:2]
            results['gt_mask'] = results['gt_mask'].pad(
                pad_shape, pad_val=self.pad_val)
            results['gt_mask_rle'] = maskUtils.encode(
                numpy.asfortranarray(results['gt_mask'].masks[0]))

    def __call__(self, results):
        """Call function to pad image, gt_mask.
        Args:
            results (dict): Result dict from upstream pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class LargeScaleJitter:
    # placed right after loading the image from disk
    # followed by padding possibly
    def __init__(self,
                 out_max_size=640,
                 jitter_min=0.3,
                 jitter_max=1.4,
                 min_iou_thr=0.3,
                 crop_iou_thr=[0.5, 0.6, 0.7, 0.8, 0.9]):
        self.out_max_size = out_max_size
        self.jitter_min = jitter_min
        self.jitter_max = jitter_max
        self.crop_iou_thr = crop_iou_thr
        self.min_iou_thr = min_iou_thr
        self.jitter_times = 100

    def _bbox_overlaps(self, crop_bbox, gt_bbox):
        lt = numpy.maximum(crop_bbox[:2], gt_bbox[:2])
        rb = numpy.minimum(crop_bbox[2:], gt_bbox[2:])
        wh = rb - lt
        overlap = wh[0] * wh[1]
        area_gt_bbox = (
            gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
        return overlap / area_gt_bbox

    def _mask_overlaps(self, crop_bbox, gt_mask):
        h, w = gt_mask.height, gt_mask.width
        crop_mask = numpy.zeros((h, w), dtype=numpy.uint8)
        crop_mask[crop_bbox[1]:crop_bbox[3],
                  crop_bbox[0]:crop_bbox[2]] = 1
        overlap = numpy.logical_and(crop_mask, gt_mask.masks[0])
        return numpy.sum(overlap) / gt_mask.areas

    def __call__(self, results):
        img = results['img']
        h, w = results['ori_shape'][:2]
        with_bbox, with_mask = results['with_bbox'], results['with_mask']

        rand_scale = self.jitter_min
        rand_scale = rand_scale + random.random() * (self.jitter_max - self.jitter_min)
        keep_aspect_ratio_scale = self.out_max_size / max(h, w)
        scale = rand_scale * keep_aspect_ratio_scale

        img = mmcv.imrescale(
            img, scale, interpolation="bilinear", backend=None)
        new_h, new_w = img.shape[:2]

        # rescale bbox & mask
        if with_bbox:
            gt_bbox = results['gt_bbox']
            factor = numpy.array([new_w/w, new_h/h,
                                  new_w/w, new_h/h])
            gt_bbox = gt_bbox * factor
        if with_mask:
            gt_mask = results['gt_mask']
            gt_mask = gt_mask.rescale(scale, interpolation="bilinear")
            assert gt_mask.height == new_h and gt_mask.width == new_w

        # only crop here, pad will do the job when rand_scale < 1
        if rand_scale > 1.:
            w_out, h_out = mmcv.rescale_size(
                (w, h), mmcv.utils.to_2tuple(self.out_max_size))
            flag, best_idx, best_iou, history = False, -1, 0, []
            for i, iou_thr in enumerate(self.crop_iou_thr[::-1]):
                if not flag:
                    for iter in range(self.jitter_times):
                        offset = (random.random()*(new_w-w_out),
                                  random.random()*(new_h-h_out))
                        crop_bbox = numpy.array([offset[0], offset[1],
                                                 offset[0]+w_out, offset[1]+h_out])
                        if with_bbox:  # rec & res default to rec
                            iou = self._bbox_overlaps(crop_bbox, gt_bbox)
                        elif with_mask:
                            iou = self._mask_overlaps(
                                crop_bbox.astype(numpy.uint32), gt_mask)
                        history.append(crop_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = i * self.jitter_times + iter
                        if iou >= iou_thr:
                            flag = True
                            break
            if not flag:
                # escape, do nothing
                if best_iou < self.min_iou_thr:
                    # do nothing
                    results['img_shape'] = img.shape
                    # in case that there is no padding
                    results['pad_shape'] = img.shape
                    results['scale_factor'] = numpy.array([1., 1., 1., 1.])
                    results['keep_ratio'] = True
                    return results

                crop_bbox = history[best_idx]

            crop_bbox = crop_bbox.astype(numpy.uint32)
            img = img[crop_bbox[1]:crop_bbox[3],
                      crop_bbox[0]:crop_bbox[2]]
            new_h, new_w = img.shape[:2]
            assert new_h == h_out and new_w == w_out
            if with_bbox:
                gt_bbox = gt_bbox - numpy.array([
                    offset[0], offset[1], offset[0], offset[1]])
            if with_mask:
                gt_mask = gt_mask.crop(crop_bbox)

        if with_bbox:
            gt_bbox[0::2] = numpy.clip(gt_bbox[0::2], 0, new_w-1)
            gt_bbox[1::2] = numpy.clip(gt_bbox[1::2], 0, new_h-1)
        if with_mask:
            assert new_h == gt_mask.height and new_w == gt_mask.width

        if with_bbox:
            assert gt_bbox[0] >= 0 and gt_bbox[1] >= 0
            assert gt_bbox[2] <= new_w and gt_bbox[3] <= new_h
            results['gt_bbox'] = gt_bbox
        if with_mask:
            results['gt_mask'] = gt_mask
            results['gt_mask_rle'] = maskUtils.encode(
                numpy.asfortranarray(gt_mask.masks[0]))

        results['img'] = img
        results['img_shape'] = img.shape
        # in case that there is no padding
        results['pad_shape'] = img.shape
        results['scale_factor'] = numpy.array(
            [new_w/w, new_h/h, new_w/w, new_h/h])
        results['keep_ratio'] = True

        return results
