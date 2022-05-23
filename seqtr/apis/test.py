import time
import torch
import numpy

import pycocotools.mask as maskUtils
from seqtr.datasets import extract_data
from seqtr.utils import get_root_logger, reduce_mean, is_main

from mmdet.core.bbox.iou_calculators.iou2d_calculator import bbox_overlaps


def mask_overlaps(gt_mask, pred_masks, is_crowd):
    """Args:
        gt_mask (list[RLE]):
        pred_mask (list[RLE]):
    """

    def computeIoU_RLE(gt_mask, pred_masks, is_crowd):
        mask_iou = maskUtils.iou(pred_masks, gt_mask, is_crowd)
        mask_iou = numpy.diag(mask_iou)
        return mask_iou

    mask_iou = computeIoU_RLE(gt_mask, pred_masks, is_crowd)
    mask_iou = torch.from_numpy(mask_iou)

    return mask_iou


def accuracy(pred_bboxes, gt_bbox, pred_masks, gt_mask, is_crowd=None, device="cuda:0"):
    eval_det = pred_bboxes is not None
    eval_mask = pred_masks is not None

    det_acc = torch.tensor([-1.], device=device)
    bbox_iou = torch.tensor([-1.], device=device)
    if eval_det:
        gt_bbox = torch.stack(gt_bbox).to(device)
        bbox_iou = bbox_overlaps(gt_bbox, pred_bboxes, is_aligned=True)
        det_acc = (bbox_iou >= 0.5).float().mean()

    mask_iou = torch.tensor([-1.], device=device)
    mask_acc_at_thrs = torch.full((5, ), -1., device=device)
    if eval_mask:
        mask_iou = mask_overlaps(gt_mask, pred_masks, is_crowd).to(device)
        for i, iou_thr in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
            mask_acc_at_thrs[i] = (
                mask_iou >= iou_thr).float().mean()

    return det_acc * 100., mask_iou * 100., mask_acc_at_thrs * 100.


def evaluate_model(epoch,
                   cfg,
                   model,
                   loader):
    model.eval()

    device = list(model.parameters())[0].device

    batches = len(loader)
    end = time.time()

    with_bbox, with_mask = False, False
    det_acc_list, mask_iou_list, mask_acc_list = [], [], []
    with torch.no_grad():
        for batch, inputs in enumerate(loader):
            gt_bbox, gt_mask, is_crowd = None, None, None
            if 'gt_bbox' in inputs:
                with_bbox = True
                gt_bbox = inputs.pop('gt_bbox').data[0]
            if 'gt_mask_rle' in inputs:
                with_mask = True
                gt_mask = inputs.pop('gt_mask_rle').data[0]
            if 'is_crowd' in inputs:
                is_crowd = inputs.pop('is_crowd').data[0]

            if not cfg.distributed:
                inputs = extract_data(inputs)

            predictions = model(**inputs,
                                return_loss=False,
                                rescale=False,
                                with_bbox=with_bbox,
                                with_mask=with_mask)

            pred_bboxes = predictions.pop('pred_bboxes')
            pred_masks = predictions.pop('pred_masks')

            batch_det_acc, batch_mask_iou, batch_mask_acc_at_thrs = accuracy(
                pred_bboxes, gt_bbox, pred_masks, gt_mask, is_crowd=is_crowd, device=device)
            if cfg.distributed:
                batch_det_acc = reduce_mean(batch_det_acc)
                batch_mask_iou = reduce_mean(batch_mask_iou)
                batch_mask_acc_at_thrs = reduce_mean(batch_mask_acc_at_thrs)

            det_acc_list.append(batch_det_acc.item())
            mask_iou_list.append(batch_mask_iou)
            mask_acc_list.append(batch_mask_acc_at_thrs)

            det_acc = sum(det_acc_list) / len(det_acc_list)
            mask_iou = torch.cat(mask_iou_list).mean().item()
            mask_acc = torch.vstack(
                mask_acc_list).mean(dim=0).tolist()
            if is_main():
                if batch % cfg.log_interval == 0 or batch == batches - 1:
                    logger = get_root_logger()
                    logger.info(f"validate - epoch [{epoch+1}]-[{batch+1}/{batches}] " +
                                f"time: {(time.time() - end):.2f}, " +
                                f"DetACC@0.5: {det_acc:.2f}, " +
                                f"mIoU: {mask_iou:.2f}, " +
                                f"MaskACC@0.5-0.9: [{mask_acc[0]:.2f}, {mask_acc[1]:.2f}, {mask_acc[2]:.2f},  {mask_acc[3]:.2f},  {mask_acc[4]:.2f}]"
                                )

            end = time.time()

    return det_acc, mask_iou
