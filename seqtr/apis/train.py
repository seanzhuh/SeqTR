import time
import copy
import numpy
import torch
import random

from .test import accuracy
from seqtr.datasets import extract_data
from seqtr.utils import get_root_logger, reduce_mean, is_main
try:
    import apex
except:
    pass


def set_random_seed(seed, deterministic=False):
    """Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(epoch,
                cfg,
                model,
                model_ema,
                optimizer,
                loader):
    model.train()

    if cfg.distributed:
        loader.sampler.set_epoch(epoch)

    device = list(model.parameters())[0].device

    batches = len(loader)
    end = time.time()

    det_acc_list, mask_iou_list, mask_acc_list, ie_list = [], [], [], []
    loss_det_list, loss_mask_list = [], []
    for batch, inputs in enumerate(loader):
        data_time = time.time() - end
        gt_bbox, gt_mask, is_crowd = None, None, None
        if 'gt_bbox' in inputs:
            gt_bbox = copy.deepcopy(inputs['gt_bbox'].data[0])
        if 'gt_mask_rle' in inputs:
            gt_mask = inputs.pop('gt_mask_rle').data[0]
        if 'is_crowd' in inputs:
            is_crowd = inputs.pop('is_crowd').data[0]

        if not cfg.distributed:
            inputs = extract_data(inputs)

        losses, predictions = model(**inputs, rescale=False)

        loss_det = losses.pop('loss_det', torch.tensor([0.], device=device))
        loss_mask = losses.pop('loss_mask', torch.tensor([0.], device=device))
        loss = loss_det + loss_mask

        optimizer.zero_grad()
        if cfg.use_fp16:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if cfg.grad_norm_clip:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.grad_norm_clip
            )
        optimizer.step()

        if cfg.ema:
            model_ema.update_params()

        if cfg.distributed:
            loss_det = reduce_mean(loss_det)
            loss_mask = reduce_mean(loss_mask)

        pred_bboxes = predictions.pop('pred_bboxes')
        pred_masks = predictions.pop('pred_masks')

        with torch.no_grad():
            batch_det_acc, batch_mask_iou, batch_mask_acc_at_thrs, batch_ie = accuracy(
                pred_bboxes, gt_bbox, pred_masks, gt_mask, is_crowd=is_crowd, device=device)
            if cfg.distributed:
                batch_det_acc = reduce_mean(batch_det_acc)
                batch_mask_iou = reduce_mean(batch_mask_iou)
                batch_mask_acc_at_thrs = reduce_mean(batch_mask_acc_at_thrs)
                batch_ie = reduce_mean(batch_ie)

        det_acc_list.append(batch_det_acc.item())
        mask_iou_list.append(batch_mask_iou)
        mask_acc_list.append(batch_mask_acc_at_thrs)
        ie_list.append(batch_ie.item())
        loss_det_list.append(loss_det.item())
        loss_mask_list.append(loss_mask.item())

        det_acc = sum(det_acc_list) / len(det_acc_list)
        mask_iou = torch.cat(mask_iou_list).mean().item()
        mask_acc = torch.vstack(
            mask_acc_list).mean(dim=0).tolist()
        ie = sum(ie_list) / len(ie_list)
        if is_main():
            if batch % cfg.log_interval == 0 or batch == batches - 1:
                logger = get_root_logger()
                logger.info(f"train - epoch [{epoch+1}]-[{batch+1}/{batches}] " +
                            f"time: {(time.time()- end):.2f}, data_time: {data_time:.2f}, " +
                            f"loss_det: {sum(loss_det_list) / len(loss_det_list) :.4f}, " +
                            f"loss_mask: {sum(loss_mask_list) / len(loss_mask_list):.4f}, " +
                            f"lr: {optimizer.param_groups[0]['lr']:.6f}, " +
                            f"DetACC@0.5: {det_acc:.2f}, " +
                            f"mIoU: {mask_iou:.2f}, " +
                            f"MaskACC@0.5-0.9: [{mask_acc[0]:.2f}, {mask_acc[1]:.2f}, {mask_acc[2]:.2f},  {mask_acc[3]:.2f},  {mask_acc[4]:.2f}], " +
                            f"IE: {ie:.2f}")

        end = time.time()
