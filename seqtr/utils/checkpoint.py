import torch
import shutil
import os.path as osp
from seqtr.utils import is_main
from .logger import get_root_logger
try:
    import apex
except:
    pass


def is_paral_model(model):
    from mmcv.parallel import MMDistributedDataParallel
    from torch.nn.parallel import DistributedDataParallel
    return isinstance(model, MMDistributedDataParallel) or isinstance(model, DistributedDataParallel)


def is_paral_state(state_dict):
    return list(state_dict.keys())[0].startswith("module.")


def de_parallel(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict[key[7:]] = value
    return new_state_dict


def log_loaded_info(ckpt, load_file):
    logger = get_root_logger()
    log_str = f"loaded checkpoint from {load_file}\n"
    if 'epoch' and 'lr' in ckpt:
        log_str += f"epoch: {ckpt['epoch']+1} lr: {ckpt['lr']:.6f}\n"
    if "best_det_acc" in ckpt:
        log_str += f"best det acc: {ckpt['best_det_acc']:.2f}\n"
        best_det_acc = ckpt['best_det_acc']
    if "best_mask_iou" in ckpt:
        log_str += f"best mIoU: {ckpt['best_mask_iou']:.2f}\n"
        best_mask_iou = ckpt['best_mask_iou']
    if "det_acc" in ckpt:
        log_str += f"loaded det acc: {ckpt['det_acc']:.2f}\n"
    if "mask_iou" in ckpt:
        log_str += f"loaded mIoU: {ckpt['mask_iou']:.2f}\n"
    logger.info(log_str)
    return best_det_acc, best_mask_iou


# only for finetuning, if resume from pretraining, use load_checkpoint
def load_pretrained(model,
                    model_ema=None,
                    load_pretrained_from=None,
                    amp=False):
    start_epoch, best_det_acc, best_mask_iou = -1, 0., 0.
    ckpt = torch.load(load_pretrained_from,
                      map_location=lambda storage, loc: storage.cuda())
    state, ema_state = ckpt['state_dict'], ckpt['ema_state_dict']
    state = de_parallel(state)
    ema_state = de_parallel(ema_state)
    state.pop("lan_enc.embedding.weight")
    ema_state.pop("lan_enc.embedding.weight")

    model_seq_embed_dim = model.head.transformer.seq_positional_encoding.embedding.weight.size(
        0)
    state_seq_embed_dim = state["head.transformer.seq_positional_encoding.embedding.weight"].size(
        0)
    # finetuning on RES since pretraining is only performed on REC
    if model_seq_embed_dim != state_seq_embed_dim:
        state.pop("head.transformer.seq_positional_encoding.embedding.weight")
        ema_state.pop(
            "head.transformer.seq_positional_encoding.embedding.weight")
    model.load_state_dict(state, strict=False)
    if model_ema is not None:
        model_ema.shadow = ema_state
    if 'amp' in ckpt and amp:
        apex.amp.load_state_dict(ckpt['amp'])
    if is_main():
        best_det_acc, best_mask_iou = log_loaded_info(
            ckpt, load_pretrained_from)
    return start_epoch, best_det_acc, best_mask_iou


def load_checkpoint(model,
                    model_ema=None,
                    resume_from=None,
                    load_from=None,
                    amp=False,
                    optimizer=None,
                    scheduler=None):
    start_epoch, best_det_acc, best_mask_iou = -1, 0., 0.
    load_file = resume_from if resume_from is not None else load_from
    ckpt = torch.load(load_file,
                      map_location=lambda storage, loc: storage.cuda())
    state = ckpt['state_dict']
    if "ema_state_dict" in ckpt:
        ema_state = ckpt['ema_state_dict']
        if is_paral_state(ema_state) and not is_paral_model(model):
            ema_state = de_parallel(ema_state)
    if is_paral_state(state) and not is_paral_model(model):
        state = de_parallel(state)

    model.load_state_dict(state, strict=True)
    if model_ema is not None:
        model_ema.shadow = ema_state
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    if 'amp' in ckpt and amp:
        apex.amp.load_state_dict(ckpt['amp'])

    if 'epoch' in ckpt:
        if load_from is None and resume_from is not None:
            start_epoch = ckpt['epoch']
    if is_main():
        best_det_acc, best_mask_iou = log_loaded_info(ckpt, load_file)
    return start_epoch, best_det_acc, best_mask_iou


def save_checkpoint(work_dir, interval, model, model_ema, optimizer, scheduler, checkpoint):
    epoch = checkpoint['epoch'] + 1
    logger = get_root_logger()
    use_fp16 = checkpoint.pop('use_fp16', False)
    if use_fp16:
        checkpoint.update({'amp': apex.amp.state_dict()})
    checkpoint.update({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'lr': optimizer.param_groups[0]["lr"],
    })
    if model_ema is not None:
        checkpoint.update({'ema_state_dict': model_ema.shadow})
    latest_path = osp.join(work_dir, "latest.pth")
    det_best_path = osp.join(work_dir, "det_best.pth")
    mask_best_path = osp.join(work_dir, "mask_best.pth")
    torch.save(checkpoint, latest_path)
    if is_main():
        logger.info(
            f"saved epoch {epoch} checkpoint at {latest_path}")
    if interval > 0 and epoch % interval == 0:
        torch.save(checkpoint, osp.join(work_dir, f'epoch_{epoch}.pth'))
    if checkpoint['det_acc'] > checkpoint['best_det_acc']:
        shutil.copyfile(latest_path, det_best_path)
        if is_main():
            logger.info(f"saved epoch {epoch} checkpoint at {det_best_path}")
    if checkpoint['mask_iou'] > checkpoint['best_mask_iou']:
        shutil.copyfile(latest_path, mask_best_path)
        if is_main():
            logger.info(f"saved epoch {epoch} checkpoint at {mask_best_path}")
