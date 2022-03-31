import time
import argparse
import os.path as osp
import torch.distributed as dist

import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info
from mmcv.parallel import MMDistributedDataParallel

from seqtr.core import build_optimizer, build_scheduler
from seqtr.datasets import build_dataset, build_dataloader
from seqtr.models import build_model, ExponentialMovingAverage
from seqtr.apis import set_random_seed, train_model, validate_model
from seqtr.utils import (get_root_logger, load_checkpoint, save_checkpoint,
                         load_pretrained, is_main, init_dist)

try:
    import apex
except:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="macvg-train")
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--load-from', help='the checkpoint file to initialize from')
    parser.add_argument('--load-pretrained-from',
                        help='the checkpoint file to finetune from')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main_worker(cfg):
    cfg.distributed = False
    if cfg.launcher == "pytorch":
        cfg.distributed = True
        init_dist()
    cfg.rank, cfg.world_size = get_dist_info()
    if is_main():
        logger = get_root_logger(
            log_file=osp.join(cfg.work_dir, "train_log.txt"))
        logger.info(cfg.pretty_text)
        cfg.dump(
            osp.join(cfg.work_dir, f'{cfg.timestamp}_' + osp.basename(cfg.config)))

    if cfg.dataset == "PretrainingVG":
        datasets = list(map(build_dataset, (cfg.data.train,
                                            cfg.data.val_refcoco_unc,
                                            cfg.data.val_refcocoplus_unc,
                                            cfg.data.val_refcocog_umd)))
    else:
        datasets = list(
            map(build_dataset, (cfg.data.train, cfg.data.val)))
    dataloaders = list(
        map(lambda dataset: build_dataloader(cfg, dataset), datasets))

    model = build_model(cfg.model,
                        word_emb=datasets[0].word_emb,
                        num_token=datasets[0].num_token)
    if is_main():
        logger = get_root_logger()
        logger.info(model)

    model = model.cuda()
    if model.vis_enc.do_train:
        train_params = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "vis_enc" not in n and p.requires_grad
                ]
            },
            {
                "params": [p for n, p in model.named_parameters() if "vis_enc" in n and p.requires_grad],
                "lr": cfg.optimizer_config.lr / 10.,
            },
        ]
    else:
        train_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = build_optimizer(cfg.optimizer_config, train_params)
    scheduler = build_scheduler(cfg.scheduler_config, optimizer)

    if cfg.use_fp16:
        model, optimizer = apex.amp.initialize(
            model, optimizer, opt_level="O1")
        for m in model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True
    if cfg.distributed:
        model = MMDistributedDataParallel(model, device_ids=[cfg.rank])
    model_ema = ExponentialMovingAverage(
        model, cfg.ema_factor) if cfg.ema else None

    start_epoch, best_det_acc, best_mask_iou = -1, 0., 0.
    if cfg.resume_from or cfg.load_from:
        start_epoch, best_det_acc, best_mask_iou = load_checkpoint(
            model, model_ema, cfg.resume_from, cfg.load_from,
            amp=cfg.use_fp16,
            optimizer=optimizer,
            scheduler=scheduler)
    elif cfg.load_pretrained_from:
        start_epoch, best_det_acc, best_mask_iou = load_pretrained(
            model, model_ema, cfg.load_pretrained_from, amp=cfg.use_fp16)

    for epoch in range(start_epoch + 1, cfg.scheduler_config.max_epoch):
        train_model(epoch, cfg, model, model_ema, optimizer, dataloaders[0])

        det_acc, mask_iou = 0, 0
        for eval_loader in dataloaders[1:]:
            if is_main():
                logger.info("Evaluating dataset: {}".format(
                    eval_loader.dataset.which_set))
            set_det_acc, set_mask_iou = validate_model(
                epoch, cfg, model, eval_loader)

            if cfg.ema:
                if is_main():
                    logger.info("Evaluating dataset using ema: {}".format(
                        eval_loader.dataset.which_set))
                model_ema.apply_shadow()
                ema_set_det_acc, ema_set_mask_iou = validate_model(
                    epoch, cfg, model, eval_loader)
                model_ema.restore()

            if cfg.ema:
                det_acc += ema_set_det_acc
                mask_iou += ema_set_mask_iou
            else:
                det_acc += set_det_acc
                mask_iou += set_mask_iou

        det_acc /= len(dataloaders[1:])
        mask_iou /= len(dataloaders[1:])

        if is_main():
            save_checkpoint(cfg.work_dir,
                            cfg.save_interval,
                            model, model_ema, optimizer, scheduler,
                            {'epoch': epoch,
                             'det_acc': det_acc,
                             'mask_iou': mask_iou,
                             'best_det_acc': best_det_acc,
                             'best_mask_iou': best_mask_iou,
                             'use_fp16': cfg.use_fp16})

        scheduler.step()

        best_det_acc = max(det_acc, best_det_acc)
        best_mask_iou = max(mask_iou, best_mask_iou)

        if cfg.distributed:
            dist.barrier()

    if cfg.distributed:
        dist.destroy_process_group()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir is not None:
        cfg.work_dir = osp.join(args.work_dir, f'{cfg.timestamp}')
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = f'./work_dir/{cfg.timestamp}_' + \
            osp.splitext(osp.basename(args.config))[0]
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.load_pretrained_from is not None:
        cfg.load_pretrained_from = args.load_pretrained_from
    assert not (cfg.load_from and cfg.resume_from)
    cfg.launcher = args.launcher
    cfg.config = args.config

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    if cfg.seed is not None:
        set_random_seed(cfg.seed, deterministic=cfg.deterministic)

    main_worker(cfg)


if __name__ == '__main__':
    main()
