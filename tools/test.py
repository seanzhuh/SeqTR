import argparse
from mmcv.utils import Config, DictAction

import torch.distributed as dist
from seqtr.apis import validate_model, set_random_seed
from seqtr.datasets import build_dataset, build_dataloader
from seqtr.models import build_model, ExponentialMovingAverage
from seqtr.utils import get_root_logger, load_checkpoint, init_dist, is_main, load_pretrained

from mmcv.runner import get_dist_info
from mmcv.parallel import MMDistributedDataParallel
try:
    import apex
except:
    pass


def main_worker(cfg):
    cfg.distributed = False
    if cfg.launcher == "pytorch":
        cfg.distributed = True
        init_dist()
    cfg.rank, cfg.world_size = get_dist_info()
    if is_main():
        logger = get_root_logger()
        logger.info(cfg.pretty_text)

    if cfg.dataset == "PretrainingVG":
        prefix = ['val_refcoco_unc', 'val_refcocoplus_unc', 'val_refcocog_umd']
        datasets_cfg = [cfg.data.train,
                        cfg.data.val_refcoco_unc,
                        cfg.data.val_refcocoplus_unc,
                        cfg.data.val_refcocog_umd]
    else:
        prefix = ['val']
        datasets_cfg = [cfg.data.train, cfg.data.val]
        if hasattr(cfg.data, 'testA') and hasattr(cfg.data, 'testB'):
            datasets_cfg.append(cfg.data.testA)
            datasets_cfg.append(cfg.data.testB)
            prefix.extend(['testA', 'testB'])
        elif hasattr(cfg.data, 'test'):
            datasets_cfg.append(cfg.data.test)
            prefix.extend(['test'])
    datasets = list(map(build_dataset, datasets_cfg))
    dataloaders = list(
        map(lambda dataset: build_dataloader(cfg, dataset), datasets[1:]))

    model = build_model(cfg.model,
                        word_emb=datasets[0].word_emb,
                        num_token=datasets[0].num_token)
    model = model.cuda()
    if cfg.use_fp16:
        model = apex.amp.initialize(
            model, opt_level="O1")
        for m in model.modules():
            if hasattr(m, "fp16_enabled"):
                m.fp16_enabled = True
    if cfg.distributed:
        model = MMDistributedDataParallel(model, device_ids=[cfg.rank])
    model_ema = ExponentialMovingAverage(
        model, cfg.ema_factor) if cfg.ema else None
    if cfg.load_from:
        eval_epoch, _, _ = load_checkpoint(
            model, model_ema, None, cfg.load_from)
    elif cfg.load_pretrained_from:
        # hacky way
        eval_epoch = 0
        start_epoch, best_det_acc, best_mask_iou = load_pretrained(
            model, model_ema, cfg.load_pretrained_from, amp=cfg.use_fp16)

    for eval_loader, _prefix in zip(dataloaders, prefix):
        if is_main():
            logger = get_root_logger()
            logger.info(f"macvg - evaluating set {_prefix}")
        validate_model(eval_epoch, cfg, model, eval_loader)
        if cfg.ema:
            if is_main():
                logger = get_root_logger()
                logger.info(f"macvg - evaluating set {_prefix} using ema")
            model_ema.apply_shadow()
            validate_model(eval_epoch, cfg, model, eval_loader)
            model_ema.restore()

    if cfg.distributed:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="macvg-test")
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load from.')
    parser.add_argument(
        '--load-pretrained-from', help='the pretrained checkpoint file to load from.')
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


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.load_from = args.load_from
    cfg.load_pretrained_from = args.load_pretrained_from
    cfg.launcher = args.launcher

    if cfg.seed is not None:
        set_random_seed(cfg.seed, deterministic=cfg.deterministic)

    main_worker(cfg)


if __name__ == '__main__':
    main()
