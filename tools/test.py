import argparse
import torch.distributed as dist

from seqtr.apis import evaluate_model, set_random_seed
from seqtr.datasets import build_dataset, build_dataloader
from seqtr.models import build_model, ExponentialMovingAverage
from seqtr.utils import (get_root_logger, load_checkpoint, init_dist, 
                         is_main, load_pretrained_checkpoint)

from mmcv.runner import get_dist_info
from mmcv.utils import Config, DictAction
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

    if cfg.dataset == "Mixed":
        prefix = ['val_refcoco_unc', 
                  'val_refcocoplus_unc', 
                  'val_refcocog_umd',
                  'val_referitgame_berkeley',
                  'val_flickr30k']
        datasets_cfgs = [cfg.data.train,
                        cfg.data.val_refcoco_unc,
                        cfg.data.val_refcocoplus_unc,
                        cfg.data.val_refcocog_umd,
                        cfg.data.val_referitgame_berkeley,
                        cfg.data.val_flickr30k]
    else:
        prefix = ['val']
        datasets_cfgs = [cfg.data.train, cfg.data.val]
        if hasattr(cfg.data, 'testA') and hasattr(cfg.data, 'testB'):
            datasets_cfgs.append(cfg.data.testA)
            datasets_cfgs.append(cfg.data.testB)
            prefix.extend(['testA', 'testB'])
        elif hasattr(cfg.data, 'test'):
            datasets_cfgs.append(cfg.data.test)
            prefix.extend(['test'])
    datasets = list(map(build_dataset, datasets_cfgs))
    dataloaders = list(map(lambda dataset: build_dataloader(cfg, dataset), datasets[1:]))

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
    model_ema = ExponentialMovingAverage(model, cfg.ema_factor) if cfg.ema else None
    if cfg.load_from:
        load_checkpoint(model, model_ema, load_from=cfg.load_from)
    elif cfg.finetune_from:
        # hacky way
        load_pretrained_checkpoint(model, model_ema, cfg.finetune_from, amp=cfg.use_fp16)

    for eval_loader, _prefix in zip(dataloaders, prefix):
        if is_main():
            logger = get_root_logger()
            logger.info(f"SeqTR - evaluating set {_prefix}")
        evaluate_model(-1, cfg, model, eval_loader)
        if cfg.ema:
            if is_main():
                logger = get_root_logger()
                logger.info(f"SeqTR - evaluating set {_prefix} using ema")
            model_ema.apply_shadow()
            evaluate_model(-1, cfg, model, eval_loader)
            model_ema.restore()

    if cfg.distributed:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="SeqTR-test")
    parser.add_argument('config', help='test configuration file path.')
    parser.add_argument(
        '--load-from', help='load from the saved .pth checkpoint, only used in validation.')
    parser.add_argument(
        '--finetune-from', help='load from the pretrained checkpoint, only used in validation.')
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
    cfg.finetune_from = args.finetune_from
    cfg.launcher = args.launcher

    if cfg.seed is not None:
        set_random_seed(cfg.seed, deterministic=cfg.deterministic)

    main_worker(cfg)


if __name__ == '__main__':
    main()
