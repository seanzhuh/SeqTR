import numpy
import random

from functools import partial
from .utils import collate_fn
from mmcv.utils import Registry
from mmcv.parallel import collate
from torch.utils.data import DataLoader
from mmdet.datasets import GroupSampler, DistributedGroupSampler, DistributedSampler


DATASETS = Registry('DATASETS')
PIPELINES = Registry('PIPELINES')


def build_dataset(cfg, default_args=None):
    """Build dataset."""
    return DATASETS.build(cfg, default_args=default_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(cfg,
                     dataset):
    if cfg.distributed:
        if dataset.which_set == "train":
            sampler = DistributedGroupSampler(
                dataset, cfg.data.samples_per_gpu, cfg.world_size, cfg.rank, seed=cfg.seed)
        else:
            sampler = DistributedSampler(
                dataset, cfg.world_size, cfg.rank, shuffle=False, seed=cfg.seed)
    else:
        sampler = GroupSampler(
            dataset, cfg.data.samples_per_gpu) if dataset.which_set == "train" else None

    init_fn = partial(
        worker_init_fn, num_workers=cfg.data.workers_per_gpu, rank=cfg.rank, seed=cfg.seed) if cfg.seed is not None else None

    return DataLoader(dataset,
                      batch_size=cfg.data.samples_per_gpu,
                      sampler=sampler,
                      shuffle=False,
                      batch_sampler=None,
                      num_workers=cfg.data.workers_per_gpu,
                      pin_memory=False,
                      collate_fn=partial(
                          collate, samples_per_gpu=cfg.data.samples_per_gpu),
                      worker_init_fn=init_fn,
                      drop_last=False,
                      persistent_workers=cfg.distributed)
