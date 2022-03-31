import os
import torch
from datetime import timedelta
import torch.multiprocessing as mp
from torch import distributed as dist
from mmcv.runner import get_dist_info


def init_dist():
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend='nccl',
                            timeout=timedelta(minutes=3))


def is_main():
    rank, _ = get_dist_info()
    return rank == 0


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor
