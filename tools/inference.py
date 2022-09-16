import argparse
import os.path as osp
from typing import Sequence
from mmcv import Config, DictAction
from mmcv.utils import mkdir_or_exist
from seqtr.apis import inference_model


def parse_args():
    parser = argparse.ArgumentParser(description="macvg-inference")
    parser.add_argument('config', help='inference config file path.')
    parser.add_argument(
        'checkpoint', help='the checkpoint file to load from.')
    parser.add_argument(
        '--output-dir', help='directory where inference results will be saved.')
    parser.add_argument('--with-gt', action='store_true',
                        help='draw ground-truth bbox/mask on image if true.')
    parser.add_argument('--no-overlay', action='store_false', dest='overlay')
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
    parser.add_argument(
        '--which-set',
        type=str,
        nargs='+',
        default='val',
        help="evaluation which_sets, which depends on the dataset, e.g., \
        'val', 'testA', 'testB' for RefCOCO(Plus)UNC, and 'val', 'test' for RefCOCOgUMD.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.checkpoint = args.checkpoint
    assert args.which_set is not None, "please specify at least one which_set to inference on."
    if isinstance(args.which_set, str):
        cfg.which_set = [args.which_set]
    elif isinstance(args.which_set, Sequence):
        cfg.which_set = args.which_set
    cfg.overlay = args.overlay
    cfg.output_dir = args.output_dir
    cfg.with_gt = args.with_gt
    cfg.rank = 0
    cfg.distributed = False

    for which_set in cfg.which_set:
        mkdir_or_exist(
            osp.join(args.output_dir, cfg.dataset + "_" + which_set))

    inference_model(cfg)
