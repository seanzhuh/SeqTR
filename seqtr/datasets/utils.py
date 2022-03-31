import re
import torch
import numpy
import pickle
import os.path as osp
from typing import Sequence, Mapping
from mmcv.parallel import DataContainer
from torch.utils.data.dataloader import default_collate


def get_device(data):
    if isinstance(data, list):
        for item in data:
            device = get_device(item)
            if device != -1:
                return device
        return -1
    if isinstance(data, torch.Tensor):
        return data.get_device() if data.is_cuda else -1
    else:
        raise Exception(f'Unknown type{type(data)}.')


def cpu_to_gpu(input, device):
    if isinstance(input, list):
        outputs = list(map(lambda input_single: cpu_to_gpu(
            input_single, device), input))
        return outputs
    elif isinstance(input, torch.Tensor):
        output = input.contiguous()
        if device != -1:  # already reside on gpu
            output = output.cuda(device, non_blocking=True)
        return output
    else:
        raise Exception(f'Unknown type {type(input)}.')


def extract_data(inputs):
    assert isinstance(inputs, dict)
    new_inputs = {}
    for key, value in inputs.items():
        assert isinstance(value, DataContainer)
        data = value.data
        if value.cpu_only:
            # img_metas and gt_mask
            new_inputs[key] = data[0]
        else:
            device = get_device(data)
            if device == -1:
                data = cpu_to_gpu(data[0], torch.cuda.current_device())
            new_inputs[key] = data
    return new_inputs


def collate_fn(batch):
    """
    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`.
    Args
        batch (list[dict]): list length == samples_per_gpu,
            dict is the output of CollectData.
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        stacked = []
        if batch[0].cpu_only:  # img_metas and gt_masks
            stacked.append([sample.data for sample in batch])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:  # img, ref_expr_inds
            pad_dims = batch[0].pad_dims
            assert isinstance(batch[0].data, torch.Tensor)

            if pad_dims is not None:  # img
                ndim = batch[0].dim()
                assert ndim > pad_dims
                max_shape = [0 for _ in range(pad_dims)]
                for dim in range(1, pad_dims + 1):
                    max_shape[dim - 1] = batch[0].size(-dim)
                for sample in batch:
                    for dim in range(0, ndim - pad_dims):
                        assert batch[0].size(dim) == sample.size(dim)
                    for dim in range(1, pad_dims + 1):
                        max_shape[dim - 1] = max(max_shape[dim - 1],
                                                 sample.size(-dim))
                pad_seqs = []
                for sample in batch:
                    pad_seq = [0 for _ in range(pad_dims * 2)]
                    for dim in range(1, pad_dims + 1):
                        pad_seq[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                    pad_seqs.append(pad_seq)
                padded_samples = list(map(lambda sample, pad_seq: f.pad(
                    sample.data, pad_seq, value=sample.padding_value), batch, pad_seqs))
                stacked.append(default_collate(padded_samples))
            elif pad_dims is None:
                stacked.append(
                    default_collate([
                        sample.data
                        for sample in batch
                    ]))
            else:
                raise ValueError(
                    'pad_dims should be either None or integers (1-3)')
        else:
            # gt_bbox
            stacked.append([sample.data for sample in batch])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Mapping):
        return {
            key: collate_fn([d[key] for d in batch])
            for key in batch[0]
        }
    else:
        return default_collate(batch)


def build_word_emb_loader(cfg):
    word_emb_loader = None
    if cfg is not None:
        if cfg.type == 'GloVe':
            try:
                import en_vectors_web_lg
                word_emb_loader = en_vectors_web_lg.load()
            except:
                raise ImportError(
                    "spacy and en_vectors_web_lg is not installed")
        elif cfg.type == 'EloMo':
            pass
        else:
            raise TypeError(f"Unknown type {cfg.type} of word embedding")
    return word_emb_loader


def tokenize(annsfile, anns_all, word_emb_cfg=None):
    annsdir = osp.dirname(annsfile)
    token2ix_path = osp.join(annsdir, 'token_to_ix.pkl')
    ix2token_path = osp.join(annsdir, 'ix_to_token.pkl')
    word_emb_path = osp.join(annsdir, 'word_emb.npz')
    if osp.exists(token2ix_path) and osp.exists(ix2token_path) and osp.exists(word_emb_path):
        with open(token2ix_path, 'rb') as handle:
            token2ix = pickle.load(handle)
        with open(ix2token_path, 'rb') as handle:
            ix2token = pickle.load(handle)
        npz = numpy.load(word_emb_path, allow_pickle=True)
        return token2ix, ix2token, npz['word_emb']
    else:
        token2ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        word_emb = []
        word_emb_loader = build_word_emb_loader(word_emb_cfg)
        if word_emb_loader:
            word_emb.append(word_emb_loader('PAD').vector)
            word_emb.append(word_emb_loader('UNK').vector)
            word_emb.append(word_emb_loader('CLS').vector)

        for which_set in anns_all:
            for anns_which_set in anns_all[which_set]:
                for expression in anns_which_set['expressions']:
                    words = re.sub(
                        r"([.,'!?\"()*#:;])",
                        '',
                        expression.lower()
                    ).replace('-', ' ').replace('/', ' ').split()

                    for word in words:
                        if word not in token2ix:
                            token2ix[word] = len(token2ix)
                            if word_emb_loader:
                                word_emb.append(word_emb_loader(word).vector)

        ix2token = {}
        for token in token2ix:
            ix2token[token2ix[token]] = token

        word_emb = numpy.array(word_emb)

        with open(token2ix_path, 'wb') as handle:
            pickle.dump(token2ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(ix2token_path, 'wb') as handle:
            pickle.dump(ix2token, handle, protocol=pickle.HIGHEST_PROTOCOL)
        numpy.savez_compressed(word_emb_path,
                               word_emb=word_emb)
        return token2ix, ix2token, word_emb
