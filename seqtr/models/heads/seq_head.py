import torch
import random
import torch.nn as nn
from seqtr.models import HEADS
import torch.nn.functional as f
from seqtr.core.layers import LinearModule
from mmdet.models.losses import CrossEntropyLoss
from mmdet.models.utils import build_transformer
from seqtr.core.losses import LabelSmoothCrossEntropyLoss


@HEADS.register_module()
class SeqHead(nn.Module):
    def __init__(self,
                 in_ch=1024,
                 num_bin=1000,
                 multi_task="none",
                 shuffle_fraction=-1,
                 mapping="relative",
                 top_p=-1,
                 num_ray=18,
                 det_coord=[-1],
                 det_coord_weight=1.,
                 loss=dict(
                     type="LabelSmoothCrossEntropyLoss",
                     neg_factor=0.1
                 ),
                 predictor=dict(
                     num_fcs=3, in_chs=[256, 256, 256], out_chs=[256, 256, 1001],
                     fc=[
                         dict(
                             linear=dict(type='Linear', bias=True),
                             act=dict(type='ReLU', inplace=True),
                             drop=None),
                         dict(
                             linear=dict(type='Linear', bias=True),
                             act=dict(type='ReLU', inplace=True),
                             drop=None),
                         dict(
                             linear=dict(type='Linear', bias=True),
                             act=None,
                             drop=None)
                     ]
                 ),
                 transformer=dict(
                     type='AutoRegressiveTransformer',
                     encoder=dict(
                         num_layers=6,
                         layer=dict(
                             d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu', batch_first=True)),
                     decoder=dict(
                         num_layers=3,
                         layer=dict(
                             d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, activation='relu', batch_first=True),
                     )),
                 x_positional_encoding=dict(
                     type='SinePositionalEncoding2D',
                     num_feature=128,
                     normalize=True),
                 seq_positional_encoding=dict(
                     type='LearnedPositionalEncoding1D',
                     num_embedding=5,
                     num_feature=256)
                 ):
        super(SeqHead, self).__init__()
        self.num_bin = num_bin
        assert multi_task in ["v1", "none"]
        self.multi_task = multi_task
        self.shuffle_fraction = shuffle_fraction
        assert mapping in ["relative", "absolute"]
        self.mapping = mapping
        self.top_p = top_p
        self.num_ray = num_ray
        self.det_coord = det_coord
        self.det_coord_weight = det_coord_weight

        self.transformer = build_transformer(transformer)
        self.d_model = self.transformer.d_model

        self._init_layers(in_ch,
                          predictor,
                          multi_task,
                          x_positional_encoding,
                          seq_positional_encoding)

        loss_type = loss.pop('type')
        if loss_type == "CrossEntropyLoss":
            self.loss_ce = CrossEntropyLoss()
        elif loss_type == "LabelSmoothCrossEntropyLoss":
            self.loss_ce = LabelSmoothCrossEntropyLoss(
                neg_factor=loss.pop('neg_factor', 0.1))

    def _init_layers(self,
                     in_ch,
                     predictor_cfg,
                     multi_task,
                     x_positional_encoding,
                     seq_positional_encoding):
        num_fcs = predictor_cfg.pop('num_fcs')
        in_chs, out_chs = predictor_cfg.pop(
            'in_chs'), predictor_cfg.pop('out_chs')
        fc_cfg = predictor_cfg.pop('fc')
        assert num_fcs == len(fc_cfg) == len(in_chs) == len(out_chs)
        predictor = []
        for i in range(num_fcs):
            _cfg = fc_cfg[i]
            _cfg['linear']['in_features'] = in_chs[i]
            _cfg['linear']['out_features'] = out_chs[i]
            predictor.append(LinearModule(**_cfg))
            if i == num_fcs - 1:
                self.vocab_size = out_chs[i]
        assert self.vocab_size == self.num_bin + 1
        self.end = self.vocab_size - 1
        self.predictor = nn.Sequential(*predictor)

        if multi_task == "v1":
            # bbox_token, x1, y1, x2, y2, mask_token, x1, y1, ..., xN, yN
            self.task_embedding = nn.Embedding(2, self.d_model)

        self.transformer._init_layers(in_ch,
                                      self.vocab_size,
                                      x_positional_encoding,
                                      seq_positional_encoding)

    def quantize(self, seq, img_metas):
        if self.mapping == "relative":
            num_pts = seq.size(1) // 2
            norm_factor = [img_meta['pad_shape'][:2][::-1]
                           for img_meta in img_metas]
            norm_factor = seq.new_tensor(norm_factor)
            norm_factor = torch.cat(
                [norm_factor for _ in range(num_pts)], dim=1)
            return (seq / norm_factor * self.num_bin).long()
        elif self.mapping == "absolute":
            return (seq / 640. * self.num_bin).long()

    def dequantize(self, seq, scale_factor):
        if self.mapping == "relative":
            return seq * scale_factor / self.num_bin
        elif self.mapping == "absolute":
            return seq * 640. / self.num_bin

    def shuffle_sequence(self, seq):
        batch_size, num_pts = seq.size(0), seq.size(1) // 2
        seq = seq.reshape(batch_size * num_pts, 2)
        shuffle_idx = random.sample(
            range(batch_size), int(batch_size * self.shuffle_fraction))
        shuffle_idx = [idx * num_pts for idx in shuffle_idx]
        perm = torch.randperm(num_pts, device=seq.device)
        for idx in shuffle_idx:
            s = idx
            e = s + num_pts
            seq[s:e, :] = seq[s:e, :][perm]
        seq = seq.reshape(batch_size, num_pts * 2)
        return seq

    def sequentialize(self,
                      img_metas,
                      gt_bbox=None,
                      gt_mask_vertices=None,
                      ):
        """Args:
            gt_bbox (list[tensor]): [4, ].

            gt_mask_vertices (tensor): [batch_size, 2 (x, y), num_ray].
        """
        with_bbox = gt_bbox is not None
        with_mask = gt_mask_vertices is not None
        assert with_bbox or with_mask
        batch_size = len(img_metas)

        if with_bbox:
            seq_in_bbox = torch.vstack(gt_bbox)

        if with_mask:
            seq_in_mask = gt_mask_vertices.transpose(
                1, 2).reshape(batch_size, -1)

        if with_bbox and with_mask:
            assert self.multi_task != "none"
            seq_in = torch.cat([seq_in_bbox, seq_in_mask], dim=-1)
        elif with_bbox:
            seq_in = seq_in_bbox
        elif with_mask:
            seq_in = seq_in_mask

        seq_in = self.quantize(seq_in, img_metas)
        if with_mask:
            seq_in[seq_in < 0] = self.end
        seq_in[seq_in != self.end].clamp_(min=0, max=self.num_bin-1)

        if with_bbox and with_mask:
            # bbox_token, x1, y1, x2, y2, mask_token, x1, y1, ..., xN, yN
            if self.shuffle_fraction > 0.:
                seq_in[:, 4:] = self.shuffle_sequence(seq_in[:, 4:])
            seq_in_bbox, seq_in_mask = torch.split(
                seq_in, [4, seq_in.size(1)-4], dim=1)
            targets = torch.cat([seq_in_bbox, seq_in_bbox.new_full(
                (batch_size, 1), self.end), seq_in_mask, seq_in_mask.new_full((batch_size, 1), self.end)], dim=-1)
            seq_in_embeds_bbox = self.transformer.query_embedding(
                seq_in_bbox)
            seq_in_embeds_mask = self.transformer.query_embedding(
                seq_in_mask)
            task_bbox = self.task_embedding.weight[0].unsqueeze(
                0).unsqueeze(0).expand(batch_size, -1, -1)
            task_mask = self.task_embedding.weight[1].unsqueeze(
                0).unsqueeze(0).expand(batch_size, -1, -1)
            seq_in_embeds = torch.cat(
                [task_bbox, seq_in_embeds_bbox, task_mask, seq_in_embeds_mask], dim=1)
            return seq_in_embeds, targets
        else:
            if with_mask and self.shuffle_fraction > 0.:
                seq_in = self.shuffle_sequence(seq_in)
            seq_in_embeds = self.transformer.query_embedding(seq_in)
            targets = torch.cat(
                [seq_in, seq_in.new_full((batch_size, 1), self.end)], dim=-1)
            seq_in_embeds = torch.cat(
                [seq_in_embeds.new_zeros((batch_size, 1, self.d_model)), seq_in_embeds], dim=1)
            return seq_in_embeds, targets

    def forward_train(self,
                      x_mm,
                      img_metas,
                      gt_bbox=None,
                      gt_mask_vertices=None,
                      ):
        """Args:
            x (tensor): [batch_size, c, h, w].

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `macvg/datasets/pipelines/formatting.py:CollectData`.

            gt_bbox (list[tensor]): [4, ], [tl_x, tl_y, br_x, br_y] format,
                in 'img_shape' scale.

            gt_mask_vertices (list[tensor]): [batch_size, 2, num_ray], padded values are -1, 
                in 'pad_shape' scale.
        """
        with_bbox = gt_bbox is not None
        with_mask = gt_mask_vertices is not None

        x_mask, x_pos_embeds = self.transformer.x_mask_pos_enc(x_mm, img_metas)

        memory = self.transformer.forward_encoder(x_mm, x_mask, x_pos_embeds)

        seq_in_embeds, targets = self.sequentialize(
            img_metas,
            gt_bbox=gt_bbox,
            gt_mask_vertices=gt_mask_vertices)
        logits = self.transformer.forward_decoder(
            seq_in_embeds, memory, x_pos_embeds, x_mask)
        logits = self.predictor(logits)
        loss_ce = self.loss(
            logits, targets, with_bbox=with_bbox, with_mask=with_mask)

        with torch.no_grad():
            if with_mask and with_bbox:
                # bbox_token, x1, y1, x2, y2, mask_token, x1, y1, ..., xN, yN
                logits_bbox = logits[:, :4, :-1]
                scores_bbox = f.softmax(logits_bbox, dim=-1)
                _, seq_out_bbox = scores_bbox.max(
                    dim=-1, keepdim=False)
                logits_mask = logits[:, 5:, :]
                scores_mask = f.softmax(logits_mask, dim=-1)
                _, seq_out_mask = scores_mask.max(
                    dim=-1, keepdim=False)
                return dict(loss_multi_task=loss_ce), \
                    dict(seq_out_bbox=seq_out_bbox.detach(),
                         seq_out_mask=seq_out_mask.detach())
            else:
                if with_bbox:
                    logits = logits[:, :-1, :-1]
                scores = f.softmax(logits, dim=-1)
                _, seq_out = scores.max(dim=-1, keepdim=False)

                if with_bbox:
                    return dict(loss_det=loss_ce), \
                        dict(seq_out_bbox=seq_out.detach())
                elif with_mask:
                    return dict(loss_mask=loss_ce), \
                        dict(seq_out_mask=seq_out.detach())

    def loss(self, logits, targets, with_bbox=False, with_mask=False):
        """Args:
            logits (tensor): [batch_size, 5/2*num_ray+1, vocab_size].

            target (tensor): [batch_size, 5/2*num_ray+1].
        """
        batch_size, num_token = logits.size()[:2]

        if with_bbox and with_mask:
            weight = logits.new_ones((batch_size, num_token))
            overlay = [self.det_coord_weight if i %
                       5 in self.det_coord else 1. for i in range(5)]
            overlay = torch.tensor(
                overlay, device=weight.device, dtype=weight.dtype)
            for elem in weight:
                elem[:5] = overlay
            weight = weight.reshape(-1)
        elif with_bbox:
            weight = logits.new_tensor([self.det_coord_weight if i % 5 in self.det_coord else 1.
                                        for i in range(batch_size * num_token)])
        elif with_mask:
            weight = logits.new_tensor(
                [1. for _ in range(batch_size * num_token)])
            weight[targets.view(-1) == self.end] /= 10.

        loss_ce = self.loss_ce(logits, targets, weight=weight)
        return loss_ce

    def forward_test(self, x_mm, img_metas, with_bbox=False, with_mask=False):
        """Args:
            x (tensor): [batch_size, c, h, w].

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `macvg/datasets/pipelines/formatting.py:CollectData`.
        """
        x_mask, x_pos_embeds = self.transformer.x_mask_pos_enc(x_mm, img_metas)

        memory = self.transformer.forward_encoder(x_mm, x_mask, x_pos_embeds)

        return self.generate_sequence(memory, x_mask, x_pos_embeds, with_bbox=with_bbox, with_mask=with_mask)

    def generate(self, seq_in_embeds, memory, x_pos_embeds, x_mask, decode_steps, with_mask):
        seq_out = []
        for step in range(decode_steps):
            out = self.transformer.forward_decoder(
                seq_in_embeds, memory, x_pos_embeds, x_mask)
            logits = out[:, -1, :]
            logits = self.predictor(logits)
            if self.multi_task == "v1":
                if step < 4:
                    logits = logits[:, :-1]
            else:
                if not with_mask:
                    logits = logits[:, :-1]
            probs = f.softmax(logits, dim=-1)
            if self.top_p > 0.:
                sorted_score, sorted_idx = torch.sort(
                    probs, descending=True)
                cum_score = sorted_score.cumsum(dim=-1)
                sorted_idx_to_remove = cum_score > self.top_p
                sorted_idx_to_remove[...,
                                     1:] = sorted_idx_to_remove[..., :-1].clone()
                sorted_idx_to_remove[..., 0] = 0
                idx_to_remove = sorted_idx_to_remove.scatter(
                    1, sorted_idx, sorted_idx_to_remove)
                probs = probs.masked_fill(idx_to_remove, 0.)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = probs.max(dim=-1, keepdim=True)

            seq_in_embeds = torch.cat(
                [seq_in_embeds, self.transformer.query_embedding(next_token)], dim=1)

            seq_out.append(next_token)

        seq_out = torch.cat(seq_out, dim=-1)

        return seq_out

    def generate_sequence(self, memory, x_mask, x_pos_embeds, with_bbox=False, with_mask=False):
        """Args:
            memory (tensor): encoder's output, [batch_size, h*w, d_model].

            x_mask (tensor): [batch_size, h*w], dtype is torch.bool, True means
                ignored position.

            x_pos_embeds (tensor): [batch_size, h*w, d_model].
        """
        batch_size = memory.size(0)
        if with_bbox and with_mask:
            # bbox_token, x1, y1, x2, y2, mask_token, x1, y1, ..., xN, yN
            task_bbox = self.task_embedding.weight[0].unsqueeze(
                0).unsqueeze(0).expand(batch_size, -1, -1)
            seq_out_bbox = self.generate(
                task_bbox, memory, x_pos_embeds, x_mask, 4, False)
            task_mask = self.task_embedding.weight[1].unsqueeze(
                0).unsqueeze(0).expand(batch_size, -1, -1)
            seq_in_embeds_box = self.transformer.query_embedding(
                seq_out_bbox)
            seq_in_embeds_mask = torch.cat(
                [task_bbox, seq_in_embeds_box, task_mask], dim=1)
            seq_out_mask = self.generate(
                seq_in_embeds_mask, memory, x_pos_embeds, x_mask, 2 * self.num_ray + 1, True)
            return dict(seq_out_bbox=seq_out_bbox,
                        seq_out_mask=seq_out_mask)
        else:
            seq_in_embeds = memory.new_zeros((batch_size, 1, self.d_model))
            if with_mask:
                decode_steps = self.num_ray * 2 + 1
            elif with_bbox:
                decode_steps = 4
            seq_out = self.generate(
                seq_in_embeds, memory, x_pos_embeds, x_mask, decode_steps, with_mask)
            if with_bbox:
                return dict(seq_out_bbox=seq_out)
            elif with_mask:
                return dict(seq_out_mask=seq_out)
