import torch
import torch.nn as nn

from mmdet.models.losses import weight_reduce_loss


class LabelSmoothCrossEntropyLoss(nn.Module):
    def __init__(self,
                 neg_factor=0.1):
        super(LabelSmoothCrossEntropyLoss, self).__init__()
        self.neg_factor = neg_factor
        self.reduction = 'mean'
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, targets, weight):
        logits = logits.float()
        batch_size, num_pts, num_classes = logits.size(
            0), logits.size(1), logits.size(2)
        logits = logits.reshape(-1, num_classes)
        targets = targets.reshape(-1, 1)

        with torch.no_grad():
            targets = targets.clone().detach()
            label_pos, label_neg = 1. - self.neg_factor, self.neg_factor / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(label_neg)
            lb_one_hot.scatter_(1, targets, label_pos)
            lb_one_hot = lb_one_hot.detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)

        loss = weight_reduce_loss(
            loss, weight=weight, reduction=self.reduction, avg_factor=batch_size*num_pts)

        return loss
