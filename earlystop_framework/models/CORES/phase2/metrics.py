import copy

import torch
from collections import defaultdict

from torch import nn


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # get top-k predictions
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # compare against ground truth
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # flatten non‑contiguous slice safely
        correct_k = correct[:k].reshape(-1).float().sum(0)
        # compute fraction
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def cross_entropy_smooth(input, target, size_average=True, label_smoothing=0.1):
    y = torch.eye(10).cuda()
    lb_oh = y[target]

    target = lb_oh * (1 - label_smoothing) + 0.5 * label_smoothing

    logsoftmax = nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


class Accumulator:
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other != key:
                    newone[key] = value / self[other]
                else:
                    newone[key] = value
            else:
                newone[key] = value / other
        return newone


class SummaryWriterDummy:
    def __init__(self, logdir):
        pass

    def add_scalar(self, *args, **kwargs):
        pass
