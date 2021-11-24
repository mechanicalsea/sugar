#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Modified from https://github.com/wujiyang/Face_Pytorch (Apache License)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    """
    AAMSoftmax (additive angular margin) is with hyper-parameter: margin/m and scale/s, where margin works for penalty.
        It is modified from: https://github.com/clovaai/voxceleb_trainer.

        Note that if sophisticated learning rate scheduler is available, some practice from [1] is helpful.
            1. Cyclic learning rate varys between 1e-8 and 1e-3 using the triangular2 policy.
            2. Using a margin of 0.2 and softmax prescaling (scale) of 30 for 4 cycles

        [1] Desplanques, B., Thienpondt, J., Demuynck, K., 2020. ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification, in: Interspeech 2020. ISCA, ISCA, pp. 3830–3834. https://doi.org/10.21437/Interspeech.2020-2650
    """

    def __init__(self, nOut, nClasses, margin=0.2, scale=30, easy_margin=False):
        super(AAMSoftmax, self).__init__()

        self.test_normalize = True
        self.is_softmax = False

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        # for adapting AM-Softmax
        self.W = nn.Parameter(torch.Tensor(nOut, nClasses))
        nn.init.xavier_normal_(self.W, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing 
        #   while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        print('Initialized AAMSoftmax margin %.3f scale %.3f' % (self.m, self.s))

    def soft(self):
        """Convert AAM-Softmax mode to Softmax for the first training."""
        self.is_softmax = True

    def notsoft(self):
        """Convert Softmax mode to AAM-Softmax for the fine tune."""
        self.is_softmax = False

    def forward(self, x, label=None):
        """
            x : torch.Tensor [batch, in_feats]
        """

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # Softmax mode
        if self.is_softmax:
            return torch.mm(x, self.W)

        # cos(theta)
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.W, p=2, dim=0)
        cosine = torch.mm(x_norm, w_norm)

        # cos(theta + m)
        # torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output
