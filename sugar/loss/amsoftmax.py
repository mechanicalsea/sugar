#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Modified from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
import torch.nn.functional as F

class AMSoftmax(nn.Module):
    """
    AMSoftmax is with hyper-parameter: margin/m and scale/s, where margin works for penalty.
        It is modified from: https://github.com/clovaai/voxceleb_trainer. 

        Note that if fine tune is available, some practice from [1] is helpful.
            1. First margin = 0.3 and scale = 40 when training from scratch,
            2. Then margin = 0.5 and scale = 30 when fine tune.
        Or fine tune after training under softmax, margin = 0.4 and s = 50 as [2].

        [1] Garcia-Romero, D., Sell, G., Mccree, A., 2020. MagNetO: X-vector Magnitude Estimation Network plus Offset for Improved Speaker Recognition, in: Odyssey 2020 The Speaker and Language Recognition Workshop. ISCA, ISCA, pp. 1–8. https://doi.org/10.21437/Odyssey.2020-1
        [2] Hajibabaei, M., Dai, D., 2018. Unified Hypersphere Embedding for Speaker Recognition. arXiv. http://arxiv.org/abs/1807.08312

        Note
        ----
            Margin is a penalty term to loss function, where inceasing margin results in larger loss because it forces the objective to learning more margin.
            Scale is like a scale of the margin, which means the large scale brings more space within intra-class.
            Intuitively, when classified correctly, larger scale lower loss, but larger margin higher loss. On the other hand, when classified uncorrectly, increasing both of them leads to higher loss.
            However, large loss do not means the accuracy of a model is bad.
    """
    def __init__(self, nOut, nClasses, margin=0.3, scale=40):

        super(AMSoftmax, self).__init__()

        self.test_normalize = True
        self.is_softmax = False
        
        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.W = nn.Parameter(torch.Tensor(nOut, nClasses)) # for adapting AAM-Softmax
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialized AMSoftmax margin=%.3f scale=%.3f without CrossEntropyLoss' % (self.m,self.s))

    def soft(self):
        """Convert AM-Softmax mode to Softmax for the first training."""
        self.is_softmax = True

    def notsoft(self):
        """Convert Softmax mode to AM-Softmax for the fine tune."""
        self.is_softmax = False

    def forward(self, x, label=None):
        """
            x : torch.Tensor [batch, in_feats]
        """

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        # Softmax mode
        if self.is_softmax: return torch.mm(x, self.W)

        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.W, p=2, dim=0)
        cosine = torch.mm(x_norm, w_norm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)
        return output
