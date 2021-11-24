# -*- encoding: utf-8 -*-

from torch import optim


def Adam(parameters, lr, weight_decay):
    print('Initialized Adam optimizer: lr=%f, weight_decay=%f' % (lr, weight_decay))
    return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
