# -*- encoding: utf-8 -*-

from torch import optim


def SGD(parameters, lr, weight_decay):
    print('Initialized SGD optimizer: lr=%f, weight_decay=%f' % (
        lr, weight_decay)
    )
    return optim.SGD(parameters,
                     lr=lr, weight_decay=weight_decay, momentum=0.9)
