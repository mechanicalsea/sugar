# -*- encoding: utf-8 -*-

from torch.optim import lr_scheduler


def StepLR(optimizer, step_epoch, lr_decay, lr_step='epoch'):
    sche_fn = lr_scheduler.StepLR(optimizer, step_size=step_epoch, gamma=lr_decay)
    print('Initialized step LR scheduler: step %d, gamma %f' % (step_epoch, lr_decay))
    return sche_fn, lr_step
