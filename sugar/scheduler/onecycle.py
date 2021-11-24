# -*- encoding: utf-8 -*-

from torch.optim import lr_scheduler


def OneCycle(optimizer, max_lr, total_steps, pct_start=0.3, lr_step='step'):
    sche_fn = lr_scheduler.OneCycleLR(optimizer, max_lr,
                         total_steps=total_steps, pct_start=pct_start,
                         epochs=None, steps_per_epoch=None,
                         anneal_strategy='cos', cycle_momentum=True,
                         base_momentum=0.85, max_momentum=0.95,
                         div_factor=25.0, final_div_factor=10000.0,
                         last_epoch=-1)
    print('Initialized One Cycle LR scheduler: total_step %d, max_lr %f, pct_start %.2g' % (
        total_steps, max_lr, pct_start)
    )
    return sche_fn, lr_step
