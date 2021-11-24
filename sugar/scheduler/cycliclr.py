from torch.optim import lr_scheduler
from torch.optim import Adam

def CyclicLR(optimizer, max_lr, step_size, lr_step='step'):
    cycle_momentum = not isinstance(optimizer, Adam)
    sche_fn = lr_scheduler.CyclicLR(optimizer, 1e-8, max_lr, step_size_up=step_size, mode='triangular2', cycle_momentum=cycle_momentum)
    print('Initialized Cyclic LR scheduler: step_size %d, max_lr %f' % (step_size, max_lr))
    return sche_fn, lr_step