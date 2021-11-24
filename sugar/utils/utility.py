"""
Practical Tools.

Author: Wang Rui
Date: 2021.03.10 ~
"""
import torch

def bn_state_dict(model, removed_prefix=""):
    """Extract bn_state_dict from model.
    
        Parameters
        ----------
            model : torch.nn.Module
    """
    from collections import OrderedDict
    from torch import nn
    bn = OrderedDict()
    bn._metadata = OrderedDict()
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.state_dict(bn, prefix=name.replace(removed_prefix, "") + '.')
    return bn


def load_bn_state_dict(model, state_dict):
    """Load bn_state_dict into model.
    
        Parameters
        ----------
            model : torch.nn.Module
            state_dict : collections.OrderedDict
    """
    self_state = model.state_dict()
    unloaded = []
    for name, param in state_dict.items():
        if name in self_state and self_state[name].size() == state_dict[name].size():
            self_state[name].copy_(param)
        else:
            unloaded.append(name)
    assert len(unloaded) == 0, f"{unloaded} NOT Loaded!"
    model.load_state_dict(self_state)


def print_size_of_model(model):
    """Count the size of saved model.

        Parameters
        ----------
            model : torch.nn.Module

        Notes
        -----
            'temp.p' is used.
    """
    import os
    if isinstance(model, dict):
        torch.save(model, "temp.p")
    else:
        torch.save(model.state_dict(), "temp.p")
    model_size = os.path.getsize("temp.p")/(1.0*1024*1024)
    print('Size (MB):', model_size)
    os.remove('temp.p')
    return model_size


def keep_params(state_dict_path, params_path):
    import torch, os
    assert os.path.exists(state_dict_path) and not os.path.exists(params_path)
    state_dict = torch.load(state_dict_path, 'cpu')
    assert 'state_dict' in state_dict
    del state_dict['optimizer']
    del state_dict['scheduler']
    del state_dict['lr_step']
    torch.save(state_dict, params_path)


def list2dict(log, typ=None, nokernel1=False):
    """
    Convert the list of evaluate log to dict with the key of eer, dcf, depth, kernel, width, config, type.
    """
    import numpy as np

    def log2list(log):
        lines = []
        with open(log, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.split(' ')
                lines.append([float(v) for v in line[:2]] + [int(v)
                             for v in line[2:]])
        return lines

    lst = log2list(log)

    if nokernel1:
        # remove kernel == 1
        lst = [v for v in lst if 1 not in v[4:][v[4] + 2: 2 * v[4] + 3]]

    eer = [v[0] for v in lst]
    dcf_0_01 = [v[1] for v in lst]
    macs = [v[2] for v in lst]
    params = [v[3] for v in lst]
    confs = [v[4:] for v in lst]

    print('Min/Max EER {}% / {}%'.format(min(eer), max(eer)))
    print('Min/Max DCF {} / {}'.format(min(dcf_0_01), max(dcf_0_01)))
    print('Min/Max MACs {:,} / {:,}'.format(min(macs), max(macs)))
    print('Min/Max Params {:,} / {:,}'.format(min(params), max(params)))

    rdict = {'eer': np.array(eer),
             'dcf': np.array(dcf_0_01),
             'macs': np.array(macs),
             'params': np.array(params),
             'depth': np.array([v[0] for v in confs]),
             'width': np.array([v[1] for v in confs]),
             'kernel': np.array([v[v[0] + 2] for v in confs]),
             'config': confs}
    if typ is not None:
        rdict['type'] = np.array([typ for _ in range(len(eer))])

    return rdict


def print_dict(d, logger, delimiter=0):
    """
    Recursively visualize a dict and
    indenting acrrording by the relationship of keys.
    """
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            print_dict(v, logger, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logger.info("{}{} : ".format(delimiter * " ", str(k)))
            for value in v:
                print_dict(value, logger, delimiter + 4)
        else:
            logger.info("{}{} : {}".format(delimiter * " ", k, v))

def flatten(lst):
    """
    Recursively flatten a tuple or list.
    """
    return sum(([x] if not isinstance(x, (list, tuple)) else flatten(x) for x in lst), [])