import os
import torch
import yaml
import torch.distributed as dist
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from sugar.utils.logging import get_logger
from sugar.utils.utility import print_dict

__all__ = ['preprocess', 'load_config']


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("--local_rank", type=int, help="ensure code runs on LOCAL_PROCESS_RANK GPU device")
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config

class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


global_config = AttrDict()

def merge_config(config):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in config.items():
        if "." not in key:
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            else:
                global_config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in global_config
            ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                global_config.keys(), sub_keys[0])
            cur = global_config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]

def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    merge_config(yaml.load(open(file_path, 'rb'), Loader=yaml.Loader))
    return global_config

def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = "Config use_gpu cannot be set as true while you are " \
          "using pytorch cpu version ! \nPlease try: \n" \
          "\t1. Install pytorch-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and not torch.cuda.is_available():
            print(err)
            sys.exit(1)
    except Exception as e:
        pass

def preprocess(is_train=False):
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    merge_config(FLAGS.opt)

    # check if set use_gpu=True in torch cpu version
    use_gpu = config['Global']['use_gpu']
    check_gpu(use_gpu)

    alg = config['Architecture']['algorithm']
    assert alg in [
        'Once-for-all'
    ]

    if use_gpu:
        device = FLAGS.local_rank
        torch.cuda.set_device(device)
        FLAGS.world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend='gloo', init_method='env://', world_size=FLAGS.world_size, rank=FLAGS.local_rank)
    else:
        device = 'cpu'

    config['Global']['distributed'] = torch.cuda.device_count() > 1 and dist.is_available()
    if is_train:
        # save_config
        save_model_dir = config['Global']['save_model_dir']
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, 'config.yml'), 'w') as f:
            yaml.dump(
                dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = '{}/train.log'.format(save_model_dir)
    else:
        log_file = None
    logger = get_logger(name='root', log_file=log_file)
    if config['Global']['use_tensorboard']:
        from torch.utils.tensorboard import SummaryWriter
        save_model_dir = config['Global']['save_model_dir']
        tbd_writer_path = '{}/tbd/'.format(save_model_dir)
        os.makedirs(tbd_writer_path, exist_ok=True)
        tbd_writer = SummaryWriter(log_dir=tbd_writer_path)
    else:
        tbd_writer = None
    print_dict(config, logger)
    logger.info('train with torch {} and device {}'.format(torch.__version__,
                                                            device))
    return config, device, logger, tbd_writer

if __name__ == '__main__':
    """
    Example
    -------
    CUDA_VISIBLE_DEVICES="0,1,2,3" /usr/bin/python -m torch.distributed.launch --nproc_per_node=4 \
        /workspace/projects/sugar/tools/program.py -c /workspace/projects/sugar/configs/demo.yml -o Global.use_debug=True
    """
    config = load_config('/workspace/projects/sugar/configs/demo.yml')
    print(config)
    quit()
    
    config, device, logger, tbd_writer = preprocess(is_train=False)
    print(1)