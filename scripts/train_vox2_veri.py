"""
Description: Experiment - Speaker Verification on VoxCeleb2
Author: Rui Wang
Date: 2021-03-19 ~ 2021-04-04

Model:
1. FBanks 80
2. resnet30m/resnet3m
3. AAM-Softmax (margin=0.2, scale=30)

- Training Dataset: voxceleb2 as `train_list.txt`.
- Validate Dataset: A part of voxceleb1 as `veri_test2.txt` (Vox1-O).
- Test Dataset: A part of voxceleb1 as `list_test_all2.txt` (Vox1-E) and `list_test_hard2` (Vox1-H).

It support distributed training and multiprocess validating.
"""

import argparse
import glob
import os
import random
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from sugar.data.voxceleb1 import veriset
from sugar.data.voxceleb2 import veritrain
from sugar.data.augmentation import augset
from sugar.database import AugmentedUtterance, LabeledUtterance, UniqueSampler, DistributedUniqueSampler
from sugar.loss.amsoftmax import AMSoftmax
from sugar.loss.aamsoftmax import AAMSoftmax
from sugar.models import (SpeakerModel, WrappedModel, clf_train,
                          dynamic_walks_train, veri_validate, batch_forward)
from sugar.optimizer.adam import Adam
from sugar.optimizer.sgd import SGD
from sugar.reports import Unbuffered
from sugar.scheduler.steplr import StepLR
from sugar.scheduler.cycliclr import CyclicLR
from sugar.transforms import LogMelFbanks
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import sugar.models.dynamictdnn as models

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ResNet Training for Speaker Verification Task on the VoxCeleb2')

parser.add_argument('--config', type=str, default='', help='Config YAML file') # reserved
parser.add_argument('--debug', action='store_true', dest='debug', help='Enable debug mode using few training data')
parser.add_argument('--logdir', type=str, default='/workspace/projects/sugar/works/nas/exps/exp3/runs/tmp', dest='logdir', help='Directory of log for tensorboard: hyperparameter, training, and validate.')

## model
parser.add_argument('--arch', type=str, default='tdnn8m2g', dest='arch', choices=model_names, help='model architecture')
parser.add_argument('--n_mels', type=int, default=80, help='Number of mel filterbanks')
parser.add_argument('--n_embeds', type=int, default=192, help='Embedding size in the last FC layer')
parser.add_argument('--n_classes', type=int, default=5994, help='Number of speakers in the softmax layer, only for softmax-based losses, e.g., softmax, arcface, cosface.')
parser.add_argument('--lossfunc', type=str, default='AAM-Softmax', choices=['AM-Softmax', 'AAM-Softmax'], help='loss function of AM-Softmax or AAM-Softmax.')
parser.add_argument('--margin', type=float, default=0.2, help='Penalty for AM-Softmax or AAM-Softmax')
parser.add_argument('--scale', type=float, default=30, help='Scale for AM-Softmax or AAM-Softmax')

## Data loader
parser.add_argument('--augment', action='store_true', dest='augment', help='Augment input')
parser.add_argument('--augmode', type=str, default='v2+', choices=['v1', 'v2', 'v2+'], help='Augment input')
parser.add_argument('--second', type=int, default=32000, help='Input length to the network for training and calibrating batch normalization')
parser.add_argument('--eval-second', type=int, default=64000, dest='eval_sec', help='Input length to the network for evaluating')
parser.add_argument('--batch-size', type=int, default=128, dest='batch_size', help='Batch size, number of utterances per batch on every subprocess')
parser.add_argument('--eval-batch-size', type=int, default=8, dest='eval_batch_size', help='Batch size for evaluate/validate, number of utterances per batch')
parser.add_argument('--workers', type=int, default=5, dest='workers', help='Number of loader threads')
parser.add_argument('--sampler', type=str, default='normal', choices=['unique', 'normal'], help='Sampler, works for train data loader')
parser.add_argument('--max-utterances', type=int, default=200, dest='max_utt_per_spk', help='Maximum utterances per speaker for unique sampler')

## Training details
parser.add_argument('--seed', type=int, default=666, help='seed for initializing training.')
parser.add_argument('--masker', action='store_true', dest='masker', help='Enable masking while training')
parser.add_argument('--epochs-softmax', type=int, default=0, dest='epochs_softmax', help='Enable number of softmax-training epochs to pretrained for AM-Softmax or AAM-Softmax')
parser.add_argument('--epochs', type=int, default=36, help='number of total epochs to run')
parser.add_argument('--start-epoch', type=int, default=1, dest='start_epoch', help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', type=int, default=10, dest='print_freq', help='print frequency (default: 10) for both training and validating.')
parser.add_argument('--test_interval', type=int, default=4, help='Enable test every [test_interval] epochs.')
parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint')

## Optimizer
parser.add_argument('--optimizer', type=str, default='adam', dest='learner', choices=['adam', 'sgd'], help='Optimizer')
parser.add_argument('--learning-rate', type=float, default=0.001, dest='lr', help='Learning rate')
parser.add_argument('--lr-softmax', type=float, default=0.001, dest='lr_softmax', help='Learning rate if softmax is available.')
parser.add_argument('--weight-decay', type=float, default=2e-5, dest='weight_decay', help='Weight decay in the optimizer')
parser.add_argument('--scheduler', type=str, default='cycliclr', choices=['steplr', 'cycliclr'], help='learning rate scheduler for training.')
parser.add_argument('--cycle-step', type=int, default=10236, dest='cycle_step', help='step size up of a cycle for cycliclr scheduler, e.g. 10236 for voxceleb2 using 4 normal distributed sampler in 12 epochs of a cycle.')
parser.add_argument('--step-epoch', type=int, default=12, dest='step_epoch', help='epoch of steplr')
parser.add_argument('--lr-decay', type=float, default=0.5, dest='lr_decay', help='lr gamma for steplr')

## Evaluate
parser.add_argument('--validate', action='store_true', dest='validate', help='Enable validate on Vox1-O test set')
parser.add_argument('--initial-weights', type=str, default='', dest='initial_weights',  help='path to latest checkpoint (default: '')')
parser.add_argument('--num-statistics', type=int, default=6000, dest='num_statistics', help='Number of samples to recompute statistics of batchnorm')
parser.add_argument('--batch-size-statistics', type=int, default=128, dest='batch_statistics', help='Batch size to recompute statistics of batchnorm')

## distributed training
parser.add_argument('--port', type=str, default="12355", help='Port for distributed training, input as text')
parser.add_argument('--distributed', action='store_true', dest='distributed', help='Enable distributed training')

## mixed precision training
parser.add_argument('--mixedprec', action='store_true', dest='mixedprec', help='Enable mixed precision training with Gradient Scaler if Tensor Cores is available, but GTX 1080Ti has None.')

## argument for neural architecture search
parser.add_argument('--task', type=str, default='supernet', help='Progressive training task: joint choices=["supernet", "kernel", "depth", "width", "width1", "width2"]')
parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 66, 75, 90], help='Progressive training phase for width')
parser.add_argument('--dynamic-walks', type=int, default=1, dest='dynamic_walks', help='Number of forward in each training step')
parser.add_argument('--specific-tdnn', default=None, dest='subnet_config', help='Fixed architecture, e.g., 3-512-512-512-512-5-5-3-3-1536)')


args = parser.parse_args()
args.model_path = args.logdir.replace('runs/', '')
## argument for progressive training
if len(args.task.split("_")) > 0: # e.g., --task supernet_kernel_depth_width1_width2
    args_tasks = args.task.split("_")
    args.task = [ti for ti in args_tasks if ti != "supernet"]
    args.modes = []
    args.ks = None
    args.ds = None
    args.ws = None
    if "kernel" in args_tasks:
        args.modes.append("kernel")
        args.ks = [1, 3, 5]
    if "depth" in args_tasks:
        args.modes.append("depth")
        args.ds = [-2, -1, 0]
    if "width1" in args_tasks:
        args.modes.append("width1")
        args.ws = [0.5, 0.75, 1]
    if "width2" in args_tasks:
        args.modes.append("width2")
        args.ws = [0.25, 0.35, 0.5, 0.75, 1.0]
else:
    if args.task == 'supernet' or args.validate:
        # update the largest weights
        # second 32000
        # learning rate 0.001
        # epochs 64
        args.modes = ['supernet']
        args.ks = None
        args.ds = None
        args.ws = None
    elif args.task == 'kernel':
        if args.phase == 1:
            # second 48000
            # learning rate 0.001
            # epochs 64
            # dynamic walk 1
            args.modes = ['kernel']
            args.ks = [1, 3, 5]
            args.ds = [0]
            args.ws = [1]
        elif args.phase == 66:
            # second 48000
            # learning rate 0.001
            # epochs 64
            # dynamic walk 1
            args.modes = ['kernel']
            args.ks = [3, 5]
            args.ds = [0]
            args.ws = [1]
    elif args.task == 'depth':
        if args.phase == 1:
            # second 48000
            # learning rate 0.001
            # epochs 64
            # dynamic walk 1
            args.modes = ['kernel', 'depth']
            args.ks = [1, 3, 5]
            args.ds = [-2, -1, 0]
            args.ws = [1]
        elif args.phase == 2:
            # second 48000
            # learning rate 0.001
            # epochs 64
            # dynamic walk 1
            args.modes = ['kernel', 'depth']
            args.ks = [3, 5]
            args.ds = [-1, 0]
            args.ws = [1]
        elif args.phase == 66:
            # second 48000
            # learning rate 0.001
            # epochs 64
            # dynamic walk 1
            args.modes = ['kernel', 'depth']
            args.ks = [3, 5]
            args.ds = [-2, -1, 0]
            args.ws = [1]
        elif args.phase == 75:
            # second 48000
            # learning rate 0.001
            # epochs 64
            # dynamic walk 1
            args.modes = ['kernel', 'depth']
            args.ks = [3, 5]
            args.ds = [-3, -2, -1, 0]
            args.ws = [1]
    elif args.task == 'width':
        args.model_path = os.path.join(args.model_path, 'phase%d' % args.phase)
        if args.phase == 1:
            # second 48000
            # learning rate 0.001
            # epochs 64
            # dynamic walk 1
            args.modes = ['kernel', 'depth', 'width1']
            args.ks = [1, 3, 5]
            args.ds = [-2, -1, 0]
            args.ws = [0.5, 0.75, 1]
        elif args.phase == 2:
            # second 48000
            # learning rate 0.001
            # epochs 64
            # dynamic walk 1
            args.modes = ['kernel', 'depth', 'width1', 'width2']
            args.ks = [1, 3, 5]
            args.ds = [-2, -1, 0]
            args.ws = [0.25, 0.35, 0.5, 0.75, 1.0]
        elif args.phase == 66: # best wish for clever widths
            # second 48000
            # learning rate 0.001
            # epochs 64
            # dynamic walk 1
            args.modes = ['kernel', 'depth', 'width']
            args.ks = [1, 3, 5]
            args.ds = [-2, -1, 0]
            args.ws = [0.66, 1]
        elif args.phase == 75: # best wish for clever widths
            # second 48000
            # learning rate 0.001
            # epochs 64
            # dynamic walk 1
            args.modes = ['kernel', 'depth', 'width']
            args.ks = [1, 3, 5]
            args.ds = [-2, -1, 0]
            args.ws = [0.75, 1]
        elif args.phase == 90: # best wish for clever widths
            # second 48000
            # learning rate 0.001
            # epochs 64
            # dynamic walk 1
            args.modes = ['kernel', 'depth', 'width']
            args.ks = [1, 3, 5]
            args.ds = [-2, -1, 0]
            args.ws = [0.90, 1]


def save_checkpoint(state, args, is_latest=False):
    """Save current checkpoint and the best accuracy model."""
    if is_latest:
        save_path = os.path.join(args.model_path, 'latest.pth.tar')
    else:
        save_path = os.path.join(args.model_path, 'checkpoint%06d.pth.tar' % args.epoch)
    torch.save(state, save_path)

def reset_lr(scheduler, lr):
    """Reset learning rate of Optimizer to the given value."""
    for param_group in scheduler.optimizer.param_groups:
        param_group['lr'] = lr


def main_worker(gpu, ngpus_per_node, args):
    global tboard
    stage_file = open(args.stage_log, 'a')
    sys.stdout = Unbuffered(sys.stdout, stage_file)

    args.gpu = gpu
    print("Use GPU: {} for {}".format(args.gpu, 'training' if not args.validate else 'validating'))

    ## create model
    assert args.arch in ['tdnn8m2g', 'tdnn16m4g']
    modelarch = models.__dict__[args.arch](args.n_mels, args.n_embeds, args.modes, ks=args.ks, ds=args.ds, ws=args.ws)
    
    if args.lossfunc == 'AM-Softmax':
        lossfunc  = AMSoftmax(args.n_embeds, args.n_classes, margin=args.margin, scale=args.scale)
    else:
        assert args.lossfunc == 'AAM-Softmax'
        lossfunc  = AAMSoftmax(args.n_embeds, args.n_classes, margin=args.margin, scale=args.scale)
    
    transform = LogMelFbanks(n_mels=args.n_mels, masker=args.masker)
    evaltrans = LogMelFbanks(n_mels=args.n_mels).cuda(args.gpu)
    model = SpeakerModel(modelarch, lossfunc, transform, keep_channel='resnet' in args.arch, features_last=False)
    
    ## initial distributed environment in single machine
    if args.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)
        torch.cuda.set_device(args.gpu)
        if not args.validate:
            model.cuda(args.gpu)
            model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model = WrappedModel(model).cuda(args.gpu)
    else:
        model = WrappedModel(model).cuda(args.gpu)

    ## load initial weights
    if not args.validate and args.initial_weights:
        if os.path.isfile(args.initial_weights):
            print("=> loading weights '{}'".format(args.initial_weights))
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.initial_weights, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded weights '{}'".format(args.initial_weights))
        elif not args.validate:
            print("=> no weights found at '{}'".format(args.initial_weights))

    ## Fixed architecture
    if args.subnet_config:
        subnet_config = args.subnet_config.strip().split('-')
        subnet_config = [int(c) for c in subnet_config]
        subnet_config = (
            subnet_config[0], 
            subnet_config[1 : subnet_config[0] + 2],
            subnet_config[subnet_config[0] + 2 : 2 * subnet_config[0] + 3], 
            subnet_config[2 * subnet_config[0] + 3]
        )
        if args.initial_weights:
            subnet = model.module.__S__.clone(subnet_config)
            model.module.add_module('__S__', subnet)
        else:
            scratchnet = models.specfic_tdnn(subnet_config)
            model.module.add_module('__S__', scratchnet)
        
    print(model)
    
    ## prepare data for training
    if not args.validate:
        train, spks = veritrain(num_samples=args.second)

        if args.augment is True:
            aug_wav  = augset(num_samples=args.second, seed=args.seed + (args.gpu if args.gpu is not None else 0))
            trainset = AugmentedUtterance(train, spks, augment=aug_wav, mode=args.augmode)
        else:
            trainset = LabeledUtterance(train, spks)

        ## debug
        if args.debug:
            trainset.datalst = trainset.datalst[:1000]

        if args.sampler == 'unique':
            train_sampler = DistributedUniqueSampler(trainset.label_dict, args.batch_size, args.max_utt_per_spk) if args.distributed else UniqueSampler(trainset.label_dict, args.batch_size, args.max_utt_per_spk)
        else:
            assert args.sampler == 'normal'
            train_sampler = DistributedSampler(trainset, seed=args.seed) if args.distributed else None

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, 
            pin_memory=True, drop_last=True, sampler=train_sampler
        )
        print('The number of batchs of Training Loader:', len(train_loader))

    ## load augmented data for recomputed the statistic of batchnorm
    bnset, spks = veritrain(num_samples=args.second)
    random.shuffle(bnset.datalst)
    bnset.datalst = bnset.datalst[:args.num_statistics]
    aug_wav = augset(num_samples=args.second, seed=args.seed)
    aug_bnset = AugmentedUtterance(bnset, spks, augment=aug_wav, mode=args.augmode)
    batchnorm_loader = torch.utils.data.DataLoader(
        aug_bnset, batch_size=args.batch_statistics, shuffle=True, num_workers=args.workers, drop_last=True
    )

    ## prepare data for evaluate
    assert not (args.eval_sec == 0 and args.eval_batch_size > 1)
    veri_test2, _, _, _ = veriset(all2=None, hard2=None, num_samples=args.eval_sec, num_eval=2)
    val_loader = torch.utils.data.DataLoader(veri_test2, batch_size=args.eval_batch_size, num_workers=1)
    
    ## if validate
    if args.validate:
        eval_file  = open(args.evaluate_log, 'a')
        model.module.__T__.masker = False

        weights_lst  = [args.initial_weights]
        if os.path.isdir(args.initial_weights):
            weights_lst = glob.glob(os.path.join(args.initial_weights, '*'))
            weights_lst.sort()
        ngpus_per_node = 1 if not args.distributed else ngpus_per_node
        for i in range(args.gpu, len(weights_lst), ngpus_per_node):
            weights_path = weights_lst[i]

            print("=> loading weights '{}'".format(weights_path))
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(weights_path, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded weights '{}'".format(weights_path))

            if args.task in ['kernel', 'depth', 'width']:
                batch_forward(batchnorm_loader, model, args=args)

            eer, dcf = veri_validate(val_loader, model, args=args)
            eval_file.write('Evaluate on Validate @{}: * EER {eer:.3f}% DCF {dcf:.3f}\n'.format(
                os.path.basename(weights_path), eer=eer[0] * 100, dcf=dcf[0])
            )
            eval_file.flush()
        eval_file.close()
        return
    
    ## define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    learning_rate = args.lr if args.epochs_softmax == 0 else args.lr_softmax

    assert args.learner in ['adam', 'sgd']
    if args.learner == 'adam':
        optimizer = Adam(model.parameters(), learning_rate, args.weight_decay)
    else:
        optimizer = SGD(model.parameters(), learning_rate, args.weight_decay)

    assert args.scheduler in ['steplr', 'cycliclr']
    if args.scheduler == 'steplr':
        scheduler, args.lr_step = StepLR(optimizer, args.step_epoch, args.lr_decay)
    else:
        scheduler, args.lr_step = CyclicLR(optimizer, args.lr, args.cycle_step)
    scaler = GradScaler() if args.mixedprec else None

    ## (optionally) resume from a checkpoint
    if os.path.exists(os.path.join(args.model_path, 'latest.pth.tar')):
        # resume the latest checkpoint from model directory
        args.resume = os.path.join(args.model_path, 'latest.pth.tar')
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
            args.train_step = checkpoint['train_step'] if 'train_step' in checkpoint else args.start_epoch * len(train_loader)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.lr_step = checkpoint['lr_step']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ## set step of train and validate/test
    if not hasattr(args, "train_step"):
        args.train_step = 0
        args.epoch = 0

    ## start train
    for epoch in range(args.start_epoch, args.epochs + 1):
        args.epoch = epoch

        if args.distributed:
            train_sampler.set_epoch(epoch)

        if isinstance(model.module.__L__, (AMSoftmax, AAMSoftmax)):
            if args.epoch <= args.epochs_softmax:
                # epochs for Softmax loss
                model.module.__L__.soft()
            else: 
                if args.epochs_softmax > 0 and args.epoch == (args.epochs_softmax + 1):
                    # reset learning rate for AM-Softmax
                    reset_lr(scheduler, args.lr)
                model.module.__L__.notsoft()

        # train for one epoch
        if 'supernet' in args.task:
            clf_train(train_loader, model, criterion, optimizer, args, scheduler=scheduler, tboard=tboard, scaler=scaler)
        else: # ['kernel', 'depth', 'width'] any in args.task:
            dynamic_walks_train(train_loader, model, criterion, optimizer, args, scheduler=scheduler, tboard=tboard, scaler=scaler)
        args.train_step += len(train_loader)

        # save latest checkpoint
        if args.gpu == 0:
            save_checkpoint({
                'epoch': epoch,
                'train_step': args.train_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'lr_step': args.lr_step
            }, args, is_latest=True)

        # evaluate on validation set: cost less than 2 minutes on identification
        if epoch % args.test_interval == 0 or epoch == args.epochs:

            # Perform evaluation during training, some tasks require extanding sub-network mode to supernet.
            # Take care of .backward to model's buffer, such as BatchNorm running mean/var.
            # cloned = model.module.__S__.clone([3, 4, 6, 3], [1.0 for _ in range(16)], [5 for _ in range(16)])
            if 'supernet' not in args.task:
                cloned = model.module.__S__.clone()
                cloned.set_static()
                cloned = WrappedModel(SpeakerModel(cloned, None, evaltrans, keep_channel='resnet' in args.arch, features_last=False))
                batch_forward(batchnorm_loader, cloned, args=args)
            else:
                cloned = model
            
            eer, dcf = veri_validate(val_loader, cloned, args=args, tboard=tboard)
            print(time.strftime('%Y-%m-%d %H:%M:%S %Z:', time.localtime(time.time())), 'GPU {} Epoch: [{}] * EER {eer:.3f}% DCF {dcf:.3f}'.format(args.gpu, epoch, eer=eer[0] * 100, dcf=dcf[0]))
            
            # save checkpoint
            if args.gpu == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'train_step': args.train_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'lr_step': args.lr_step
                }, args)


def main():

    assert args.n_mels == 80 and args.n_embeds == 192, f'{args.n_mels} != 80, {args.n_embeds} != 192'

    args.train_dir = os.path.join(os.path.dirname(os.path.dirname(args.logdir)), 'logs')
    args.stage_log = os.path.join(args.train_dir, os.path.basename(args.logdir) + '.log')
    args.train_log = os.path.join(args.train_dir, os.path.basename(args.logdir) + '.train.log')
    args.evaluate_log = args.train_log.replace('train', 'evaluate')

    stage_file = open(args.stage_log, 'a')
    sys.stdout = Unbuffered(sys.stdout, stage_file)

    print(time.strftime('%Y-%m-%d %H:%M:%S %Z:', time.localtime(time.time())), ' '.join(sys.argv))
    print(vars(args))

    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.train_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    n_gpus = torch.cuda.device_count()
    
    print('Python Version:\t', sys.version.replace('\n', ''))
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:\t', n_gpus)
    print('Model directory:\t', args.model_path)
    print('Logging directory:\t', args.logdir)
    print('Train directory:\t', args.train_dir)
    print('Save stage path:\t', args.stage_log)
    print('Save train path:\t', args.train_log)
    print('Save evaluate path:\t', args.evaluate_log)

    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)

    print(time.strftime('%Y-%m-%d %H:%M:%S %Z:', time.localtime(time.time())), "Finish Job.")
    

## global variables
tboard = SummaryWriter(args.logdir)
torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    main()
