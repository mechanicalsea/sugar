#!/usr/bin/python
# -*- coding: utf-8 -*-

from .elasticresnet import resnet3m, resnet10m, resnet30m
from .dynamictdnn import specfic_tdnn, ecapa_tdnn, tdnn8m2g, tdnn14m4g, tdnn16m4g, tdnn26m7g

"""
Speaker Network has 4 methods:
    1. training
    2. evaluation
    3. save weights
    4. load weights
    5. support log in format of txt and tensorboard.
"""

import os
import time
import torch
import torch.nn as nn
from sugar.vectors import extract_vectors
from torch.cuda.amp import autocast
from sugar.transforms import LogMelFbanks
from sugar.reports import AverageMeter, ProgressMeter
from sugar.metrics import accuracy, calculate_mindcf, calculate_eer
from sugar.vectors import extract_vectors
from sugar.scores import score_trials
from sugar.utils import utility

import sys
if any('jupyter' in arg for arg in sys.argv):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def count_parameters(net, verbose=0):
    if verbose > 0:
        for name, p in net.named_parameters():
            if p.requires_grad:
                print(name, p.numel())
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_params

class SpeakerModel(nn.Module):
    def __init__(self, modelarch, lossfunc=None, transform=None, keep_channel=False, features_last=False):
        """
        Parameter
        ---------
            modelarch : torch.nn.Module
                Model extracting speaker embeddings from a speech signal
            lossfunc : torch.nn.Module
                Loss function
            transform : torch.nn.Module
                Transform feeding the Model. If transoform is None, the model enable the input in form of waveforms.
            keep_channel : bool
                Enable the dimension of (batch, channel, feature, frame) rather than (batch, feature, frame).
            features_last : bool
                Make the dimension of feature after frame.
        """
        super(SpeakerModel, self).__init__()
        self.__S__ = modelarch  # model to extract embeddings
        self.__L__ = lossfunc  # loss to optimize weights
        self.__T__ = transform
        self.keep_channel = keep_channel # enable inputs like single-channel images
        self.features_last = features_last # features_last inputs allow some model, like some implementation of TDNN

    def __repr__(self):
        arch_params = count_parameters(self.__S__)
        if self.__L__ is not None: 
            loss_params = count_parameters(self.__L__)
        else:
            return 'Arch: %d' % (arch_params)
        return 'Arch: %d, Loss: %d' % (arch_params, loss_params)

    def classifier(self, embeddings, label=None):
        """Support the function in the speaker identification task."""
        return self.__L__.forward(embeddings, label)

    def forward(self, batch, label=None):
        """
        Support (1) classfication objective, (2) and end-to-end objective.            
            (1) batch (batch_size, 1, wav_length/feature_dim),
            (2) batch (batch_size, 2+, wav_length/feature_dim).
            (3) batch (batch_size, wav_length/feature_dim)
            where wav_length has 1-dim, feature_dim has 2-dim
        """
        sizes = list(batch.size())
        if len(sizes) == 2 or (len(sizes) == 3 and sizes[1] != 1):
            batch = batch.unsqueeze(1)
            sizes = list(batch.size())
        batch_size = sizes[0]
        utts_size = sizes[1]
        _sizes = [-1] + sizes[2:]
        batch = batch.reshape(_sizes)

        if self.__T__ is not None:
            batch = self.__T__.forward(batch)

        if not self.keep_channel: # (batch, 1, feat_num, seq_len) -> (batch, feat_num, seq_len)
            batch = batch.squeeze(1)

        if self.features_last: # (batch, 1, feat_num, seq_len) -> (batch, seq_len, feat_num)
            batch = batch.transpose(1, 2)

        embeds = self.__S__.forward(batch)

        if label is None or self.__L__ is None:
            return embeds
        else:
            embeds_size = embeds.size()[-1]
            embeds = embeds.reshape(
                [batch_size, utts_size, embeds_size]).squeeze(1)
            out = self.__L__.forward(embeds, label)
            return out


class WrappedModel(nn.Module):
    """
    The purpose of this wrapper is to make the model structure consistent between single and multi-GPU,
        where the `module` is in DistributedDataParallel model. In this wrapper, we crate module for it.
    """
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)

    @classmethod
    def from_pretrained(
        cls,
        repo_id="mechanicalsea/efficient-tdnn",
        supernet_filename="depth/depth.torchparams",
        subnet_filename="depth/depth.ecapa-tdnn.3.512.512.512.512.5.3.3.3.1536.bn.tar",
        feat_type="logmelfbank",
        feat_dim=80,
        model_type="tdnn8m2g",
        embed_dim=192,
    ):
        from huggingface_hub import hf_hub_download
        supernet_path = hf_hub_download(repo_id=repo_id, filename=supernet_filename)
        sup_state_dict = torch.load(supernet_path, map_location='cpu')['state_dict']
        if feat_type == "logmelfbank":
            transform = LogMelFbanks(n_mels=feat_dim)
        else:
            raise ValueError(f"{feat_type} not in {['logmelfbank']}")
        if model_type == "tdnn8m2g":
            modelarch = tdnn8m2g(in_feats=feat_dim, out_embeds=embed_dim)
        else:
            raise ValueError(f"{model_type} not in {['tdnn8m2g']}")
        model = SpeakerModel(modelarch, transform=transform)
        model = cls(model)
        model.load_state_dict(sup_state_dict, strict=False)
        if subnet_filename not in ["", None, '']:
            subnet_bn_path = hf_hub_download(repo_id=repo_id, filename=subnet_filename)
            sub_state_dict = torch.load(subnet_bn_path, map_location='cpu')
            subnet_config = sub_state_dict['subnet']
            subnet = model.module.__S__.clone(subnet_config)
            subnet.load_state_dict(sub_state_dict['bn'], strict=False)
            subnet = cls(SpeakerModel(subnet, transform=transform))
        else:
            subnet = subnet_bn_path = sub_state_dict = None
        info = {'supernet_file': supernet_path, 'supernet': sup_state_dict, 'supernet_model': model,
                'subnet_file': subnet_bn_path, 'subnet': sub_state_dict}
        return subnet, info


def checkpoint2backbone(model_path):
    """
    Load checkpoint and Save backbone.

        Parameters
        ----------
            model_path : str
                The path of model's state dict than includes `state_dict` which is backbone.
        
        Notes
        -----
            The backbone weights are saved at the same directory with the prefix of `backbone`.
    
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    state_dict = torch.load(model_path, map_location='cpu')

    if 'state_dict' not in state_dict.keys():
        raise ValueError('state_dict not in {}'.format(state_dict.keys()))
    
    backbone_state_dict = {}
    for key, value in state_dict['state_dict'].items():
        if '__S__' in key:
            k = key.replace('module.__S__.', '')
            backbone_state_dict[k] = value
    
    backbone_name = '.'.join(['backbone'] + os.path.basename(model_path).split('.')[1:])
    backbone_path = os.path.join(os.path.dirname(model_path), backbone_name)
    torch.save(backbone_state_dict, backbone_path)


def batch_forward(loader, model, args=None, device='cpu'):
    """
    Forward the model on the dataloader. It is helpful to the sampled architecture from a supernet in field of
        neural architecture search. Specifically, it helps to recompute statistics of batchnorm.

        Parameter
        ---------
            loader : torch.utils.data.DataLoader
                Data loader with training data
            model : torch.nn.Module
                Model including the architecture extracting embeddings from speech signals
            args : parser.parse_args()
                require args.gpu
            device : str
                Device to compute. 

        Note
        ----
            loader : keep it small and functional.
    """
    if args is not None and 'gpu' in vars(args) and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    
    with torch.no_grad():
        origin_device = 'cpu'
        if len(list(model.parameters())):
            origin_device = model.parameters().__next__().device
        origin_state  = model.training

        model.train()
        model.to(device)
        for batch in tqdm(loader, desc='Forward Model', total=len(loader)):

            inp = batch[0]
            inp = inp.to(device)

            # forward
            model.forward(inp, None)

        model.train(origin_state)
        if len(list(model.parameters())):
            model.to(origin_device)


def clf_train(loader, model, criterion, optimizer, args, scheduler=None, scaler=None, tboard=None):
    """
    Training the model in a epoch in a discriminant training method that classify categories.

        Parameter
        ---------
            loader : torch.utils.data.DataLoader
            model : torch.nn.Module
            criterion : torch.nn.CrossEntropyLoss
            optimizer : torch.nn.Optimizer
            args : parser.parse_args()
            scheduler : torch.nn.Scheduler
            scaler : torch.cuda.amp.GradScaler
                Enable Gradient Scaling fo mix presion training to prevent underflow.
            tboard : torch.utils.tensorboard.SummaryWriter

        Note
        ----
            (1) Print process with top1, top5, loss, lr, [batch time, and data time]
            (2) Equip with mixed precision training with `args.mixedprec`
    """
    if args.gpu is not None:
        name = 'Cuda:{}'.format(args.gpu)
    else:
        name = 'NoCuda'

    if tboard is None:
        losses = AverageMeter('Loss', ':.4f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f') 
        progress = ProgressMeter(
            len(loader),
            [losses, top1, top5],
            prefix='Epoch: [{}]'.format(args.epoch))

    model.train()

    log_file  = open(args.train_log, 'a')
    pbar      = tqdm(enumerate(loader), desc='Clf. Training - Epoch %06d' % (args.epoch), mininterval=10.0, miniters=args.print_freq, file=log_file)
    for i, (inp, lab) in pbar:

        lab = torch.LongTensor(lab)
        if args.gpu is not None and torch.cuda.is_available(): 
            inp = inp.cuda(args.gpu, non_blocking=True)
            lab = lab.cuda(args.gpu, non_blocking=True)

        # clear gradients
        optimizer.zero_grad()

        # mixed precision training or not
        if args.mixedprec is True:
            with autocast(enabled=True):
                # forward
                outp = model.forward(inp, lab)
                loss = criterion(outp, lab)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(outp, lab, topk=(1, 5))
            # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # forward
            outp = model.forward(inp, lab)
            loss = criterion(outp, lab)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(outp, lab, topk=(1, 5))
            # backward
            loss.backward()
            optimizer.step()

        # print process
        if i % args.print_freq == 0 or i == len(loader) - 1:
            if tboard is None:
                losses.update(loss.item(), inp.size(0))
                top1.update(acc1.item(), inp.size(0))
                top5.update(acc5.item(), inp.size(0))
                progress.display(i + 1)
            else:
                ploss = loss.item()
                pacc1 = acc1.item()
                pacc5 = acc5.item()
                post_dict = {"Loss": ploss,
                             "Acc1": pacc1}
                tboard.add_scalars('Loss/train', {name: ploss}, args.train_step + i)
                tboard.add_scalars('Accuracy/train/Acc1', {name: pacc1}, args.train_step + i)
                tboard.add_scalars('Accuracy/train/Acc5', {name: pacc5}, args.train_step + i)
                if scheduler is not None:
                    post_dict["lr"] = scheduler.get_last_lr()[0]
                    tboard.add_scalars('Loss/train/lr', {name: scheduler.get_last_lr()[0]}, args.train_step + i)
                pbar.set_postfix(post_dict)

        if scheduler is not None and args.lr_step == 'step':
            scheduler.step()

    if scheduler is not None and args.lr_step == 'epoch':
        scheduler.step()
    
    log_file.close()

def dynamic_walks_train(loader, model, criterion, optimizer, args, scheduler=None, scaler=None, tboard=None, verbose=False):
    """
    Support to train an elastic/dynamic archtecture via multiple path in single path one-shot method [1].
        [1] Guo, Z., et al. Single Path One-Shot Neural Architecture Search with Uniform Sampling, ECCV 2020 pp. 544–560. https://doi.org/10.1007/978-3-030-58517-4_32

        Parameter
        ---------
            loader : torch.utils.data.DataLoader
            model : torch.nn.Module or WrappedModel
            criterion : torch.nn.CrossEntropyLoss
            optimizer : torch.nn.Optimizer
            args : parser.parse_args()
                args.dynamic_walks indicates the number of forward in a step.
            scheduler : torch.nn.Scheduler
            scaler : torch.cuda.amp.GradScaler
                Enable Gradient Scaling fo mix presion training to prevent underflow.
            tboard : torch.utils.tensorboard.SummaryWriter
            verbose : bool
                Enable to something diagnose if necessary.

        Note
        ----
            There is question why the model of large dynamic cost more time in DDP training.
    """
    if args.gpu is not None:
        name = 'Cuda:{}'.format(args.gpu)
    else:
        name = 'NoCuda'

    if tboard is None:
        losses = AverageMeter('Loss', ':.4f')
        top1 = AverageMeter('Acc@1', ':6.2f') # EER or accuracy
        top5 = AverageMeter('Acc@5', ':6.2f') 
        progress = ProgressMeter(
            len(loader),
            [losses, top1, top5],
            prefix='Epoch: [{}]'.format(args.epoch))

    model.train()

    log_file  = open(args.train_log, 'a')
    pbar      = tqdm(enumerate(loader), desc='Clf. Dynamic Path Training - Epoch %06d' % (args.epoch), mininterval=10.0, miniters=args.print_freq, file=log_file)

    for i, (inp, lab) in pbar:

        lab = torch.LongTensor(lab)
        if args.gpu is not None and torch.cuda.is_available(): 
            inp = inp.cuda(args.gpu, non_blocking=True)
            lab = lab.cuda(args.gpu, non_blocking=True)

        # clear gradients
        optimizer.zero_grad()

        # dynamic path
        loss_subnets = 0
        acc1_subnets = 0
        acc5_subnets = 0

        for i_ in range(args.dynamic_walks):
            # mixed precision training or not
            if args.mixedprec is True:
                with autocast(enabled=True):
                    outp = model.forward(inp, lab)
                    loss = criterion(outp, lab)
                    acc1, acc5 = accuracy(outp, lab, topk=(1, 5))
                scaler.scale(loss).backward()
            else:
                # forward
                outp = model.forward(inp, lab)
                # measure loss
                loss = criterion(outp, lab)
                # measure accuracy
                acc1, acc5 = accuracy(outp, lab, topk=(1, 5))
                # backward at multiple times that aggregate gradients
                loss.backward()

            # record forward path
            if tboard and getattr(model.module.__S__, 'config', None):
                forward_path = model.module.__S__.config
                forward_path = ', '.join([str(pi) for pi in utility.flatten(forward_path)])
                tboard.add_text(f'GPU:[{name}] Step:[{args.train_step + i}-{i_}]', forward_path, args.train_step + i)

            # accumulate loss and accuracy
            if i % args.print_freq == 0 or i == len(loader) - 1:
                loss_subnets += loss.item()
                acc1_subnets += acc1.item()
                acc5_subnets += acc5.item()
        
        # update gradients
        if args.mixedprec is True:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # print process
        if i % args.print_freq == 0 or i == len(loader) - 1:
            if tboard is None:
                losses.update(loss_subnets/args.dynamic_walks, inp.size(0))
                top1.update(acc1_subnets/args.dynamic_walks, inp.size(0))
                top5.update(acc5_subnets/args.dynamic_walks, inp.size(0))
                progress.display(i + 1)
            else:
                post_dict = {"Loss": loss_subnets/args.dynamic_walks,
                             "Acc1": acc1_subnets/args.dynamic_walks}
                tboard.add_scalars('Loss/train', {name: loss_subnets/args.dynamic_walks}, args.train_step + i)
                tboard.add_scalars('Accuracy/train/Acc1', {name: acc1_subnets/args.dynamic_walks}, args.train_step + i)
                tboard.add_scalars('Accuracy/train/Acc5', {name: acc5_subnets/args.dynamic_walks}, args.train_step + i)
                # tboard.add_text
                if scheduler is not None:
                    post_dict["lr"] = scheduler.get_last_lr()[0]
                    tboard.add_scalars('Loss/train/lr', {name: scheduler.get_last_lr()[0]}, args.train_step + i)
                pbar.set_postfix(post_dict)
        
        if scheduler is not None and args.lr_step == 'step':
            scheduler.step()

    if scheduler is not None and args.lr_step == 'epoch':
        scheduler.step()

def iden_validate(loader, model, criterion=None, args=None, tboard=None):
    """
    Validate the model in the speaker identification task. It works for the close-set identification task, 
        where the identification is implemented by a classifier.

        Parameter
        ---------
            loader : sugar.database.LabeledUtterance
                A dataset of labeled utterances via torch.utils.data.Dataset.
            model : SpeakerModel or WrappedModel
                A model includes input transform, embedding extractor, and loss function.
            criterion : torch.nn.Module [Deprecated]
                A criterion is used to compute loss, such as cross entropy.
            args : argparse.Namespace
                Hyperparameters for validate.
            tboard : torch.utils.tensorboard.SummaryWriter
                Tensorboard interface is used to record process.

        Return
        ------
            top1: float
            top5: float
    """
    model.eval()
    vectors = {}

    val_task = 'test' if 'evaluate' in vars(args) and args.evaluate is True else 'validate'
    device = 'cpu' if args.gpu is None else 'cuda:{}'.format(args.gpu)

    # extract vectors from the model and inputs. Note that embeddings without normalization outperform that with it.
    with torch.no_grad():
        vectors = extract_vectors(model, loader.dataset.files, loader.dataset.read, norm=False, device=device,
                                  n_jobs=loader.num_workers, batch_size=loader.batch_size)
                                  
    # evaluate on the dataset
    with torch.no_grad():
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(loader), [top1, top5], prefix='Test:')

        if args.gpu is not None:
            name = 'Cuda:{}'.format(args.gpu)
        else:
            name = 'NoCuda'

        for i, (path, lab) in tqdm(enumerate(loader), total=len(loader), desc='Validate Identification'):

            lab = torch.LongTensor(lab)
            if args.gpu is not None and torch.cuda.is_available(): 
                lab = lab.cuda(args.gpu, non_blocking=True)

            # forward
            embed = torch.cat([vectors[file].mean(0, keepdim=True) for file in path], 0) # average embeddings
            outp = model.module.classifier(embed, lab)
            
            # measure accuracy 
            acc1, acc5 = accuracy(outp, lab, topk=(1, 5))
            top1.update(acc1.item(), lab.size(0))
            top5.update(acc5.item(), lab.size(0))
            
            # print process
            if i % args.print_freq == 0 or i == len(loader) - 1:
                if tboard is None:
                    progress.display(i + 1)
        
    if tboard is not None:
        tboard.add_scalars('Metric/%s/Acc1' % val_task, {name: top1.avg}, args.epoch)
        tboard.add_scalars('Metric/%s/Acc5' % val_task, {name: top5.avg}, args.epoch)
    return top1.avg, top5.avg

def veri_validate(loader, model, args=None, tboard=None, norm=True, p_target=0.05, device='cpu', mode='cosine', ret_info=False, vectors=None):
    """
    Validate the model in the speaker verification task.

        Parameter
        ---------
            loader : sugar.database.VerificationTrials
                A dataset of trials via torch.utils.data.Dataset.
            model : SpeakerModel or WrappedModel
                A model includes input transform, embedding extractor, and loss function.
            args : argparse.Namespace
                Hyperparameters for validate.
            tboard : torch.utils.tensorboard.SummaryWriter
                Tensorboard interface is used to record process.
            norm : bool
                L2 Normalized speaker vectors or not.
            p_target : float
                0.05 for NIST SRE 2016 DCF, 0.1 for ? (unsured)
            mode : str
                ['cosine', 'norm2']
            ret_info : bool
                If True, return additional scores, vectors

        Return
        ------
            eer: float
            dcf: float
    """
    if args is not None:
        val_task = 'test' if 'evaluate' in vars(args) and args.evaluate is True else 'validate'
        device = 'cpu' if args.gpu is None else 'cuda:{}'.format(args.gpu)
    else:
        val_task = 'evaluate'
        device = device

    if 'cuda' in device:
        name = 'Cuda:{}'.format(device.split(':')[1])
    else:
        name = 'NoCuda'

    # extract vectors from the model and inputs
    if vectors is None:
        with torch.no_grad():
            vectors = extract_vectors(
                model, loader.dataset.utterances, norm=norm, device=device,
                n_jobs=loader.num_workers, batch_size=loader.batch_size
            )

    # compute scores of trials
    scs, labs, _ = score_trials(loader.dataset, vectors, mode=mode, n_jobs=1)
    # compute metrics
    eer = calculate_eer(labs, scs)
    dcf = calculate_mindcf(labs, scs, p_target=p_target)
    # record eer and dcf
    if tboard is not None:
        tboard.add_scalars('Metric/%s/EER' % val_task, {name: eer[0]}, args.epoch)
        tboard.add_scalars('Metric/%s/DCF' % val_task, {name: dcf[0]}, args.epoch)
    if not ret_info:
        return eer, dcf
    else:
        return eer, dcf, vectors, scs
