"""
Function to metrics from numpy.ndarray.

Metric | numpy, scipy, sklearn, torch

Author: Wang Rui
Version: v0.1
Create Date: 2020/9/27
"""
import numpy as np
import thop
import time
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def print_size_of_model(model, verbose=False):
    """Count the size of saved model.

        Parameters
        ----------
            model : torch.nn.Module

        Returns
        -------
            model_size : Unit (MB)

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
    if verbose: 
        print('Size (MB):', model_size)
    os.remove('temp.p')
    return model_size

def calculate_eer(y, y_score, pos=1):
    """
    Calculate Equal Error Rate (EER).

        Parameters
        ----------
            y : array_like, ndim = 1
                y denotes groundtruth scores {0, 1}.
            y_score : array_like, ndim = 1 or 2
                y_score denotes (multiple) the prediction scores.

        Returns
        -------
            eer : array
                EER between y and y_score.
            thresh : float
                threshold of EER.
    """
    assert y_score.ndim == 1 or y_score.ndim == 2
    if y_score.ndim == 1:
        fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=pos)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
    else:
        eer_thresh = [calculate_eer(y, y_score[:, i], pos)
                      for i in range(y_score.shape[1])]
        eer_thresh = np.array(eer_thresh)
        best_mode = eer_thresh[:, 0].argmin()
        eer, thresh = eer_thresh[best_mode, 0], eer_thresh[best_mode, 1]
    return eer, np.float(thresh)


def compute_error_rates(scores, labels):
    """
    Creates a list of false-negative rates (fnr), a list of false-positive rates (fpr)
        and a list of decision thresholds that give those error-rates.

        Copyright
        ---------
            def ComputeErrorRates(scores, labels):
                ...
            2018 David Snyder
            This script is modified from: https://github.com/kaldi-asr/kaldi/blob/8ce3a95761e0eb97d95d3db2fcb6b2bfb7ffec5b/egs/sre08/v1/sid/compute_min_dcf.py
    """
    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(scores, list):
        scores = np.array(scores)

    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes = np.argsort(scores)  # vectorized
    thresholds = scores[sorted_indexes]  # vectorized
    # sorted_indexes, thresholds = zip(*sorted(
    #     [(index, threshold) for index, threshold in enumerate(scores)],
    #     key=itemgetter(1)))

    # sorted_labels
    labels = labels[sorted_indexes]  # vectorized
    # labels = [labels[i] for i in sorted_indexes]

    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = np.array(fnrs) / float(fnrs_norm)  # vectorized
    # fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = 1 - np.array(fprs) / float(fprs_norm)  # vectorized
    # fprs = [1 - x / float(fprs_norm) for x in fprs]

    return fnrs, fprs, thresholds


def compute_mindcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1, mode="vectorized"):
    """
    Computes the minimum of the detection cost function.  The comments refer 
        to equations in Section 3 of the NIST 2016 Speaker Recognition 
        Evaluation Plan.

        Copyright
        ---------
            def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
                ...
            2018 David Snyder: 
            This script is modified from: https://github.com/kaldi-asr/kaldi/blob/8ce3a95761e0eb97d95d3db2fcb6b2bfb7ffec5b/egs/sre08/v1/sid/compute_min_dcf.py
    """
    assert mode is "vectorized" or mode is "for" 
    if mode is "vectorized":
        # vectorized-version
        if isinstance(fnrs, list):
            fnrs = np.array(fnrs)
        if isinstance(fprs, list):
            fprs = np.array(fprs)
        c_det = c_miss * fnrs * p_target + c_fa * fprs * (1 - p_target)
        min_index = c_det.argmin()
        min_c_det = c_det[min_index]
        min_c_det_threshold = thresholds[min_index]
    else:
        # for-version
        min_c_det = float("inf")
        min_c_det_threshold = thresholds[0]
        for i in range(0, len(fnrs)):
            # See Equation (2).  it is a weighted sum of false negative
            # and false positive errors.
            c_det = c_miss * fnrs[i] * p_target + \
                c_fa * fprs[i] * (1 - p_target)
            if c_det < min_c_det:
                min_c_det = c_det
                min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def calculate_mindcf(y, y_score, p_target=0.01, c_miss=1, c_fa=1):
    """
    Calculate MinDCF with p_target, c_miss, and c_fa in the NIST 2016 
        Speaker Recognition Evaluation Plan.

        Parameters
        ----------
            y : array, ndim = 1
                y denotes groundtruth scores {0, 1}.
            y_score : array, ndim = 1 or 2
                y_score denotes (multiple) the prediction scores.
            p_target : float 
            c_miss : float
            c_fa : float

        Returns
        -------
            mindcf : float
                MinDCF between y and y_score.
            threshold : float
                threshold of MinDCF.
    """
    assert y_score.ndim == 1 or y_score.ndim == 2
    if y_score.ndim == 1:
        fnrs, fprs, thresholds = compute_error_rates(y_score, y)
        mindcf, threshold = compute_mindcf(fnrs, fprs, thresholds,
                                           p_target=p_target,
                                           c_miss=c_miss,
                                           c_fa=c_fa)
    else:
        dcf_thresh = [calculate_mindcf(y, y_score[:, i],
                                       p_target=p_target,
                                       c_miss=c_miss,
                                       c_fa=c_fa)
                      for i in range(y_score.shape[1])]
        dcf_thresh = np.array(dcf_thresh)
        best_mode = dcf_thresh[:, 0].argmin()
        mindcf, threshold = dcf_thresh[best_mode, 0], dcf_thresh[best_mode, 1]
    return mindcf, threshold


def calculate_val(y, y_score, far=1e-3):
    """
    Calculate validation rate (VAL) at a specified false accept rate (FAR).
        It also works for training process.

        Inspired by [1].
        [1] Zhang, C., Koishida, K., 2017. End-to-End Text-Independent 
        Speaker Verification with Triplet Loss on Short Utterances, in: 
        Interspeech 2017. pp. 1487–1491. 
        https://doi.org/10.21437/Interspeech.2017-1608
        
        Face Verification also uses it as metrics as follows, such as 
        VR@FAR=0.01%, VR@FAR=0.1%, VR@FAR=1%, VR@FAR=0.01%, VR@FAR=1e-6
        [2 ]Wang, F., Cheng, J., Liu, W., Liu, H., 2018. Additive Margin 
        Softmax for Face Verification. IEEE Signal Process. Lett. 25, 926–930. 
        https://doi.org/10.1109/LSP.2018.2822810

        Parameter
        ---------
            y : array
                A list of label {0, 1} corresponding to 
                each trial.
            y_score : array
                A list of score corresponding to each trial.
            far : float
                Specified false accept rate that the rate of real non-target assigned non-target.

        Return
        ------
            VAL : float
                Validation rate, real target assigned target, at a specified FAR.
            threshold : float
                Threshold such that satisfies the given very low FAR.
    """
    assert y_score.ndim == 1 or y_score.ndim == 2
    if y_score.ndim == 1:
        l, r = y_score.min(), y_score.max()
        gap = r - l
        nontar = (y == 0)
        n_nontar = nontar.sum()
        while (r - l) >= far * 1e-3:
            m = (l + r) / 2
            FAR = (nontar & (y_score >= m)).sum() / n_nontar
            if FAR > far:
                l = m
            else:
                r = m
        threshold = r
        FAR = (nontar & (y_score >= threshold)).sum() / n_nontar
        assert FAR <= far
        target = (y == 1)
        n_target = target.sum()
        VAL = y[target & (y_score >= threshold)].sum() / n_target
    else:
        VAL_thresh = [calculate_val(y, y_score[:, i], far=far)
                      for i in range(y_score.shape[1])]
        VAL_thresh = np.array(VAL_thresh)
        best_mode = VAL_thresh[:, 0].argmin()
        VAL, threshold = VAL_thresh[best_mode, 0], VAL_thresh[best_mode, 1]
    return VAL, threshold


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k, such as
        top 1 precision and top 5 precision. It also works for 
        training process.
        It works well on Softmax Loss.

        Parameter
        ---------
            output : 1-dim tensor
                predicted probilities.
            target : 1-dim tensor
                groudtruth labels.

        Return
        ------
            res : list
                a list of precision@k, e.g., top 1, top 5.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def profile(model, input_size, custom_ops=None, verbose=False, device='cpu', format=True):
    """
    Calculate a model/module's MACs/MAdd and parameters in PyTorch.

        Parameter
        ---------
            model : torch.nn.Module
            input_size : tuple or list
            custom_ops : dict
                Ops as register_ops in `thop`.
            verbose : bool

        Return
        ------
            macs : str
                the number of MACs/MAdd
            params : str
                the number of parameters

        Note
        ----
            There always is a gap between the MACs from `profile` and from `count_ops`, in which the 
                former is lower. The reason is that the operation of `x * y` in SELayer and the 
                operation of `out += identify` in L3BasicBlock are not considered in `profile`. The 
                comment 'This is a place that profile can not count' is appended to those operations.
    """
    origin_device = 'cpu'
    if len(list(model.parameters())):
        origin_device = model.parameters().__next__().device
    origin_state  = model.training

    model.eval()
    model.to(device)
    with torch.no_grad():
        inp = torch.rand(*input_size, device=device)
        macs, params = thop.profile(model, inputs=(inp,), custom_ops=custom_ops, verbose=verbose)
    if format: 
        macs, params = thop.clever_format([macs, params], "%.2f")

    model.train(origin_state)
    if len(list(model.parameters())):
        model.to(origin_device)

    return macs, params


def latency(model, input_size, device='cpu', warmup_steps=10, measure_steps=50):
    """
    Calculate a model/module's latency in PyTorch, which relates to the used device.

        Parameters
        ----------
            model : torch.nn.Module
            input_size : list or tuple

        Return
        ------
            avg_time : float
                The ms of latency.

        Notes
        -----
            The latency on cpu/edge device usually consider the input size of batch size of 1, while
            the latency on gpu/cloud device usually consider the input size of batch size of 64.
    """
    origin_device = 'cpu'
    if len(list(model.parameters())):
        origin_device = model.parameters().__next__().device
    origin_state  = model.training

    model.eval()
    model.to(device)

    with torch.no_grad():
        inp = torch.rand(*input_size, device=device)
        if device != 'cpu': torch.cuda.synchronize(device=device)
        for i in range(warmup_steps):
            model(inp)
        if device != 'cpu': torch.cuda.synchronize(device=device)

        if device != 'cpu': torch.cuda.synchronize(device=device)
        st = time.time()
        for i in range(measure_steps):
            model(inp)
        if device != 'cpu': torch.cuda.synchronize(device=device)
        ed = time.time()
        total_time = ed - st

    avg_time = total_time / measure_steps * 1000 # ms

    model.train(origin_state)
    if len(list(model.parameters())):
        model.to(origin_device)

    return avg_time
