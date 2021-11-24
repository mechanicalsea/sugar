"""
Function to scores from a given pandas.DataFrame, numpy.ndarray.

Score | numpy, pandas, torch

Author: Wang Rui
Version: v0.1
Create Date: 2020/9/27
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sugar.database import VerificationTrials
import joblib
from joblib import Parallel, delayed
from sklearn.preprocessing import minmax_scale

import sys
if any('jupyter' in arg for arg in sys.argv):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def save_sctr(sc, tr, score_file, root_dir='', scale=True):
    assert not os.path.exists(score_file)
    if scale is True:
        sc = minmax_scale(sc)
    with open(score_file, 'w') as f:
        for s, t in tqdm(zip(sc, tr), total=len(sc)):
            t1 = t[0].replace(root_dir, '')
            t2 = t[1].replace(root_dir, '')
            f.writelines('%.6f %s %s\n' % (s, t1, t2))

def load_sctr(score_file):
    """
    Load scores/label file with pair of trials.
    """
    assert os.path.exists(score_file)
    sc = []
    tr = []
    with open(score_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            s, t1, t2 = line.split(' ')
            t2 = t2.replace('\n', '')
            sc.append(float(s))
            tr.append([t1, t2])
    return sc, tr
    

def score_batch(vec, batch, mode="cosine"):
    """
    Calculate score between a vector and a batch of vectors.
    
        Parameter
        ---------
            vec : Tensor
            batch : Tensor
            mode : str
            
        Return
        ------
            dist : float
    
    """
    dist = F.cosine_similarity(
                vec.unsqueeze(-1),
                batch.unsqueeze(-1).transpose(0, 2)).detach().numpy()
    return dist


def score_cohorts(cohorts : torch.Tensor, 
                  vectors : dict, 
                  mode='cosine', 
                  device='cpu'):
    """
    Calculate scores between each vector and all cohorts.
    
        Parameter
        ---------
            cohorts : Tensor
            vectors : dict {path: Tensor}
            mode : str
                Now only support cosine.
            device : str
                A device of implementation. CUDA is recommended for large 
                cohorts, where the size of cohorts is larger than 1000.
                
        Return
        ------
            cohort_score : dict
                Scores between each vector and all cohorts.
    """
    

    task = "Score Cohorts"
    cohort_score = {}
    cohorts = cohorts.to(device)

    for spk, vec in tqdm(vectors.items(), total=len(vectors), desc=task):
        vec   = vec.to(device)
        score = F.cosine_similarity(
                vec.unsqueeze(-1),
                cohorts.unsqueeze(-1).transpose(0, 2)).mean(0).detach().cpu().numpy()
        cohort_score[spk] = score
    return cohort_score


def calculate_score(enroll_feat, test_feat, mode="norm2"):
    """
    Calculate score using the given pair-wise scoring.
        Now support cosine and nom2 scoring.Scoring should 
        be implemented in CPU that may be faster than that 
        in GPU.
        CPU: 10000it/s, GPU(busy): 800it/s, GPU: 6000it/s.

        Parameter
        ---------
            enroll_feat : 2-dim tensor
                Multiple embeddings/vectors represent an enrollment
                utterance. The number of these can be 1 or more.
            test_feat : 2-dim tensor
                Multiple embeddings/vectors represent an test
                utterance with the same number as enroll_feat.
            mode : (list of) str
                Scoring method, e.g., norm2 and cosine.

        Return
        ------
            score : (list) float
                Score(s) corresponding to the number of mode(s).
    """
    enroll_feat = enroll_feat.cpu()
    test_feat = test_feat.cpu()
    if isinstance(mode, str):
        if mode is "norm2":
            dist = F.pairwise_distance(
                enroll_feat.unsqueeze(-1),
                test_feat.unsqueeze(-1).transpose(0, 2)).mean().detach().numpy()
            score = -dist.item()
        elif mode is "cosine":
            dist = F.cosine_similarity(
                enroll_feat.unsqueeze(-1),
                test_feat.unsqueeze(-1).transpose(0, 2)).mean().detach().numpy()
            score = dist.item()
    elif isinstance(mode, list):
        score = [calculate_score(enroll_feat, test_feat, m) for m in mode]
    return score


def score_trials(trials : VerificationTrials, 
                 vectors : dict, 
                 mode="cosine", 
                 n_jobs=4):
    """
    Compute scores of trials based on pre-computed embeddings.
    
        Parameter
        ---------
            trials : Dataset {label, enroll, test}
            vectors : dict
                It consists of pairs of {utterance: embedding(s)}
            mode : (list of) str 
                The mode to calculate socres, e.g., cosine or norm2.
            n_jobs : int
                The number of multiple cpu.
                
        Return
        ------
            all_scores : 1-dim or 2-dim array
                scores corresponds to each trial. If mode is a list, 
                it is 2-dim, each column corresponds to a mode. 
            all_labels : 1-dim array
                labels corresponds to each trial.
            all_trials : list
                pairs of (enroll, test).
    """
    
    all_scores = []
    all_labels = []
    all_trials = []
    
    def compute_score(trial):
        label = trial["label"]
        enroll = trial["enroll"]
        test = trial["test"]
        enroll_embed = vectors[enroll]
        test_embed = vectors[test]
        score = calculate_score(enroll_embed, test_embed, mode=mode)
        return score, label, enroll, test
    
    task = "Compute Scores"

    if n_jobs > 1:
        with joblib.parallel_backend('threading', n_jobs=n_jobs):
            scored_trials = Parallel(verbose=0)(
                delayed(compute_score)(trial) 
                for trial in tqdm(trials, total=len(trials), desc=task))
    else:
        scored_trials = [compute_score(trial) 
            for trial in tqdm(trials, total=len(trials), desc=task)]
    
    for row in scored_trials:
        score, label, enroll, test = row
        all_scores.append(score)
        all_labels.append(label)
        all_trials.append([enroll, test])
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    return all_scores, all_labels, all_trials
    

def ztnorm(scores: pd.DataFrame,
           cohort: dict,
           nTop: int = None):
    """
    (Top) Score normalization using ZT-Norm where Z-Norm followed by T-Norm.

        Parameter
        ---------
            scores : DataFrame
                Scores of trials are normalized with columns {score, enroll, test}.
            cohort : dict
            nTop : int
                The number of selected samples, usually top X scoring or most 
                similar samples, where X is set to be, e.g., 200 ~ 400.

        Return
        ------
            scores : array
                Normalized scores of trials.
    """
    scores_mean = {}
    scores_std  = {}
    for key, value in cohort.items():
        if nTop is not None:
            value = np.sort(value, kind="mergesort")
            if nTop < 0:
                value = value[: -nTop]
            else:
                value = value[nTop:]
        scores_mean[key] = value.mean()
        scores_std[key]  = value.std()

    def score_ztnorm(row):
        """
        row : DataFrame.row {score, enroll, test}
        """
        enroll, test, orig_score = row["enroll"], row["test"], row["score"]
        if scores_std[enroll] != 0:
            znorm_score = (orig_score - scores_mean[enroll]) / scores_std[enroll]
        else:
            znorm_score = orig_score
        if scores_std[test] != 0:
            ztnorm_score = (znorm_score - scores_mean[test]) / scores_std[test]
        else:
            ztnorm_score = znorm_score
        return ztnorm_score
    norm_scores = scores.apply(lambda row: score_ztnorm(row), axis=1)
    return norm_scores


def snorm(scores: pd.DataFrame,
          cohort: dict,
          nTop: int = None,
          std_eps: float = 1e-8):
    """
    Vectorization implementation of snorm via vectorizing 
        the mean and std of znorm and tnorm.

        Parameter
        ---------
            scores : DataFrame
                Scores of trials are normalized with columns 
                {score, enroll, test}.
            cohort : dict
            nTop : int
                if not None, use adaptive normalization.
            std_eps : float

        Return
        ------
            scores : array
                Normalized scores of trials.
    """
    if nTop is not None:
        return asnorm(scores, cohort, nTop, std_eps)
    
    N = scores.shape[0]
    orig_scores   = scores['score'].values
    trial_enrolls = scores['enroll'].values
    trial_tests   = scores['test'].values
    
    task = "Cohort Statistics"
    
    scores_mean = {}
    scores_std  = {}
    
    for key, value in tqdm(cohort.items(), total=len(cohort), desc=task):
        m = value.mean()
        scores_mean[key] = m
        scores_std[key]  = np.sum((value - m)**2/N)**0.5
    
    task = "Normalization Statistics"
    
    znorm_mean = np.empty(N)
    znorm_std  = np.empty(N)
    tnorm_mean = np.empty(N)
    tnorm_std  = np.empty(N)
    
    for i in tqdm(range(N), total=N, desc=task):
        enroll, test, orig_score = trial_enrolls[i], trial_tests[i], orig_scores[i]
        znorm_mean[i] = scores_mean[enroll]
        znorm_std[i]  = max(scores_std[enroll], std_eps)
        tnorm_mean[i] = scores_mean[test]
        tnorm_std[i]  = max(scores_std[test], std_eps)
    
    znorm_scores = (orig_scores - znorm_mean) / znorm_std
    
    tnorm_scores = (orig_scores - tnorm_mean) / tnorm_std
    
    norm_scores  =  0.5 * (znorm_scores + tnorm_scores)
    return norm_scores


def asnorm(scores: pd.DataFrame,
           cohort: dict,
           nTop: int = 300,
           std_eps: float = 1e-8):
    """
    Adaptive score normalization with top n cohort samples.

        Parameter
        ---------
            scores : DataFrame
                Scores of trials are normalized with columns {score, enroll, test}.
            cohort : dict or Tensor
                
            nTop : int
                The number of selected samples, usually top X scoring or most 
                similar samples, where X is set to be, e.g., 200 ~ 400.
                If nTop < 0: select the nTop most disimiliar scores.
                If nTop > 0: select the nTop most similiar scores.

        Return
        ------
            scores : array
                Normalized scores of trials.
                
        Note
        ----
            torch.topk is faster than numpy.sort.
    """
    N = scores.shape[0]
    orig_scores   = scores['score'].values
    trial_enrolls = scores['enroll'].values
    trial_tests   = scores['test'].values
    
    task = "Cohort Statistics"

    scores_mean = {}
    scores_std  = {}
    
    for key, value in tqdm(cohort.items(), total=len(cohort), desc=task):
        if nTop < 0:
            value, _ = torch.topk(torch.from_numpy(-value), nTop, sorted=False)
            value = -value.numpy()
        else:
            value, _ = torch.topk(torch.from_numpy(value), nTop, sorted=False)
            value = value.numpy()
        m = value.mean()
        scores_mean[key] = m
        scores_std[key]  = np.sum((value - m)**2/abs(nTop))**0.5

    task = "Normalization Statistics"
    
    znorm_mean = np.empty(N)
    znorm_std  = np.empty(N)
    tnorm_mean = np.empty(N)
    tnorm_std  = np.empty(N)
    
    for i in tqdm(range(N), total=N, desc=task):
        enroll, test, orig_score = trial_enrolls[i], trial_tests[i], orig_scores[i]
        znorm_mean[i] = scores_mean[enroll]
        znorm_std[i]  = max(scores_std[enroll], std_eps)
        tnorm_mean[i] = scores_mean[test]
        tnorm_std[i]  = max(scores_std[test], std_eps)
    
    znorm_scores = (orig_scores - znorm_mean) / znorm_std
    
    tnorm_scores = (orig_scores - tnorm_mean) / tnorm_std
    
    norm_scores  =  0.5 * (znorm_scores + tnorm_scores)
    return norm_scores

def normalize_scores(scores, s_min=0, s_max=1):
    """
    Normalize scores into specific range.

        Parameter
        ---------
            scores : np.ndarray
            s_min : float
            s_max : float
        
        Return
        ------
            norm_scores : np.ndarray

        Requirement
        -----------
            sklearn >= 0.17
    """
    return minmax_scale(scores, (s_min, s_max))
