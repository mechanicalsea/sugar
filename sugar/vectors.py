"""
Function to vectors from a pre-trained model, a given file list, numpy.ndarray, or torch.Tensor.

Vector | torch, numpy

Author: Wang Rui
Version: v0.1
Create Date: 2020/9/27
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sugar.database import Utterance
import joblib
from joblib import Parallel, delayed

import sys
if any('jupyter' in arg for arg in sys.argv):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def length_normalize(ivectors, scaleup=False):
    """
    Apply length normalization to i-vector. It also works for other
        vectors.

        This function is modified from: 
        http://www.kaldi-asr.org/doc/ivector-normalize-length_8cc_source.html

        Parameter
        ---------
            ivectors : 1-dim/2dim array or tensor
                (Batch of) i-vector.
            scaleup: bool
                If true, the normalized iVector is scaled-up by sqrt 
                (i-vector dim); if false, the normalized iVector is 
                p2-normalized.

        Return
        ------
            norm_ivectors : 1-dim/2dim array or tensor
                Normalized (batch of) i-vector.
            ratio_avg : float
                Average ratio of i-vector to expected length.
            ratio_std : float
                Standard deviation of ratio of i-vector.
    """
    if isinstance(ivectors, torch.Tensor):
        ratio = ivectors.norm(p=2, dim=-1, keepdim=True)
    elif isinstance(ivectors, np.ndarray):
        ratio = np.linalg.norm(ivectors, ord=2, axis=-1, keepdims=True)
    if scaleup is True:
        ratio /= np.sqrt(ivectors.shape[-1])
    norm_ivectors = ivectors / ratio
    ratio_avg = ratio.mean()
    ratio_std = ((ratio**2).mean() - ratio_avg**2)**0.5
    return norm_ivectors, ratio_avg, ratio_std


def extract_vectors(model, datalst, norm=True, device="cpu",
                    n_jobs=4, batch_size=1):
    """
    Extract vectors from trials via mode. There is no repeated utterances
        computed by filtering them, so the computation is efficient.
        There is 2 mode: 
        1. batch-style implementation enable GPU,
        2. multi-cpu implementation via joblib in multiple threading.
        
    Parameter
    ---------
        model : torch.nn.Module
            A model to compute vectors of utterances.
        datalst :  Utterance
            A utterance.
        load_wav : function
            A function is used to read waveform. This function is essential 
            to evaluting, so consider it.
        norm : bool
            Determine whether normlization vectors using norm2.
        device : str
            CPU or GPU, e.g., cpu or cuda:0
        n_jobs : int
            The number of subprocess.
        batch_size : int
            If batch_size is 1, the mode of computation is multi-threading 
            implementation. If batch_size > 1, the mode of computation is 
            batch-style implementation enable GPU.
            
    Return 
    ------
        vectors : dict {utterance: embedding}
    """
    vectors = {}
    
    if batch_size == 1:

        model.to(device)
        model.eval()

        def extract_vec(wav):
            file = wav["file"]
            wave = wav["input"].to(device)
            vec  = model(wave)

            if norm is True:
                vec = F.normalize(vec, p=2, dim=1).detach()
            else:
                vec = vec.detach()

            vectors[file] = vec
        
        task = "Extract Vectors"

        if n_jobs > 1:
            with joblib.parallel_backend('threading', n_jobs=n_jobs):
                Parallel(verbose=0)(
                    delayed(extract_vec)(wav) 
                    for wav in tqdm(datalst, total=len(datalst), desc=task))
        else:
            for wav in tqdm(datalst, total=len(datalst), desc=task):
                extract_vec(wav) 
                        
    elif batch_size > 1:
        batch_size = max(2, batch_size)
        trialoader = DataLoader(datalst, batch_size=batch_size,
                                collate_fn=datalst.collate_fn,
                                num_workers=n_jobs, shuffle=False)

        model.to(device)
        model.eval()

        task = "Extract Vectors"

        for data in tqdm(trialoader, total=len(trialoader), desc=task):
            file = data["file"]
            wave = data["input"].to(device)
            num  = data["num"]
            vec  = model(wave)

            if norm is True:
                vec = F.normalize(vec, p=2, dim=1).detach()
            else:
                vec = vec.detach()

            start_i = 0
            i = 0
            while start_i < vec.shape[0]:
                vectors[file[i]] = vec[start_i : start_i + num[i]]
                start_i += num[i]
                i += 1
    return vectors


def vector_transform(vectors, transform=None):
    """
    Transform vectors/embeddings. If transform is None,
    it copies them.

        Parameter
        ---------
            vectors : dict
            transform : function

        Return
        ------
            transformed_vectors : dict
    """
    transformed_vectors = {}
    if transform is not None:
        for key, value in tqdm(vectors.items(), desc="Transform Vectors"):
            transformed_vectors[key] = transform(value)
    else:
        for key, value in tqdm(vectors.items(), desc="Copy Vectors"):
            transformed_vectors[key] = value
    return transformed_vectors

