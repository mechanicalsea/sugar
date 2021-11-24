"""
Function to features from a given file, numpy.ndarray, or torch.Tensor.

Feature | 

Author: Wang Rui
Version: v0.1
Create Date: 2020/9/27
"""
import random
import numpy as np


def load_feat(path, num_frames=200, mode_eval=False, num_eval=6, min_frames=300):
    """
    Read an acoustic feature npy file from the given path. A forced 
        aligned feature is extracted by wrap padding (concatenate repeat 
        data) or cropping the original feature. This operation is from [1].
        The npy file is (F, T) 2-dim float32 array and is saved using 
        allow_pickle=True.

        [1] Zhang, C., Koishida, K., End-to-End Text-Independent Speaker 
        Verification with Triplet Loss on Short Utterances, Interspeech 
        2017. 1487–1491. https://doi.org/10.21437/Interspeech.2017-1608

        Parameter
        ---------
            path : str
                The npy file path of a wav.
            num_frames : int
                The number of frames of the acoustic feature to be read.
            mode_eval : bool
                If true, the feature is chunked using linspace.
            num_eval : int
                If mode_eval is true, it is the number of chunks 
                of the feature.
            min_frames : int
                If num_frames is 0, it works for wrapping wavefrom 
                as num_frames.

        Return
        ------
            feat : 2/3-dim array
                A feature with the given frames. If mode_eval is 
                false, the feature is of 2-dim. If mode_eval is 
                true, the feature is of 3-dim, as a batch of features.
    """
    feat = np.load(path, allow_pickle=True)
    # NaN or -inf -> log(1e-6)
    feat[np.isnan(feat) | np.isneginf(feat)] = -13.8155  
    
    F, T = feat.shape

    if T <= num_frames:
        shortage = num_frames - T + 1
        feat     = np.pad(feat, ((0, 0), (0, shortage)), mode="wrap")
        T        = feat.shape[1]
    elif num_frames == 0:
        if min_frames is not None and T <= min_frames:
            shortage = min_frames - T + 1
            feat     = np.pad(feat, (0, shortage), mode="wrap")
            T        = feat.shape[1]

    if mode_eval is False:
        startframe = np.int64(random.random()*(T-num_frames))
        feature = feat[:, startframe: startframe + num_frames]
    else:
        if num_frames != 0:
            startframe = np.linspace(0, T-num_frames, num=num_eval)
            feature = []
            for i_start in startframe:
                feature.append(feat[:, int(i_start): int(i_start)+num_frames])
            feature = np.stack(feature, axis=0)
        else:
            feature = np.stack([feat], axis=0)
    
    return feature


def feat_segment(feat, num_frames=300, noverlap=None):
    """
    Segment a feature into the specific length.
    
        Parameter
        ---------
            audio : 2-dim/3-dim array
                A feature with (F, T).
            num_frames :
                The length of the resulted segments.
        
        Return
        ------
            feature : 3-dim array
                Multiple segments from the input with the specific length.
    """
    feat     = feat[0] if feat.ndim == 3 else feat
    shortage = num_frames - (feat.shape[1] % num_frames)
    feat     = np.pad(feat, ((0, 0), (0, shortage)), mode="wrap")
    featsize = feat.shape[1]

    start_i  = 0
    noverlap = num_frames if noverlap is None else noverlap
    feature  = []
    while start_i < featsize:
        feature.append(feat[:, start_i: start_i + num_frames])
        start_i += noverlap

    feature = np.stack(feature, axis=0)
    return feature


def vad(feat, energy_threshold=5.0, energy_mean_scale=0.5,
        frames_context=0, proportion_threshold=0.6):
    """
    Compute voice activity detection. This function is reproduced from:
        http://www.kaldi-asr.org/doc/voice-activity-detection_8cc_source.html
        It can be applied to acoustic features, e.g., MFCC, spectrogram,
        and fbanks.

        Parameter
        ---------
            feat : 2-dim array
                an acoustic feature

        Return
        ------
            feat : 2-dim array
                an voiced acoustic feature
            voiced : 1-dim bool array
                an voiced frame
    """
    F, T = feat.shape

    log_energy = feat[0, :]

    energy_threshold += energy_mean_scale * log_energy.sum() / T

    log_energy = np.pad(log_energy, (frames_context, frames_context), mode="constant")
    log_energy = np.stack([log_energy[i: i+T] for i in range(0, 2*frames_context+1)], axis=0)

    voiced = (log_energy>energy_threshold).sum(0) > ((2*frames_context+1) * proportion_threshold)

    return feat[:, voiced], voiced


def freq_mask(spec, F=10, num_masks=2, replace_with_zero=False):
    """
    Frequency Mask is intuitively described as mask certain 
        frequency bands with either the mean value of the 
        spectrogram or zero.
    
        Parameter
        ---------
            spec : 2-dim array with shape [F, T]
                2-dim feature, such spectrogram, mel-fbank, or MFCC
            F : int
                maximum number of frequency mask
            num_masks : int
                number of mask
            replace_with_zeros : bool, False
                to determine either mean or zero
            
        Return
        ------
            masked spec : 2-dim array with shape [F, T]
    """
    cloned = spec.copy()
    num_mel_channels = cloned.shape[0]
    for i in range(0, num_masks):        
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)
        if (f_zero == f_zero + f): 
            return cloned
        mask_end = random.randrange(f_zero, f_zero + f) + 1
        if replace_with_zero is True: 
            cloned[f_zero:mask_end] = 0
        else: 
            cloned[f_zero:mask_end] = cloned.mean()
    return cloned


def time_mask(spec, T=40, num_masks=2, replace_with_zero=False):
    """
    Time Mask is intuitively described as mask certain time ranges 
        with the mean value of the spectrogram or zero.
        
        Parameter
        ---------
            spec : 2-dim array with shape [F, T]
                2-dim feature, such spectrogram, mel-fbank, or MFCC
            T : int
                maximum number of frequency mask
            num_masks : int
                number of mask
            replace_with_zeros : bool
                to determine either mean or zero
            
        Return
        ------
            masked spec : 2-dim array with shape [F, T]
    """
    cloned = spec.copy()
    len_spectro = cloned.shape[1]
    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)
        if (t_zero == t_zero + t): 
            return cloned
        mask_end = random.randrange(t_zero, t_zero + t) + 1
        if replace_with_zero is True: 
            cloned[:,t_zero:mask_end] = 0
        else: 
            cloned[:,t_zero:mask_end] = cloned.mean()
    return cloned

