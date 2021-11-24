"""
为了加速实验，搞了这么一出
Author: Wang Rui / rwang
Version: 1.0

备注：
    __main__ 给出了训练和评估的案例

模型设计：
    替换测试案例中的说话人嵌入提取模型、顶层分类器模型
    相关的模块：ResNetSE34L, AMSoftmax

数据增益：
    替换测试案例中的文件读取方法
    相关的模型：loadWAV

测试案例：
```
if __name__ == "__main__":
    # 定义训练集、测试集及其两者的根目录
    trainlst = "/workspace/rwang/voxceleb/train_list.txt"
    testlst = "/workspace/rwang/VoxSRC2020/data/verif/trials.txt"
    traindir = "/workspace/rwang/voxceleb/voxceleb2/"
    testdir = "/workspace/rwang/voxceleb/"
    maptrain5994 = "/workspace/rwang/competition/voxsrc2020/maptrain5994.txt"
    # 载入训练集
    train = load_train(trainlst=trainlst, traindir=traindir,
                       maptrain5994=maptrain5994)
    # 载入测试集
    trial = load_trial(testlst=testlst, testdir=testdir)
    # 定义说话人嵌入提取模型
    net = ResNetSE34L(nOut=512, num_filters=[16, 32, 64, 128])
    # 定义顶层分类器模型
    top = AMSoftmax(in_feats=512, n_classes=5994, m=0.2, s=30)
    # sklearn 模型生成
    snet = SpeakerNet(net=net, top=top)
    # 模型训练
    modelst, step_num, loss, prec1, prec5 = snet.train(train, num_epoch=1)
    # 模型评估
    eer, thresh, all_scores, all_labels, all_trials, trials_feat = snet.eval(
        trial, step_num=0, trials_feat=None)
```

After that,

The part of Sugar. Testing Along Coding.

Author: Rui Wang
Version: 0.1
"""

import os
import glob
import math
import random
import IPython
import pandas as pd
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.deterministic = True
np.random.seed(22)
torch.manual_seed(22)
torch.cuda.manual_seed(22)
torch.cuda.manual_seed_all(22)

# 数据准备
trainlst = "/workspace/rwang/voxceleb/train_list.txt"
testlst = "/workspace/rwang/VoxSRC2020/data/verif/trials.txt"
traindir = "/workspace/rwang/voxceleb/voxceleb2/"
testdir = "/workspace/rwang/voxceleb/"
maptrain5994 = "/workspace/rwang/competition/voxsrc2020/maptrain5994.txt"
batch_size = 200
num_worker = 4
max_utt_per_spk = 100
L = 32240
len_eval = 48240
# 模型准备
n_mel = 40
n_out = 512
# 训练与测试
eval_interval = 10
num_epoch = 500
margin = 0.2
scale = 30
num_speaker = 5994
lr = 0.001
gamma = 0.95
num_eval = 10
# 模型训练
device = "cuda:2"
log_dir = "/workspace/rwang/competition/voxsrc2020/logs/exp-22"

__all__ = ["load_train", "load_trial", "loadWAV", "load_feat",
           "read", "read_feat", "length_normalize", "trials2utterance",
           "VerificationTrials", "utt2dur", "feat_segment", "wav_segment",
           "wav_embedding", "trial_scoring", "trial_evaluate",
           "MultiUtterance", "SingleUtterance", "UniqueSampler", "Utterance",
           "SpeakerNet", "ResNetSE34L", "AMSoftmax",
           "AAMSoftmax", "SoftmaxLoss", "TripletLoss", "AngularProto",
           "SEBasicBlock", "StatsPool", "SAP", "TDNNBlock",
           "calculate_eer", "calculate_score", "evaluate",
           "extract_embedding", "load_weights",
           "compute_vad", "wav_vad",
           "snorm", "asnorm", "ztnorm",
           "calculate_mindcf",
           "lr_gamma"]


# done
def wav_segment(audio, num_samples=48000, sample_rate=16000):
    """Segment a waveform into the specific length.
    
        Parameter
        ---------
            audio : 1-dim/2-dim array
                A waveform.
            num_samples :
                The length of the resulted segments.
            sample_rate :
                Sampling rate of the waveform.
        
        Return
        ------
            wave : 2-dim array
                Multiple segments from the input with the specific length.
    """
    audio = audio[0] if audio.ndim == 2 else audio
    shortage = num_samples - (audio.shape[0] % num_samples)
    audio = np.pad(audio, (0, shortage), mode="wrap")
    startsample = np.linspace(0, audio.shape[0], num=audio.shape[0]//num_samples+1)
    wave = []
    for i in range(1, len(startsample)):
        wave.append(audio[int(startsample[i-1]): int(startsample[i])])
    wave = np.stack(wave, axis=0)
    return wave

# done
def feat_segment(feat, num_frames=300):
    """Segment a feature into the specific length.
    
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
    feat = feat[0] if feat.ndim == 3 else feat
    shortage = num_frames - (feat.shape[1] % num_frames)
    feat = np.pad(feat, ((0, 0), (0, shortage)), mode="wrap")
    startframe = np.linspace(0, feat.shape[1], num=feat.shape[1]//num_frames+1)
    feature = []
    for i in range(1, len(startframe)):
        feature.append(feat[:, int(startframe[i-1]): int(startframe[i])])
    feature = np.stack(feature, axis=0)
    return feature

##############
# 一些实用的工具
##############

# done
def compute_vad(feat, energy_threshold=5.0, energy_mean_scale=0.5,
                frames_context=0, proportion_threshold=0.6):
    """Compute voice activity detection. This function is reproduced from:
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
                an voiced vector
    """
    F, T = feat.shape
    log_energy = feat[0, :]
    energy_threshold += energy_mean_scale * log_energy.sum() / T
    log_energy = np.pad(log_energy, (frames_context,
                                     frames_context), mode="constant")
    log_energy = np.stack([log_energy[i: i+T]
                           for i in range(0, 2*frames_context+1)], axis=0)
    voiced = (log_energy > energy_threshold).sum(0) > (
        (2*frames_context+1) * proportion_threshold)
    return feat[:, voiced], voiced

# done
def wav_vad(wave, fs=16000, win_len=400, hop_len=160, n_fft=512,
            energy_threshold=0.12, low_freq=300, high_freq=3000):
    """Use signal energy to detect voice activity in PyTorch's Tensor.
        Detects speech regions based on ratio between speech band energy 
        and total energy. Outputs are two tensor with the number of 
        frames where the first output is start frame and the second 
        output is to indicate voice activity.

        Parameter
        ---------
            wave : 1-dim tensor

        Return
        ------
            voiced_wave : 1-dim tensor
                a voiced waveform.
            voiced_sample : 1-dim tensor
                it indicate whether a sample is speech or not.
            energy : 2-dim tensor
                energy spectrogram of a total waveform.
    """
    if isinstance(wave, torch.Tensor):
        freq = torch.from_numpy(np.fft.rfftfreq(
            n_fft, d=1.0/fs).astype(np.float32)[1:])
        energy = torch.stft(wave, n_fft=n_fft, hop_length=hop_len, win_length=win_len,
                            window=torch.hann_window(win_len), center=False, onesided=True)
        energy = energy[:, :, 0].pow(2) + energy[:, :, 1].pow(2)
        energy = energy[1:, :]
        voiced_energy = torch.sum(
            energy[(freq >= low_freq) & (freq <= high_freq), :], dim=0)
        full_energy = torch.sum(energy, dim=0)
        full_energy[full_energy == 0] = 2.220446049250313e-16
        voiced = (voiced_energy / full_energy) >= energy_threshold
        voiced_sample = torch.arange(0, voiced.shape[0] * hop_len,
                                     device=wave.device).reshape(-1, hop_len)[voiced, :].flatten()
        voiced_wave = wave[voiced_sample]
    else:
        f, t, Sxx = signal.stft((wave * 1.0).astype(np.float32), fs=fs, window="hann",
                                nperseg=win_len, noverlap=win_len-hop_len, nfft=n_fft,
                                padded=False, boundary=None, return_onesided=True)
        energy = np.abs(Sxx)[1:, :]
        f = f[1:]
        voiced_energy = np.sum(
            energy[(f >= low_freq) & (f <= high_freq), :], axis=0)
        full_energy = np.sum(energy, axis=0)
        full_energy[full_energy == 0] = 2.220446049250313e-16
        voiced = (voiced_energy / full_energy) >= energy_threshold
        voiced_sample = torch.arange(
            0, voiced.shape[0] * hop_len).reshape(-1, hop_len)[voiced, :].flatten()
        voiced_wave = wave[voiced_sample]
    return voiced_wave, voiced_sample, energy

# done
def length_normalize(ivectors, scaleup=True):
    """Apply length normalization to i-vector.

        This function is modified from: 
        http://www.kaldi-asr.org/doc/ivector-normalize-length_8cc_source.html

        Parameter
        ---------
            ivectors : 1-dim/2dim array or tensor
                (Batch of) i-vector.
            scaleup: bool
                If true, the normalized iVector is scaled-up by sqrt(i-vector dim)

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


############
#  数据准备 #
############

# done to database.UniqueSampler
class UniqueSampler(Sampler):
    """Indices Resampler for dataset that ensure there is no repeated
        speakers in a batch. It supports multiple utterances per speaker.
    """

    def __init__(self,
                 data_source: pd.DataFrame,
                 speaker_id: dict,
                 batch_size: int = 128,
                 max_utt_per_spk: int = 100,
                 n_per_spk: int = 1):
        """

            Parameter
            ---------
                data_source : DataFrame {"speaker", "file"}
                    A dataframe container a list of pair of speaker and file.
                speaker_id : dict
                    Map speaker into id.
                batch_size : int
                    The size of sample batch in the dataloader.
                max_utt_per_spk : int
                    The maximum number of utterances in each speaker.
                n_per_spk : int
                    The number of utterances in a sample.

        """
        self.data_source = data_source.values
        self.speaker_id = speaker_id
        self.data_dict = data_source.groupby(by="speaker").groups.items()
        self.batch_size = batch_size
        self.max_utt_per_spk = max_utt_per_spk
        self.n_per_spk = n_per_spk
        self.utt_per_sample = lambda lst: [list(lst[i: i + n_per_spk])
                                           for i in range(0, len(lst), n_per_spk)]

    def __iter__(self):
        flattened_list = []
        flattened_label = []
        # Data for each class
        for speaker, indices in self.data_dict:
            label = self.speaker_id[speaker]
            numUtt = min(len(indices), self.max_utt_per_spk)
            numUtt = numUtt - (numUtt % self.n_per_spk)
            rp = self.utt_per_sample(np.random.permutation(indices)[:numUtt])
            # it may be further accelarated.
            flattened_label.extend([label] * (len(rp)))
            flattened_list.extend(rp)
        # Data in random order
        mixid = np.random.permutation(len(flattened_list))
        mixlabel = []
        mixmap = []
        # Prevent two pairs of the same speaker in the same batch
        for i in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            idx = flattened_list[i]
            label = flattened_label[i]
            if speaker not in mixlabel[startbatch:]:
                mixmap.append(idx)
                mixlabel.append(label)
        # Iteration size
        self.num_file = len(mixmap)
        self.files = mixmap
        if self.n_per_spk == 1:
            mixmap = np.array(mixmap).flatten().tolist()
        return iter(mixmap)

    def __len__(self):
        return len(self.data_source)

# done
class MultiUtterance(Dataset):
    """Format of data source format: speaker, wav_path.
        Example:
        id00012 id00012/21Uxsk56VDQ/00001.wav
        It works for pairwise learning or E2E learning.
    """

    def __init__(self, datalst: np.ndarray, speaker: dict, read_func):
        super(MultiUtterance, self).__init__()
        self.datalst = datalst
        self.speaker = speaker
        self.read = read_func

    def __len__(self):
        return len(self.datalst)

    def __getitem__(self, idx):
        label = [self.speaker[self.datalst[i][0]] for i in idx]
        assert len(set(label)) == 1
        label = label[0]
        inputs = []
        for i in idx:
            _, path = self.datalst[i]
            inp = self.read(path)
            inp = torch.FloatTensor(inp)
            inputs.append(inp)
        return {"label": label, "input": inputs}

    @staticmethod
    def collate_fn(batch):
        """Collect from a batch of MultiUtterance Dataset."""
        labels = torch.LongTensor([item["label"] for item in batch])
        nUtts = len(batch[0]["input"])
        inputs = [torch.stack([item["input"][i]
                               for item in batch], dim=0) for i in range(nUtts)]
        return {"label": labels, "input": inputs}

# done
class Utterance(Dataset):
    """
    Dataset for load utterances via a list. 
    Data source format: waveform or feature.

    Example
    -------
        id00012/21Uxsk56VDQ/00001.wav
        id00012/21Uxsk56VDQ/00001.logfbank.npy
    """

    def __init__(self, dataset, read_func):
        self.dataset = dataset
        self.read = read_func
        # ".wav" inputs a waveform
        self.ndim = 1 if ".wav" in dataset[0] else 2 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        file = self.dataset[idx]
        wave = self.read(file)
        wave = torch.FloatTensor(wave)
        return {"file": file, "wave": wave}

    def collate_fn(self, batch):
        """Collect from a batch of waveforms/feature, where an utterance 
            can be chunked into several segments as read in evaluting mode, 
            e.g., An utterance is loaded 6 segments with all 48000 samples.
            Now support different number of chunks of each input as num.
        """
        file = [item["file"] for item in batch]
        wave = torch.cat([item["wave"] for item in batch], dim=0)
        num  = [item["wave"].shape[0] if item["wave"].ndim >= self.ndim else 1 for item in batch]
        return {"file": file, "wave": wave, "num": num}

# done
class SingleUtterance(Dataset):
    """Format of data source format: speaker, wav_path
        Example:
        id00012 id00012/21Uxsk56VDQ/00001.wav
        It works for classification objective learning.
    """

    def __init__(self, datalst: np.ndarray, speaker: dict, read_func):
        super(SingleUtterance, self).__init__()
        self.datalst = datalst
        self.speaker = speaker
        self.read = read_func

    def __len__(self):
        return len(self.datalst)

    def __getitem__(self, idx):
        spk, path = self.datalst[idx]
        label = self.speaker[spk]
        inp = self.read(path)
        inp = torch.FloatTensor(inp)
        return {"label": label, "input": inp}

# done
def read(path, num_samples=48000, mode_eval=False, num_eval=6, sample_rate=16000, min_samples=None):
    """
    Read a wav file from the given path. A forced aligned waveform is 
        extracted by wrap padding (concatenate repeat data) or cropping 
        the original waveform. This operation is inspired from [1].

        [1] Zhang, C., Koishida, K., End-to-End Text-Independent Speaker 
        Verification with Triplet Loss on Short Utterances, Interspeech 
        2017. 1487–1491. https://doi.org/10.21437/Interspeech.2017-1608

    Parameter
    ---------
        path : str
            The file path of a wav.
        num_samples : int
            The number of samples of the waveform to be read.
            If 0, the total utterance is loaded.
        mode_eval : bool
            If true, the waveform is chunked using linspace.
        num_eval : int
            If mode_eval is true, it is the number of chunks 
            of the waveform.
        sample_rate : int
            The sampling rate is to be verified.
        min_samples : int
            If num_samples is 0, it works for wrapping wavefrom 
            as num_samples.
            
    Return
    ------
        wave : 1-dim/2-dim array
            A waveform with the given samples. If mode_eval is 
            false, the waveform is of 1-dim. If mode_eval is true, 
            the waveform is of 2-dim, as a batch of waveforms.
    """
    sr, audio = wavfile.read(path)
    assert sr == sample_rate, "sample rate is {} != {}".format(sr, sample_rate)
    audio_size = audio.shape[0]
    if audio_size <= num_samples:
        shortage = num_samples - audio_size + 1
        audio = np.pad(audio, (0, shortage), mode="wrap")
        audio_size = audio.shape[0]
    elif num_samples == 0:
        if min_samples is not None and audio_size <= min_samples:
            shortage = min_samples - audio_size + 1
            audio = np.pad(audio, (0, shortage), mode="wrap")
            audio_size = audio.shape[0]
    if mode_eval is False:
        startsample = max(0, np.int64(
            random.random()*(audio_size-num_samples)))
        wave = audio[startsample: startsample + num_samples]
    else:
        if num_samples != 0:
            startsample = np.linspace(0, audio_size-num_samples, num=num_eval)
            wave = []
            for i_start in startsample:
                wave.append(audio[int(i_start): int(i_start)+num_samples])
            wave = np.stack(wave, axis=0)
        else:
            wave = np.stack([audio], axis=0)
    return wave.astype(np.float32)

# done
def read_feat(path, num_frames=200, mode_eval=False, num_eval=6, min_frames=300):
    """Read an acoustic feature npy file from the given path. A forced 
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
            feat : 2-dim/3-dim array
                A feature with the given frames. If mode_eval is 
                false, the feature is of 2-dim. If mode_eval is 
                true, the feature is of 3-dim, as a batch of features.
    """
    feat = np.load(path, allow_pickle=True)
    feat[np.isnan(feat) | np.isneginf(feat)] = - \
        13.8155  # NaN or -inf -> log(1e-6)
    F, T = feat.shape
    if T <= num_frames:
        shortage = num_frames - T + 1
        feat = np.pad(feat, ((0, 0), (0, shortage)), mode="wrap")
        T = feat.shape[1]
    elif num_frames == 0:
        if min_frames is not None and T <= min_frames:
            shortage = min_frames - T + 1
            feat = np.pad(feat, (0, shortage), mode="wrap")
            T = feat.shape[1]
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


# 舍弃
def load_feat(filename, L=200, evalmode=False, num_eval=6):
    feat = np.load(filename)  # [F, T]
    feat[np.isnan(feat) | np.isneginf(feat)] = - \
        13.8155  # NaN or -inf -> log(1e-6)

    num_frame = feat.shape[-1]
    if num_frame <= L:
        shortage = math.floor((L - num_frame + 1) / 2)
        feat = np.pad(feat, ((0, 0), (shortage, shortage)), 'reflect')
        num_frame = feat.shape[-1]
    if evalmode:
        startframe = np.linspace(0, num_frame-L, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(num_frame-L))])
    feats = []
    for asf in startframe:
        feats.append(feat[:, int(asf):int(asf)+L])
    if evalmode is False:
        feats = feats[0]
    else:
        feats = np.stack(feats, axis=0)
    return feats

# 舍弃
class voxceleb2(Dataset):
    """Sample format: speaker, wav_path
    Example:
    id00012 id00012/21Uxsk56VDQ/00001.wav
    """

    def __init__(self, datalst: np.ndarray, speaker: dict, load_wav):
        super(voxceleb2, self).__init__()
        self.datalst = datalst
        self.speaker = speaker
        self.load_wav = load_wav

    def __len__(self):
        return len(self.datalst)

    def __getitem__(self, idx):
        spk, path = self.datalst[idx]
        label = self.speaker[spk]
        wav = self.load_wav(path)
        wav = torch.FloatTensor(wav)
        return {"label": label, "input": wav}

# 舍弃
class vox2trials(Dataset):
    """Trial format: label enroll test
    Example:
    1 voxceleb1/id10136/kDIpLcekk9Q/00011.wav voxceleb1/id10136/kTkrpBqwqJs/00001.wav
    """

    def __init__(self, trials: np.ndarray):
        super(vox2trials, self).__init__()
        self.trials = trials

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        label, enroll, test = self.trials[idx]
        return {"label": label, "enroll": enroll, "test": test}

# 舍弃
class voxsampler(Sampler):
    """Indices Resampler for dataset that ensure there is no repeated
    speakers in a batch.
    """

    def __init__(self, data_source: pd.DataFrame, speaker_id: dict,
                 max_utt_per_spk: int = 100, batch_size: int = 128):
        self.data_source = data_source.values
        self.speaker_id = speaker_id
        self.data_dict = data_source.groupby(by="speaker").groups.items()
        self.max_utt_per_spk = max_utt_per_spk
        self.batch_size = batch_size

    def __iter__(self):
        flattened_list = []
        # Data for each class
        for speaker, indices in self.data_dict:
            label = self.speaker_id[speaker]
            numUtt = min(len(indices), self.max_utt_per_spk)
            rp = np.random.permutation(indices)[:numUtt]
            # it may be further accelarated.
            flattened_list.extend(rp.tolist())
        # Data in random order
        mixid = np.random.permutation(len(flattened_list))
        mixlabel = []
        mixmap = []
        # Prevent two pairs of the same speaker in the same batch
        for i in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            idx = flattened_list[i]
            speaker = self.speaker_id[self.data_source[idx][0]]
            if speaker not in mixlabel[startbatch:]:
                mixlabel.append(speaker)
                mixmap.append(idx)
        # Iteration size
        self.num_file = len(mixmap)
        self.files = mixmap
        return iter(mixmap)

    def __len__(self):
        return len(self.data_source)

# 舍弃
def loadWAV(filename, L=32240, evalmode=True, num_eval=10):
    audio, sr = sf.read(filename, dtype='int16')
    assert sr == 16000, "sample rate is {} != 16000".format(sr)
    audiosize = audio.shape[0]
    if audiosize <= L:
        shortage = math.floor((L - audiosize + 1) / 2)
        audio = np.pad(audio, (shortage, shortage),
                       'constant', constant_values=0)
        audiosize = audio.shape[0]
    if evalmode:
        startframe = np.linspace(0, audiosize-L, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-L))])
    wave = []
    for asf in startframe:
        wave.append(audio[int(asf):int(asf)+L])
    if evalmode is False:
        wave = wave[0]
    else:
        wave = np.stack(wave, axis=0)
    return wave

# 舍弃
def load_train(trainlst, traindir, maptrain5994, L=L,
               batch_size=batch_size, num_worker=num_worker,
               max_utt_per_spk=max_utt_per_spk, load_wav=None):
    """dataloader for training dataset via voxsampler.
    """
    if load_wav is None:
        def load_train_wav(path): return loadWAV(path, L=L, evalmode=False)
    else:
        load_train_wav = load_wav
    df_train = pd.read_csv(trainlst, sep=" ", header=None,
                           names=["speaker", "file"])
    df_train["file"] = df_train["file"].apply(lambda x: traindir + x)
    map_train = dict(pd.read_csv(maptrain5994, header=None).values)
    data = voxceleb2(df_train.values, map_train, load_train_wav)
    sampler = voxsampler(df_train, map_train,
                         max_utt_per_spk=max_utt_per_spk, batch_size=batch_size)
    dataloader = DataLoader(data, batch_size=batch_size,
                            num_workers=num_worker, shuffle=False,
                            sampler=sampler)
    return dataloader

# 舍弃 -> VerificationTrials
def load_trial(testlst, testdir,
               batch_size=batch_size, num_worker=num_worker):
    """dataloader via vox2trials.
    Format: label enroll test
        where label is 0 or 1, enroll is the path of enrollment, 
        test is the path of test.
    """
    df_trials = pd.read_csv(testlst, sep=" ", header=None, names=[
                            "label", "enrollment", "test"])
    df_trials["enrollment"] = df_trials["enrollment"].apply(
        lambda x: testdir + x)
    df_trials["test"] = df_trials["test"].apply(lambda x: testdir + x)
    trials_set = vox2trials(df_trials.values)
    trials_loader = DataLoader(trials_set, batch_size=batch_size,
                               num_workers=num_worker, shuffle=False)
    return trials_loader

##########
# 模型准备
##########

# 核心单元

# model.TDNNBlocks
class TDNNBlock(nn.Module):

    def __init__(self, input_dim=40, output_dim=128, context_size=5, stride=1, dilation=1,
                 batch_norm=True, dropout_p=0.0, last_time=False):
        """
        input_dim : int
            F - The size of the feature of inputs, ca
        output_dim : int
            F - The size of the feature of outputs.
        last_time : bool, default False
            T - The time dimension is at the last dimension of inputs.
        """
        super(TDNNBlock, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm
        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)
        self.last_time = last_time

    def forward(self, x):
        '''
        input : tensor (B, F, T) or (B, T, F)
        output : tensor (B, F, T) or (B, T, F)
        '''
        if self.last_time is True:
            x = x.permute(0, 2, 1)
        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(
            self.input_dim, d)
        x = x.unsqueeze(1)
        # Unfold input into smaller temporal contexts
        x = F.unfold(x, (self.context_size, self.input_dim),
                     stride=(1, self.input_dim), dilation=(self.dilation, 1))
        # N, output_dim * context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x)
        x = self.nonlinearity(x)
        if self.dropout_p:
            x = self.drop(x)
        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)
        if self.last_time is True:
            x = x.permute(0, 2, 1)
        return x

# models.ResNetBlocks
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

# models.ResNetBlocks
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 池化单元

# models.TemporalPool
class StatsPool(nn.Module):
    '''
    Source: https://github.com/cvqluu/Factorized-TDNN
    '''

    def __init__(self, floor=1e-10, bessel=False):
        super(StatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features[1, *])
        outpu: size (batch, output_features[1, *])
        '''
        means = torch.mean(x, dim=1)
        t = x.shape[1]
        if self.bessel:
            t = t - 1
        residuals = x - means.unsqueeze(1)
        numerator = torch.sum(residuals**2, dim=1)
        stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        x = torch.cat([means, stds], dim=1)
        return x

# models.TemporalPool
class SAP(nn.Module):
    '''Self attention pooling layer
    '''

    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
        self.sap_linear = nn.Linear(in_dim, in_dim)
        self.attention = self.new_parameter(in_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):
        '''Self attention pooling over time/frames
        x : tensor (batch, L, D)
        '''
        # self attention pooling over time/frames
        h = torch.tanh(self.sap_linear(x))  # batch * L * D' (D' = D)
        w = torch.matmul(h, self.attention).squeeze(dim=2)  # batch * L
        w = self.softmax(w).unsqueeze(2)  # batch * L * 1
        x = torch.sum(x * w, dim=1)  # batch * D
        return x

# 嵌入网络

# models.ResNetSE34
class ResNetSE(nn.Module):
    """来自 VoxSRC 2020，Github: voxceleb_trainer 提供。
    """

    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', **kwargs):

        print('Embedding size is %d, encoder %s.' % (nOut, encoder_type))

        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        super(ResNetSE, self).__init__()

        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(
            block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(
            block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(
            block, num_filters[3], layers[3], stride=(1, 1))

        self.avgpool = nn.AvgPool2d((5, 1), stride=1)

        self.instancenorm = nn.InstanceNorm1d(40)
        self.torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                                            n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40)

        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(
                num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(
                num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x):
        """
        x (tensor) : B * T
        """
        x = self.torchfb(x) + 1e-6  # B * F * T
        x = self.instancenorm(x.log()).unsqueeze(1).detach()  # B * 1 * F * T

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        if self.encoder_type == "SAP":
            x = x.permute(0, 2, 1, 3)
            x = x.squeeze(dim=1).permute(0, 2, 1)  # batch * T * C
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

# models.ResNetSE34
def ResNetSE34L(nOut=256, num_filters=[16, 32, 64, 128], **kwargs):
    # Number of filters
    layers = [3, 4, 6, 3]
    model = ResNetSE(SEBasicBlock, layers, num_filters, nOut, **kwargs)
    return model


##########
# 训练与测试
##########

# metrics
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# loss.aamsoftmax
class AAMSoftmax(nn.Module):
    def __init__(self,
                 in_feats,
                 n_classes=10,
                 m=0.2,
                 s=30,
                 easy_margin=False):
        super(AAMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.weight = torch.nn.Parameter(torch.FloatTensor(
            n_classes, in_feats), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        print('Initialized AAMSoftmax m = %.3f s = %.3f' % (self.m, self.s))

    def forward(self, x, label=None):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        prec1, = accuracy(output.detach().cpu(),
                          label.detach().cpu(), topk=(1,))
        return loss, prec1

# models.amsoftmax
class AMSoftmax(nn.Module):
    def __init__(self, in_feats, n_classes=10, m=0.2, s=30):
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(
            in_feats, n_classes), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialized AMSoftmax margin = %.3f scale = %.3f' %
              (self.m, self.s))

    def forward(self, x, label=None):
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)

        if not self.training:
            return F.softmax(costh, dim=1)

        label_view = label.view(-1, 1)
        if label_view.is_cuda:
            label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda:
            delt_costh = delt_costh.to(x.device)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        prec1, = accuracy(costh_m_s.detach().cpu(),
                          label.detach().cpu(), topk=(1,))
        return loss, prec1

# loss.softmax
class SoftmaxLoss(nn.Module):
    def __init__(self, in_feats, n_classes=10):
        super(SoftmaxLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc = nn.Linear(in_feats, n_classes)

        print('Initialized Softmax Loss')

    def forward(self, x, label=None):
        x = self.fc(x)
        if label is None:
            return x
        loss = self.criterion(x, label)
        prec1, = accuracy(x.detach().cpu(),
                          label.detach().cpu(), topk=(1,))
        return loss, prec1

# loss.triplet
class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, hard_rank=0, hard_prob=0):
        super(TripletLoss, self).__init__()
        self.hard_rank = hard_rank
        self.hard_prob = hard_prob
        self.margin = margin

        print('Initialised Pairwise Loss - Triplet margin={}, hard={}, and probability={}'.format(
            margin, hard_rank, hard_prob))

    def forward(self, x, label=None):
        anchors = x[:, 0, :]  # anchor
        positives = x[:, 1, :]  # positive
        anchors = F.normalize(anchors, p=2, dim=1)
        positives = F.normalize(positives, p=2, dim=1)
        pair_similarity = -1 * (F.pairwise_distance(anchors.unsqueeze(-1),
                                                    positives.unsqueeze(
                                                        -1).transpose(0, 2),
                                                    p=2))
        negidx = self.mineHardNegative(pair_similarity.detach())
        negatives = positives[negidx, :]  # negative
        labelnp = np.array([1] * len(positives) + [0] * len(negatives))
        # calculate distances
        pos_dist = F.pairwise_distance(anchors, positives, p=2)
        neg_dist = F.pairwise_distance(anchors, negatives, p=2)
        loss = torch.mean(
            F.relu(pos_dist.pow(2) - neg_dist.pow(2) + self.margin))
        scores = -1 * torch.cat([pos_dist, neg_dist],
                                dim=0).detach().cpu().numpy()
        eer, _ = calculate_eer(labelnp, scores)
        return loss, torch.tensor(eer)

    def mineHardNegative(self, pair_similarity):
        """Hard negative mining: Mine hard negative embeddings from other 
            positive embeddings.
            1. Semi hard negative mining: find negative that is higher than 
               positive.
            2. Rank based negative mining: find negative that is in [0, rank].
        """
        negidx = []
        for idx, similarity in enumerate(pair_similarity):
            simval, simidx = torch.sort(similarity, descending=True)
            if self.hard_rank < 0:
                # Semi hard negative mining
                semihardidx = simidx[(
                    similarity[idx] - self.margin < simval) & (simval < similarity[idx])]
                if len(semihardidx) == 0:
                    negidx.append(random.choice(simidx))
                else:
                    negidx.append(random.choice(semihardidx))
            else:
                # Rank based negative mining
                simidx = simidx[simidx != idx]
                if random.random() < self.hard_prob:
                    negidx.append(simidx[random.randint(0, self.hard_rank)])
                else:
                    negidx.append(random.choice(simidx))
        return negidx

# loss.angleproto
class AngularProto(nn.Module):
    """Angular Prototypical.
        Smaller distance between a query set and a support set means that
        the query set from softmax can achieve more importance/weight.
    """

    def __init__(self, init_w=10.0, init_b=-5.0):
        super(AngularProto, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(init_b), requires_grad=True)
        self.ce = torch.nn.CrossEntropyLoss()
        print("Initialized Angular Prototypical Loss with w={} and b={}".format(
              init_w, init_b))

    def forward(self, x, label=None):
        """
        x : 3-dim tensor
            A pair of batch embeddings with shape (B, M, D), where M is 
            usually not less than 2, as M >= 2. Notably, each embedding
            generally comes from different speakers.
        """
        supports = torch.mean(x[:, 1:, :], 1)  # support set
        query = x[:, 0, :]  # query set
        batch_size = x.shape[0]
        cos_sim_matrix = F.cosine_similarity(
            query.unsqueeze(-1), supports.unsqueeze(-1).transpose(0, 2))
        cos_sim_matrix = cos_sim_matrix * torch.clamp(self.w, 1e-6) + self.b
        label = torch.from_numpy(np.asarray(range(0, batch_size))).to(x.device)
        loss = self.ce(cos_sim_matrix, label)
        prec1, = accuracy(cos_sim_matrix.detach().cpu(),
                          label.detach().cpu(), topk=(1,))
        return loss, prec1

# loss.angleproto
class SoftmaxAngleProto(nn.Module):
    """
    This loss function is modify from
    https://github.com/clovaai/voxceleb_trainer/blob/master/loss/softmaxproto.py
    """

    def __init__(self, in_feats, n_classes=10):
        super(SoftmaxAngleProto, self).__init__()
        self.softmax    = SoftmaxLoss(in_feats=in_feats, n_classes=n_classes)
        self.angleproto = AngularProto()
        print('Initialized SoftmaxPrototypical Loss')

    def forward(self, x, label=None):
        assert x.size()[1] == 2
        nlossS, prec1 = self.softmax(x.reshape(-1,x.size()[-1]), label.repeat_interleave(2))
        nlossP, _     = self.angleproto(x, None)
        return nlossS + nlossP, prec1


#########
# 性能评估
#########

# metrics
def calculate_val(y, y_score, far=1e-3):
    """Calculate validation rate (VAL) at a specified 
        false accept rate (FAR).

        Inspired by [1].
        [1] Zhang, C., Koishida, K., 2017. End-to-End Text-Independent 
        Speaker Verification with Triplet Loss on Short Utterances, in: 
        Interspeech 2017. pp. 1487–1491. 
        https://doi.org/10.21437/Interspeech.2017-1608

        Parameter
        ---------
            y : array
                A list of label {0, 1} corresponding to 
                each trial.
            y_score : array
                A list of score corresponding to each trial.
            far : float
                Specified false accept rate.

        Return
        ------
            VAL : float
                Validation rate at a specified FAR.
            threshold : float
                Threshold such that satisfies the given very low FAR.
    """
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
    elif y_score.ndim == 2:
        VAL_thresh = [calculate_val(y, y_score[:, i], far=far)
                      for i in range(y_score.shape[1])]
        VAL_thresh = np.array(VAL_thresh)
        best_mode = VAL_thresh[:, 0].argmin()
        VAL, threshold = VAL_thresh[best_mode, 0], VAL_thresh[best_mode, 1]
    return VAL, threshold

# scores
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
                test_feat.unsqueeze(-1).transpose(0, 2)).mean().detach().cpu().numpy()
            score = -dist.item()
        elif mode is "cosine":
            dist = F.cosine_similarity(
                enroll_feat.unsqueeze(-1),
                test_feat.unsqueeze(-1).transpose(0, 2)).mean().detach().cpu().numpy()
            score = dist.item()
    elif isinstance(mode, list):
        score = [calculate_score(enroll_feat, test_feat, m) for m in mode]
    return score

# metrics
def calculate_eer(y, y_score, pos=1):
    """Calculate Equal Error Rate (EER).

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
    if y_score.ndim == 1:
        fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=pos)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        thresh = interp1d(fpr, thresholds)(eer)
    elif y_score.ndim == 2:
        eer_thresh = [calculate_eer(y, y_score[:, i], pos)
                      for i in range(y_score.shape[1])]
        eer_thresh = np.array(eer_thresh)
        best_mode = eer_thresh[:, 0].argmin()
        eer, thresh = eer_thresh[best_mode, 0], eer_thresh[best_mode, 1]
    return eer, np.float(thresh)

# metrics
def compute_error_rates(scores, labels):
    """Creates a list of false-negative rates, a list of false-positive rates
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
    sorted_labels = []
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

# metrics
def compute_mindcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1, mode="vectorized"):
    """Computes the minimum of the detection cost function.  The comments refer 
        to equations in Section 3 of the NIST 2016 Speaker Recognition 
        Evaluation Plan.

        Copyright
        ---------
            def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
                ...
            2018 David Snyder: 
            This script is modified from: https://github.com/kaldi-asr/kaldi/blob/8ce3a95761e0eb97d95d3db2fcb6b2bfb7ffec5b/egs/sre08/v1/sid/compute_min_dcf.py
    """
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
    elif mode is "for":
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

# metrics
def calculate_mindcf(y, y_score, p_target=0.05, c_miss=1, c_fa=1):
    """Calculate MinDCF with p_target, c_miss, and c_fa in the NIST 2016 
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
    if y_score.ndim == 1:
        fnrs, fprs, thresholds = compute_error_rates(y_score, y)
        mindcf, threshold = compute_mindcf(fnrs, fprs, thresholds,
                                           p_target=p_target,
                                           c_miss=c_miss,
                                           c_fa=c_fa)
    elif y_score.ndim == 2:
        dcf_thresh = [calculate_mindcf(y, y_score[:, i],
                                       p_target=p_target,
                                       c_miss=c_miss,
                                       c_fa=c_fa)
                      for i in range(y_score.shape[1])]
        dcf_thresh = np.array(dcf_thresh)
        best_mode = dcf_thresh[:, 0].argmin()
        mindcf, threshold = dcf_thresh[best_mode, 0], dcf_thresh[best_mode, 1]
    return mindcf, threshold

# scores
def ztnorm(scores: pd.DataFrame,
           cohort: dict,
           nTop: int = None):
    """(Top) Score normalization using ZT-Norm where Z-Norm followed by T-Norm.

        Parameter
        ---------
            scores : DataFrame
                Scores of trials are normalized with columns {"enroll", "test", 
                "score"}.
            enroll_cohort : DataFrame
                Scores of pairs (enroll, cohort) are normalized with columns 
                {"enroll", "cohort", "score"}.
            test_cohort : DataFrame
                Scores of pairs (test, cohort) are normalized with columns {"test", "cohort", "score"}.
            nTop : int
                The number of selected samples, usually top X scoring or most 
                similar samples, where X is set to be, e.g., 200 ~ 400.

        Return
        ------
            scores : DataFrame
                Normalized scores of trials with columns {"enroll", "test", 
                "score", "ZT-Norm"}.
    """
    scores_mean = {}
    scores_std = {}
    for key, value in cohort.items():
        if nTop is not None:
            value = np.sort(value, kind="mergesort")
            if nTop < 0:
                value = value[: -nTop]
            else:
                value = value[nTop:]
        scores_mean[key] = value.mean()
        scores_std[key] = value.std()

    def score_ztnorm(row):
        """
        row : DataFrame.row {"enroll", "test", "score"}
        """
        enroll, test, orig_score = row["enroll"], row["test"], row["score"]
        if scores_std[enroll] != 0:
            znorm_score = (
                orig_score - scores_mean[enroll]) / scores_std[enroll]
        else:
            znorm_score = orig_score
        if scores_std[test] != 0:
            ztnorm_score = (znorm_score - scores_mean[test]) / scores_std[test]
        else:
            ztnorm_score = znorm_score
        return ztnorm_score
    norm_scores = scores.apply(lambda row: score_ztnorm(row), axis=1)
    return norm_scores

# scores
def snorm(scores: pd.DataFrame,
          cohort: dict):
    """S-Norm

        Parameter
        ---------
            scores : DataFrame
                Scores of trials are normalized with columns {"enroll", "test", 
                "score"}.
            enroll_cohort : DataFrame
                Scores of pairs (enroll, cohort) are normalized with columns 
                {"enroll", "cohort", "score"}.
            test_cohort : DataFrame
                Scores of pairs (test, cohort) are normalized with columns {"test", "cohort", "score"}.

        Return
        ------
            scores : DataFrame
                Normalized scores of trials with columns {"enroll", "test", 
                "score", "S-Norm"}.
    """

    scores_mean = {}
    scores_std = {}
    for key, value in cohort.items():
        scores_mean[key] = value.mean()
        scores_std[key] = value.std()

    def average_ztnorm(row):
        """
        row : DataFrame.row {"enroll", "test", "score"}
        """
        enroll, test, orig_score = row["enroll"], row["test"], row["score"]
        if scores_std[enroll] != 0:
            znorm_score = (
                orig_score - scores_mean[enroll]) / scores_std[enroll]
        else:
            znorm_score = orig_score
        if scores_std[test] != 0:
            tnorm_score = (orig_score - scores_mean[test]) / scores_std[test]
        else:
            tnorm_score = orig_score
        snorm_score = 0.5 * (znorm_score + tnorm_score)
        return snorm_score
    norm_scores = scores.apply(lambda row: average_ztnorm(row), axis=1).values
    return norm_scores

# scores
def asnorm(scores: pd.DataFrame,
           cohort: dict,
           nTop: int = 300):
    """Adaptive score normalization with top n cohort samples.

        Parameter
        ---------
            scores : DataFrame
                Scores of trials are normalized with columns {"enroll", "test", 
                "score"}.
            enroll_cohort : DataFrame
                Scores of pairs (enroll, cohort) are normalized with columns 
                {"enroll", "cohort", "score"}.
            test_cohort : DataFrame
                Scores of pairs (test, cohort) are normalized with columns {"test", "cohort", "score"}.
            nTop : int
                The number of selected samples, usually top X scoring or most 
                similar samples, where X is set to be, e.g., 200 ~ 400.
                If nTop < 0: select the nTop most disimiliar scores.
                If nTop > 0: select the nTop most similiar scores.

        Return
        ------
            scores : DataFrame
                Normalized scores of trials with columns {"enroll", "test", 
                "score", "AS-Norm"}.
    """

    scores_mean = {}
    scores_std = {}
    for key, value in cohort.items():
        value = np.sort(value, kind="mergesort")
        if nTop < 0:
            value = value[: -nTop]
        else:
            value = value[nTop:]
        scores_mean[key] = value.mean()
        scores_std[key] = value.std()

    def average_ztnorm(row):
        """
        row : DataFrame.row {"enroll", "test", "score"}
        """
        enroll, test, orig_score = row["enroll"], row["test"], row["score"]
        if scores_std[enroll] != 0:
            znorm_score = (
                orig_score - scores_mean[enroll]) / scores_std[enroll]
        else:
            znorm_score = orig_score
        if scores_std[test] != 0:
            tnorm_score = (orig_score - scores_mean[test]) / scores_std[test]
        else:
            tnorm_score = orig_score
        snorm_score = 0.5 * (znorm_score + tnorm_score)
        return snorm_score
    norm_scores = scores.apply(lambda row: average_ztnorm(row), axis=1).values
    return norm_scores

#########
# 模型训练
#########

def lr_gamma(initial_lr=1e-5, last_lr=1e-3, total=50, step=5, mode="log10"):
    """
    Compute the gamma of learning rate to be stepped, where gamma can be 
        larger or equal than 1 or less than 1.
    """
    num_step = (total // step) - 1
    assert num_step > 0
    gamma = (last_lr / initial_lr) ** (1/num_step)
    return gamma

# 舍弃
def extract_embedding(embed_net, utts, batch_size,
                      embed_norm, device, L=None,
                      load_wav=None):
    if load_wav is None and L is not None:
        def load_wav(path): return loadWAV(
            path, L=L, evalmode=False)
    wav_file = sorted(list(set(utts)))
    wav_list = Utterance(wav_file, load_wav)

    wav_loader = DataLoader(wav_list, batch_size=batch_size,
                            num_workers=4, shuffle=False)
    embeddings = {}
    with torch.no_grad():
        embed_net.to(device)
        embed_net.eval()
        for data in tqdm(wav_loader, total=len(wav_loader)):
            file = data["file"]
            wave = data["wave"].to(device)
            feat = embed_net(wave)
            if embed_norm is True:
                embed = F.normalize(feat, p=2, dim=1).detach().cpu()
            for i in range(0, embed.shape[0]):
                embeddings[file[i]] = embed[i].clone()
    return embeddings


def evaluate(embed_net, trials_loader, mode="norm2", batch_size=30,
             embed_norm=True, load_wav=None,
             len_eval=300, num_eval=6,
             device="cpu", trials_feat=None):
    """Evaluate function for embedding network.

        Example
        -------
            device = "cuda:2"
            eval_L = 48240
            mode = "norm2"
            eval_batch_size = 30
            embed_norm = True
            ret = evaluate(net, trials_loader, mode="cosine",
                           eval_L=eval_L, num_eval=num_eval, device=device,
                           batch_size=eval_batch_size, embed_norm=embed_norm)
    """
    def _prefeat(embed_net, trials_loader, batch_size=1, embed_norm=True,
                 load_wav=None, len_eval=None, num_eval=None, device="cpu"):
        if load_wav is None:
            def load_trial_wav(path): return loadWAV(
                path, L=L, evalmode=True, num_eval=num_eval)
        else:
            load_trial_wav = load_wav
        wav_file = [[trials_loader.dataset[i]["enroll"],
                     trials_loader.dataset[i]["test"]]
                    for i in range(len(trials_loader.dataset))]
        wav_file = sorted(list(set(np.concatenate(wav_file).tolist())))
        trials = Utterance(wav_file, load_trial_wav)
        trialoader = DataLoader(trials, batch_size=batch_size,
                                num_workers=5, collate_fn=trials.collate_fn,
                                shuffle=False)
        trials_feat = {}
        with torch.no_grad():
            embed_net.to(device)
            embed_net.eval()
            for data in tqdm(trialoader, total=len(trialoader)):
                file = data["file"]
                wave = data["wave"].to(device)
                feat = embed_net(wave)
                if embed_norm is True:
                    feat = F.normalize(feat, p=2, dim=1).detach().cpu()
                for i, j in enumerate(range(0, feat.shape[0], trials.num_chunk)):
                    trials_feat[file[i]] = feat[j: j +
                                                trials.num_chunk].clone()
        return trials_feat

    if trials_feat is None:
        trials_feat = _prefeat(embed_net, trials_loader, batch_size,
                               embed_norm, load_wav, len_eval, num_eval, device)
    all_scores = []
    all_labels = []
    all_trials = []
    for trial in tqdm(trials_loader, total=len(trials_loader)):
        label = trial["label"]
        enroll = trial["enroll"]
        test = trial["test"]
        for i in range(len(label)):
            enroll_embed = trials_feat[enroll[i]]
            test_embed = trials_feat[test[i]]
            score = calculate_score(enroll_embed, test_embed, mode=mode)
            all_scores.append(score)
            all_trials.append([enroll[i], test[i]])
        all_labels.extend(label.numpy().tolist())
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    eer, thresh = calculate_eer(all_labels, all_scores, 1)
    dcf, dcf_threshold = calculate_mindcf(all_labels, all_scores)
    return eer, thresh, all_scores, all_labels, all_trials, trials_feat, dcf, dcf_threshold


class SpeakerNet(nn.Module):
    def __init__(self,
                 device=device,
                 log_dir=log_dir,
                 eval_interval=eval_interval,
                 optimizer="Adam",
                 lr=lr,
                 gamma=gamma,
                 top=None,
                 net=None):
        super(SpeakerNet, self).__init__()
        self.net = nn.Sequential(net, top).to(device)
        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        else:
            self.optimizer = optim.SGD(
                self.net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
        self.lr_step = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=gamma)
        self.TBoard = SummaryWriter(log_dir=log_dir)
        self.device = device
        self.eval_interval = eval_interval
        self.log_dir = log_dir

    def train(self, dataloader, trials_loader=None,
              num_epoch=num_epoch, str_epoch=0, step_num=0,
              mode="norm2", eval_batch_size=30, embed_norm=True,
              eval_load_wav=None, len_eval=len_eval, num_eval=num_eval):
        modelst = []
        for epoch in range(str_epoch, num_epoch):
            self.net.train()
            pbar = tqdm(enumerate(dataloader), desc="Epoch %03d" % (epoch + 1),
                        total=dataloader.sampler.num_file//dataloader.batch_size)
            for i_step, batch in pbar:
                self.optimizer.zero_grad()
                label = batch["label"].to(self.device)
                if isinstance(batch["input"], torch.Tensor):
                    inp = batch["input"].to(self.device)
                    embed = self.net[0](inp)
                elif isinstance(batch["input"], list):  # multi-utterance
                    embed = []
                    for inp in batch["input"]:
                        inp = inp.to(self.device)
                        outp = self.net[0](inp)
                        embed.append(outp)
                    embed = torch.stack(embed, dim=1).squeeze()
                else:
                    raise TypeError(
                        "The type of input is {}.".format(type(batch["input"])))
                loss, prec1 = self.net[1](embed, label)
                loss.backward()
                self.optimizer.step()
                # TBoard
                step_num += 1
                self.TBoard.add_scalar(
                    'Metrics/train_loss', loss.item(), step_num)
                self.TBoard.add_scalar(
                    'Metrics/lr', self.lr_step.get_last_lr()[0], step_num)
                self.TBoard.add_scalar(
                    'Metrics/top1_acc', prec1.item(), step_num)
                # pbar
                pbar.set_postfix({"step": step_num,
                                  "train_loss": loss.item(),
                                  "train_acc": prec1.item()})

            if (epoch + 1) % self.eval_interval == 0:
                self.net.eval()
                modeldir = "snapshot-epoch-%03d.model" % (epoch + 1)
                modeldir = os.path.join(self.log_dir, modeldir)
                torch.save(self.net.state_dict(), modeldir)
                if trials_loader is not None:
                    self.evaluate(trials_loader, epoch + 1, mode=mode,
                                  len_eval=len_eval, num_eval=num_eval,
                                  batch_size=eval_batch_size, embed_norm=embed_norm,
                                  load_wav=eval_load_wav)
                self.lr_step.step()
                modelst.append(modeldir)
        return modelst, step_num, loss.item(), prec1

    # SpeakerNet.evaluateFromList
    def _prefeat(self, trials_loader, batch_size=1, embed_norm=True,
                 load_wav=None, len_eval=None, num_eval=None):
        if load_wav is None:
            def load_trial_wav(path): return loadWAV(
                path, L=len_eval, evalmode=True, num_eval=num_eval)
        else:
            load_trial_wav = load_wav
        wav_file = [[trials_loader.dataset[i]["enroll"],
                     trials_loader.dataset[i]["test"]]
                    for i in range(len(trials_loader.dataset))]
        wav_file = sorted(list(set(np.concatenate(wav_file).tolist())))
        trials = Utterance(wav_file, load_trial_wav)
        trialoader = DataLoader(trials, batch_size=batch_size,
                                num_workers=5, collate_fn=trials.collate_fn, shuffle=False)
        trials_feat = {}
        with torch.no_grad():
            self.net.eval()
            for data in tqdm(trialoader, total=len(trialoader)):
                file = data["file"]
                wave = data["wave"].to(self.device)
                feat = self.net[0](wave)
                if embed_norm is True:
                    feat = F.normalize(feat, p=2, dim=1).detach().cpu()
                for i, j in enumerate(range(0, feat.shape[0], trials.num_chunk)):
                    trials_feat[file[i]] = feat[j: j +
                                                trials.num_chunk].clone()
        return trials_feat

    # SpeakerNet.evaluateFromList
    def evaluate(self, trials_loader, step_num=0, mode="norm2",
                 batch_size=30, embed_norm=True, load_wav=None,
                 len_eval=len_eval, num_eval=num_eval, trials_feat=None):
        if trials_feat is None:
            trials_feat = self._prefeat(
                trials_loader, batch_size, embed_norm, load_wav, len_eval, num_eval)
        all_scores = []
        all_labels = []
        all_trials = []
        for trial in tqdm(trials_loader, total=len(trials_loader)):
            label = trial["label"]
            enroll = trial["enroll"]
            test = trial["test"]
            for i in range(len(label)):
                enroll_embed = trials_feat[enroll[i]]
                test_embed = trials_feat[test[i]]
                score = calculate_score(enroll_embed, test_embed, mode=mode)
                all_scores.append(score)
                all_trials.append([enroll[i], test[i]])
            all_labels.extend(label.numpy().tolist())
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        eer, thresh = calculate_eer(all_labels, all_scores, 1)
        dcf, dcf_threshold = calculate_mindcf(all_labels, all_scores)
        self.TBoard.add_scalar('Metrics/EER', eer, step_num)
        self.TBoard.add_scalar('Metrics/DCF', dcf, step_num)
        return eer, thresh, all_scores, all_labels, all_trials, trials_feat, dcf, dcf_threshold

    def load(self, modeldir, strict=True):
        loaded_state = torch.load(modeldir)
        if strict is True:
            self.net.load_state_dict(loaded_state)
            print("Loaded model successfully.")
        else:
            self_state = self.net.state_dict()
            for name, param in loaded_state.items():
                origname = name
                if name not in self_state:
                    name = name.replace("module.", "")
                    if name not in self_state:
                        print("%s is not in the model." % origname)
                        continue
                if self_state[name].size() != loaded_state[origname].size():
                    print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                        origname, self_state[name].size(),
                        loaded_state[origname].size()))
                    continue
                self_state[name].copy_(param)


def load_weights(net, path):
    """Load weights of model from the given path.

        Parameter
        ---------
            net : torch.nn.Module
                A pytorch network.
            path : str
                The path of state dict.
    """
    self_state = net.state_dict()
    loaded_state = torch.load(path, map_location="cpu")
    for name, param in loaded_state.items():
        origname = name
        if name not in self_state:
            name = name.replace("module.", "")
            if name not in self_state:
                name = name.replace("__S__.", "")
                if name not in self_state:
                    name = name.replace("__L__.", "")
                    if name not in self_state:
                        print("%s is not in the model." % origname)
                        continue
        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s" % (
                origname, self_state[name].size(),
                loaded_state[origname].size()))
            continue
        self_state[name].copy_(param)

# waveforms
def utt2dur(utterances):
    """
    Compute duration of utterances.
    
    Parameter
    ---------
        utterances : array list
    
    Return
    ------
        durations : array
    """
    durations = np.empty(len(utterances))
    for i, wav in tqdm(enumerate(utterances), total=len(utterances)):
        sr, audio = wavfile.read(wav)
        durations[i] = audio.shape[0]
    return durations


# database
def trials2utterance(trials):
    """
    Unique utterances from trials in the format of VerificationTrials(Dataset).
    
    Parameter
    ---------
        trials: VerificationTrials {[label], enroll, test}
            
    Return
    ------
        utterances: list
    """
    utterances = [[trials[i]["enroll"], trials[i]["test"]] for i in range(len(trials))]
    utterances = sorted(list(set(np.concatenate(utterances).tolist())))
    return utterances

# database
class VerificationTrials(Dataset):
    """
    Verification Trial format: label enroll test
    Example
    -------
    1 voxceleb1/id10136/kDIpLcekk9Q/00011.wav voxceleb1/id10136/kTkrpBqwqJs/00001.wav
    """

    def __init__(self, 
                 trials_lst: str = None, 
                 trials_dir: str = None, 
                 trials: np.ndarray = None,
                 mode : int = 0):
        super(VerificationTrials, self).__init__()
        self.mode = mode
        if trials is not None:
            self.trials = trials
        else:
            self.trials = self.load_trial(trials_lst, trials_dir)
        
    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        if self.mode == 0:
            label, enroll, test = self.trials[idx]
            return {"label": label, "enroll": enroll, "test": test}
        elif self.mode == 1:
            enroll, test = self.trials[idx]
            return {"enroll": enroll, "test": test}

    def load_trial(self, trials_lst, trials_dir):
        """
        Load trials from list file appended prefix trials_dir.
        Format: label enroll test
            seperated by spaces,
            label is 0 or 1, 
            enroll is a path of a enrollment utterance, 
            test is a path of a test utterance.
        """
        cols = ["label", "enroll", "test"][self.mode:]
        trials = pd.read_csv(trials_lst, sep=" ", header=None, names=cols)
        trials["enroll"] = trials["enroll"].apply(lambda x: trials_dir + x)
        trials["test"] = trials["test"].apply(lambda x: trials_dir + x)
        return trials[cols].values

# vectors
def wav_embedding(embed_net, trials, load_wav=None, embed_norm=True, device="cpu",
                  n_jobs=4, batch_size=1):
    """
    Extract embeddings from trials via mode. There is no repeated utterances
        computed by filtering them, so the computation is efficient.
        There is 2 mode: 
        1. batch-style implementation enable GPU,
        2. multi-cpu implementation via joblib in multiple threading.
        
    Parameter
    ---------
        embed_net : torch.nn.Module
            A model to compute embeddings of utterances.
        trials : torch.Dataset {label, enroll, test} or a list
            A trials list or a utterance list.
        load_wav : function
            A function is used to read waveform. This function is essential 
            to evaluting, so consider it.
        embed_norm : bool
            Determine whether normlization embedding using norm2.
        device : str
            CPU or GPU, e.g., cpu or cuda:0
        len_eval : int
            The length of waveform to be read.
            If load_wav is None, it is set to read waveform using default way.
        num_eval : int
            The number of segments from a single utterance.
            If load_wav is None, it is set to read waveform using default way.
        n_jobs : int
            The number of subprocess.
        batch_size : int
            If batch_size is 1, the mode of computation is multi-threading 
            implementation. If batch_size > 1, the mode of computation is 
            batch-style implementation enable GPU.
            
    Return 
    ------
        trials_embed : dict {utterance: embedding}
    """
    
    if load_wav is None:
        def load_wav(path): 
            return read(path, num_samples=48000, mode_eval=True, num_eval=6, 
                        sample_rate=16000, min_samples=48000)
    
    if not isinstance(trials, list):
        wav_file = trials2utterance(trials)
    else:
        wav_file = sorted(list(set(trials)))
    wav_file = Utterance(wav_file, load_wav)
    
    if batch_size == 1:
        trials_embed = {}

        with torch.no_grad():
            embed_net.to(device)
            embed_net.eval()

            def wav2embed(wav):
                file = wav["file"]
                wave = wav["wave"].to(device)
                embed = embed_net(wave)
                if embed_norm is True:
                    embed = F.normalize(embed, p=2, dim=1).detach().cpu()
                trials_embed[file] = embed.clone()
            
            if n_jobs > 1:
                with joblib.parallel_backend('threading', n_jobs=n_jobs):
                    Parallel(verbose=0)(
                        delayed(wav2embed)(wav) 
                        for wav in tqdm(wav_file, total=len(wav_file)))
            else:
                for wav in tqdm(wav_file, total=len(wav_file)):
                    wav2embed(wav) 
                        
    elif batch_size > 1:
        batch_size = max(2, batch_size)
        trialoader = DataLoader(wav_file, batch_size=batch_size,
                                collate_fn=wav_file.collate_fn,
                                num_workers=n_jobs, shuffle=False)
        trials_embed = {}
        with torch.no_grad():
            embed_net.to(device)
            embed_net.eval()
            for data in tqdm(trialoader, total=len(trialoader)):
                file = data["file"]
                wave = data["wave"].to(device)
                num  = data["num"]
                embed = embed_net(wave)
                if embed_norm is True:
                    embed = F.normalize(embed, p=2, dim=1).detach().cpu()
                else:
                    embed = embed.detach().cpu()
                start_i = 0
                i = 0
                while start_i < embed.shape[0]:
                    trials_embed[file[i]] = embed[start_i : start_i + num[i]].clone()
                    start_i += num[i]
                    i += 1
    return trials_embed

# scores
def trial_scoring(trials, trials_embed, mode="cosine", n_jobs=4):
    """
    Compute scores of trials based on pre-computed embeddings.
    
    Parameter
    ---------
        trials : Dataset {label, enroll, test}
        trials_embed : dict
            It consists of pairs of {utterance: embedding(s)}
        mode : (list of) str 
            The mode to calculate socres, e.g., cosine or norm2.
        n_jobs : int
            The number of multiple cpu.
            
    Return
    ------
        all_scores : 1-dim or 2-dim array
            scores corresponds to each trial. If mode is a list, it is 2-dim,
            each column corresponds to a mode. 
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
        enroll_embed = trials_embed[enroll]
        test_embed = trials_embed[test]
        score = calculate_score(enroll_embed, test_embed, mode=mode)
        return score, label, enroll, test
    
    if n_jobs > 1:
        with joblib.parallel_backend('threading', n_jobs=n_jobs):
            scored_trials = Parallel(verbose=0)(
                delayed(compute_score)(trial) 
                for trial in tqdm(trials, total=len(trials), desc="Compute Scores"))
    else:
        scored_trials = [compute_score(trial) 
                for trial in tqdm(trials, total=len(trials), desc="Compute Scores")]
    
    for row in scored_trials:
        score, label, enroll, test = row
        all_scores.append(score)
        all_labels.append(label)
        all_trials.append([enroll, test])
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    return all_scores, all_labels, all_trials

# vectors
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


def trial_evaluate(trials, trials_embed=None, mode="cosine",
             embed_net=None, load_wav=None, embed_norm=True, 
             device="cpu", n_jobs=4, batch_size=1, transform=None):
    if trials_embed is not None:
        pass
    else:
        assert embed_net is not None, "embed_net is None"
        trials_embed = wav_embedding(embed_net, trials, 
                                     load_wav=load_wav, 
                                     embed_norm=True, 
                                     device=device, 
                                     n_jobs=n_jobs, 
                                     batch_size=batch_size)
    transformed_embed = vector_transform(trials_embed, transform)
    all_scores, all_labels, all_trials = trial_scoring(
        trials, transformed_embed, mode=mode, n_jobs=n_jobs)
    EER, EER_threshold = calculate_eer(all_labels, all_scores)
    DCF, DCF_threshold = calculate_mindcf(all_labels, all_scores)
    return EER, EER_threshold, DCF, DCF_threshold, all_scores, all_labels, all_trials, trials_embed 

if __name__ == "__main__":
    trainlst = "/workspace/rwang/voxceleb/train_list.txt"
    testlst = "/workspace/rwang/VoxSRC2020/data/verif/trials.txt"
    traindir = "/workspace/rwang/voxceleb/voxceleb2/"
    testdir = "/workspace/rwang/voxceleb/"
    maptrain5994 = "/workspace/rwang/competition/voxsrc2020/maptrain5994.txt"
    train = load_train(trainlst=trainlst, traindir=traindir,
                       maptrain5994=maptrain5994)
    trial = load_trial(testlst=testlst, testdir=testdir)
    net = ResNetSE34L(nOut=512, num_filters=[16, 32, 64, 128])
    top = AMSoftmax(in_feats=512, n_classes=5994, m=0.2, s=30)
    snet = SpeakerNet(net=net, top=top)
    modelst, step_num, loss, prec1, prec5 = snet.train(train, num_epoch=1)
    eer, thresh, all_scores, all_labels, all_trials, trials_feat = snet.eval(
        trial, step_num=0, trials_feat=None)

