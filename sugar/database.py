"""
Dataset or Dataloader for utterances or trials from a given file.
Now, this interface support 3 formats:

    1. only files of utterances, e.g., 
        id00012/21Uxsk56VDQ/00001.wav
        id00012/21Uxsk56VDQ/00001.logfbank.npy

    2. labeled files of utterances, e.g.,
        id00012 id00012/21Uxsk56VDQ/00001.wav
        id00012 id00012/21Uxsk56VDQ/00001.logfbank.npy
    
    3. trials consists of pairs of utterancs, e.g.,
        1 voxceleb1/id10136/kDIpLcekk9Q/00011.wav voxceleb1/id10136/kTkrpBqwqJs/00001.wav
        voxceleb1/id10136/kDIpLcekk9Q/00011.wav voxceleb1/id10136/kTkrpBqwqJs/00001.wav
        Note that trials do not have the read function.

There are several dataset:
    1. Utterance: basic object
    2. LabeledUtterance: based on Utterance
    3. AugmentedUtterance: based on Utterance
    4. VerificationTrials: Trials object

Author: Wang Rui
Version: v0.1
Create Date: 2020/9/27
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
from .waveforms import loadWAV

__all__ = ['Utterance', 'LabeledUtterance', 'AugmentedUtterance', 'VerificationTrials',
           'UniqueSampler', 'DistributedUniqueSampler']

def txt2dict(txtfile, txtroot=''):
    """
    Load speaker and path to a dict.

        Example
        -------
            id000012 voxceleb2/id000012/xJsahkq01/00001.wav

        Parameter
        ---------
            txtfile : str
                An utterance list with speaker.
            txtroot : str
                The root directory of utterances.

        Return
        ------
            utts : dict {speaker: utterance list}
    """
    utts = {}
    with open(txtfile, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            spk, path = line.split(' ')
            path = path.replace('\n', '')
            if spk not in utts:
                utts[spk] = []
            utts[spk].append(os.path.join(txtroot, path))
    return utts

def txt2list(txtfile, txtroot=''):
    """
    Load (speaker and) path to a list.

        Example
        -------
            id000012 voxceleb2/id000012/xJsahkq01/00001.wav

        Parameter
        ---------
            txtfile : str
                An utterance list with speaker.
            txtroot : str
                The root directory of utterances.

        Return
        ------
            utts : list
    """
    utts = []
    with open(txtfile, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            path = line.split(' ')
            if len(path) == 1:
                spk, path = '', path[0]
            else:
                spk, path = path[0], path[1]
            path = path.replace('\n', '')
            utts.append(os.path.join(txtroot, path))
    return utts

class Utterance(Dataset):
    """
    Dataset for load utterances via a list. Data source format: 
        waveform or feature.
        This class supports multiple format in the same-length 
        segments via each utterance corresponding the number of
        segments.

        Example
        -------
            id00012/21Uxsk56VDQ/00001.wav
            id00012/21Uxsk56VDQ/00001.logfbank.npy
    """

    def __init__(self, datalst, num_samples=48000, mode_eval=False, num_eval=2, root_dir=''):
        """
        Parameter
        ---------
            datalst : np.ndarray or list
        """
        self.datalst = datalst
        self.num_samples = num_samples
        self.mode_eval = mode_eval
        self.num_eval = num_eval
        self.root_dir = root_dir
        # ".wav" inputs a waveform
        self.ndim = 1 if ".wav" in datalst[0] else 2

    def __len__(self):
        return len(self.datalst)

    def __getitem__(self, idx):
        path = self.datalst[idx]
        if self.root_dir:
            path = os.path.join(self.root_dir, path)
        inp  = loadWAV(path, self.num_samples, self.mode_eval, self.num_eval) 
        inp  = torch.FloatTensor(inp)
        return {"file": path, "input": inp}

    def collate_fn(self, batch):
        """
        Collect from a batch of waveforms/feature, where an utterance 
            can be chunked into several segments as read in evaluting mode, 
            e.g., An utterance is loaded 6 segments with all 48000 samples.
            Now support different number of chunks of each input as num.
        
            Parameter
            ---------
                batch : a list of dict {file : str, input : 2-dim tensor}
        """
        path = [item["file"] for item in batch]
        wave = torch.cat([item["input"] for item in batch], dim=0)
        num  = [item["input"].shape[0] if item["input"].ndim >= self.ndim else 1 for item in batch]
        return {"file": path, "input": wave, "num": num}

class LabeledUtterance(Dataset):
    """
    Labeled Utterance: add label to Utterance.
    """
    def __init__(self, 
                 utterances: Utterance, 
                 speaker: dict, 
                 speaker_loc=-3, 
                 use_file=False,
                 add_file=False):
        self.datalst = []
        self.datalab = []
        self.speaker = speaker
        self.read = utterances.read
        self.ndim = 1 if ".wav" in utterances.datalst[0] else 2
        self.utterances = utterances # reserverd Utterance for computing embeddings
        self.use_file = use_file # return file, label
        self.add_file = add_file # return file, label, path

        self.label_dict = {} # {label: idx}
        for idx, path in enumerate(utterances.datalst):
            lab = self.speaker[path.split('/')[speaker_loc]]
            if not (lab in self.label_dict):
                self.label_dict[lab] = []
            self.label_dict[lab].append(idx)
            self.datalst.append(path)
            self.datalab.append(lab)

    @property
    def files(self):
        return self.datalst

    def __len__(self):
        return len(self.datalst)

    def __getitem__(self, indices):
        if not isinstance(indices, list):
            assert isinstance(indices, int)
            indices = [indices]
        path = [self.datalst[idx] for idx in indices] if len(indices) > 1 else self.datalst[indices[0]]
        lab  = self.datalab[indices[0]]
        if self.use_file is True:
            return path, lab
        inp = []
        for index in indices:
            audio = self.read(self.datalst[index])
            inp.append(audio)
        inp = np.concatenate(inp, axis=0)
        inp = torch.FloatTensor(inp)
        if not self.add_file:
            return inp, lab
        return inp, lab, path
    
class SingleUtterance(Dataset):
    """
    Format of data source format: speaker, wav_path. It works for 
        classification objective learning.

        Example
        -------
            id00012 id00012/21Uxsk56VDQ/00001.wav
    """

    def __init__(self, datalst, speaker: dict, read_func):
        super(SingleUtterance, self).__init__()
        self.datalst = datalst
        self.speaker = speaker
        self.read    = read_func
        # ".wav" inputs a waveform
        self.ndim = 1 if ".wav" in datalst[0] else 2

    def __len__(self):
        return len(self.datalst)

    def __getitem__(self, idx):
        spk, path = self.datalst[idx]
        label     = self.speaker[spk]
        inp       = self.read(path)
        inp       = torch.FloatTensor(inp)
        return {"label": label, "input": inp}

class MultiUtterance(Dataset):
    """
    Format of data source format: speaker, wav_path. It works for 
        pairwise learning or E2E learning.

        Example
        -------
            id00012 id00012/21Uxsk56VDQ/00001.wav
    """

    def __init__(self, datalst: np.ndarray, nUtts: int, speaker: dict, read_func=None):
        super(MultiUtterance, self).__init__()
        self.datalst = datalst
        self.nUtts   = nUtts
        self.speaker = speaker
        self.read    = read_func
        # ".wav" inputs a waveform
        self.ndim = 1 if ".wav" in datalst[0] else 2

    def __len__(self):
        return len(self.datalst)

    def __getitem__(self, idx):
        label = [self.speaker[self.datalst[i][0]] for i in idx]
        assert len(set(label)) == 1
        assert len(label) == self.nUtts
        label  = label[0]
        inputs = []
        for i in idx:
            _, path = self.datalst[i]
            inp     = self.read(path)
            inp     = torch.FloatTensor(inp)
            inputs.append(inp)
        return {"label": label, "input": inputs}

    def collate_fn(self, batch):
        """
        Collect from a batch of MultiUtterance Dataset.

            Parameter
            ---------
                batch : a list of dict {label : int, 
                                        input : a list of 1/2-dim tensor}
        """
        labels = torch.LongTensor([item["label"] for item in batch])
        inputs = [torch.stack([item["input"][i] for item in batch], dim=0) for i in range(self.nUtts)]
        return {"label": labels, "input": inputs}

class VerificationTrials(Dataset):
    """
    Verification Trial format: label enroll test

        Example
        -------
            1 voxceleb1/id10136/kDIpLcekk9Q/00011.wav voxceleb1/id10136/kTkrpBqwqJs/00001.wav
    """

    def __init__(self, 
                 trials_lst: str = None, 
                 num_samples: int = 0, 
                 mode_eval: bool = True, 
                 num_eval: int = 1,
                 trials_dir: str = None):
        super(VerificationTrials, self).__init__()
        self.num_samples = num_samples
        self.mode_eval = mode_eval
        self.num_eval = num_eval
        self.load_trial(trials_lst, trials_dir)
        self.utterances = Utterance(self.files, num_samples, mode_eval, num_eval, trials_dir)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label  = self.labels[idx]
        enroll = self.enrolls[idx]
        test   = self.tests[idx]
        return {"label": label, "enroll": enroll, "test": test}

    def trials2utts(self):
        """
        Unique utterances from trials.
        
            Parameter
            ---------
                trials: VerificationTrials {[label], enroll, test}
                    
            Return
            ------
                utterances: list
        """
        return self.files

    def load_trial(self, trials_lst, trials_dir):
        """
        Load trials from list file appended prefix trials_dir.
        Format: label enroll test
            seperated by spaces,
            label is 0 or 1, 
            enroll is a path of a enrollment utterance, 
            test is a path of a test utterance.
        """
        self.labels  = []
        self.enrolls = []
        self.tests   = []

        with open(trials_lst, 'r') as trialsfile:
            while True:
                line = trialsfile.readline()
                if not line:
                    break

                data = line.split(' ')
                data[-1] = data[-1].replace('\n', '')

                ## Append random label if missing
                if len(data) == 2: data = [random.randint(0,1)] + data

                self.labels.append(int(data[0]))
                self.enrolls.append(os.path.join(trials_dir, data[1]))
                self.tests.append(os.path.join(trials_dir, data[2]))

        self.files = list(set(self.enrolls + self.tests))
        self.files.sort()

class UniqueSampler(Sampler):
    """Indices Resampler for dataset that ensure there is no repeated
        speakers in a batch. It supports multiple utterances per speaker.
    """

    def __init__(self,
                 label_dict: dict,
                 batch_size: int = 128,
                 max_utt_per_spk: int = 100,
                 n_utt_per_sample: int = 1):
        """
        Sampler ensure that each batch has no repeated speaker's utterances.

            Parameter
            ---------
                label_dict : dict
                    A *.wav list of speakers, e.g., {label: datalst}.
                batch_size : int
                    The size of sample batch in the dataloader.
                max_utt_per_spk : int
                    The maximum number of utterances in each speaker.
                n_utt_per_sample : int
                    The number of utterances in a sample.

            Return
            ------
                mixmap : list or list[list[,]]
                    indices of dataset
        """
        self.label_dict       = label_dict
        self.batch_size       = batch_size
        self.max_utt_per_spk  = max_utt_per_spk
        self.n_utt_per_sample = n_utt_per_sample

    def _list2multi(self, lst, sz):
        """[1, 2, 3, 4, 5] -> [[1,2], [3,4]]"""
        return [lst[i:i+sz] for i in range(0, len(lst), sz)]

    def unique_sample(self):
        flattened_list  = []
        flattened_label = []
        # Data for each class
        for label, indices in self.label_dict.items():
            numUtt = min(len(indices), self.max_utt_per_spk)
            numUtt = numUtt - (numUtt % self.n_utt_per_sample)
            rp     = self._list2multi(np.random.permutation(indices)[:numUtt].tolist(), self.n_utt_per_sample)
            # it may be further accelarated.
            flattened_list.extend(rp)
            flattened_label.extend([label] * len(rp))
        # Data in random order
        mixid    = np.random.permutation(len(flattened_list))
        mixlabel = []
        mixmap   = []
        # Prevent two pairs of the same speaker in the same batch
        for i in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            idx        = flattened_list[i]
            label      = flattened_label[i]
            if label not in mixlabel[startbatch:]:
                mixmap.append(idx)
                mixlabel.append(label)
        # Iteration size
        self.num_file = len(mixmap)
        self.files    = mixmap
        if self.n_utt_per_sample == 1:
            mixmap    = [ii[0] for ii in mixmap]
        return mixmap

    def __iter__(self):
        return iter(self.unique_sample())

    def __len__(self):
        return self.num_file

class DistributedUniqueSampler(UniqueSampler):
    """
    It works for distributed sampler and is based on UniqueSampler. It is modified from:
        https://pytorch.org/docs/1.6.0/_modules/torch/utils/data/distributed.html#DistributedSampler
        
        Note
        ----
            .set_epoch(epoch) is performed before each training epoch.
    """

    def __init__(self, 
                 label_dict: dict,
                 batch_size: int = 128,
                 max_utt_per_spk: int = 100,
                 n_utt_per_sample: int = 1,
                 num_replicas=None, 
                 rank=None,
                 seed=666):
        
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            
        super(DistributedUniqueSampler, self).__init__(label_dict, batch_size, max_utt_per_spk, n_utt_per_sample)
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        
    def unique_sample(self, g: torch._C.Generator):
        flattened_list  = []
        flattened_label = []
        # Data for each class
        for label, indices in self.label_dict.items():
            numUtt = min(len(indices), self.max_utt_per_spk)
            numUtt = numUtt - (numUtt % self.n_utt_per_sample)
            
            selected = torch.randperm(len(indices), generator=g)[:numUtt].tolist()
            indices_list = np.array(indices)[selected].tolist()
            rp = self._list2multi(indices_list, self.n_utt_per_sample)
            
            flattened_list.extend(rp)
            flattened_label.extend([label] * len(rp))
        # Data in random order
        mixid    = torch.randperm(len(flattened_list), generator=g).tolist()
        mixlabel = []
        mixmap   = []
        # Prevent two pairs of the same speaker in the same batch
        for i in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            idx        = flattened_list[i]
            label      = flattened_label[i]
            if label not in mixlabel[startbatch:]:
                mixmap.append(idx)
                mixlabel.append(label)
        if self.n_utt_per_sample == 1:
            mixmap    = [ii[0] for ii in mixmap]
        return mixmap

    def __iter__(self):
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = self.unique_sample(g)
        
        # remove extra samples to make it evenly divisible
        total_size    = len(indices) - len(indices) % (self.batch_size * self.num_replicas)
        indices       = indices[:total_size]
        self.num_file = len(indices)
        self.files    = indices

        # subsample
        sub_indices = []
        for i in range(self.rank, self.num_file // self.batch_size, self.num_replicas):
            sub_indices += indices[i * self.batch_size: (i + 1) * self.batch_size]
        self.num_samples = len(sub_indices)

        return iter(sub_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class AugmentedUtterance(Dataset):
    """
    Utterances augmented by noises and rir corpus. It works with AugmentWAV methods.
        It is modified from: https://github.com/clovaai/voxceleb_trainer.
        It is as LabeledUtterance that has 3 mode.
            1. path, label
            2. input, label
            3. input, label, path
    """
    def __init__(self, 
                 utterances: Utterance, 
                 speaker, 
                 speaker_loc = -3, 
                 use_file = False,
                 add_file = False,
                 augment  = None,
                 mode = 'v1',
                 seed = 666):

        self.augment_wav = augment
        self.datalst     = []
        self.datalab     = []
        self.speaker     = speaker
        self.num_samples = utterances.num_samples
        self.mode_eval   = utterances.mode_eval
        self.num_eval    = utterances.num_eval
        self.augment     = True if augment is not None else False
        self.ndim        = 1 if ".wav" in utterances.datalst[0] else 2
        self.utterances  = utterances # reserverd Utterance for computing embeddings
        self.use_file    = use_file # return file, label
        self.add_file    = add_file # return input, label, file

        assert mode in ['v1', 'v2', 'v2+']
        if mode == 'v1':
            self.augfunc = [None, 'reverberate', 'music', 'speech', 'noise']
        elif mode == 'v2':
            self.augfunc = [None, 'reverberate', 'speech', 'noise', 'speedup', 'slowdown', 'compand']
        else:
            self.augfunc = [None, 'reverberate', 'music', 'speech', 'noise', 'speedup', 'slowdown', 'compand']
        
        self.seed        = seed if augment is None else augment.seed
        self.random      = random.Random(self.seed)

        self.label_dict = {} # {label: idx}
        for idx, path in enumerate(utterances.datalst):
            lab = self.speaker[path.split('/')[speaker_loc]]
            if not (lab in self.label_dict):
                self.label_dict[lab] = []
            self.label_dict[lab].append(idx)
            self.datalst.append(path)
            self.datalab.append(lab)

    def __getitem__(self, indices):
        """
            Parameter
            ---------
                indices : list
                    a list of indices with same label.

            Return
            ------
                feat, label : tensor, int
                    input(s) with a label.
        """
        if not isinstance(indices, list):
            assert isinstance(indices, int)
            indices = [indices]
        path = [self.datalst[idx] for idx in indices] if len(indices) > 1 else self.datalst[indices[0]]
        lab  = self.datalab[indices[0]]
        if self.use_file is True:
            return path, lab
        inp = []
        for index in indices:
            audio = loadWAV(self.datalst[index], self.num_samples, self.mode_eval, self.num_eval)
            if self.augment: # choice as sugar.waveforms.AugmentWAV
                augtype = self.random.choice(self.augfunc)
                if augtype == None:
                    pass
                elif augtype == 'reverberate':
                    audio  = self.augment_wav.reverberate(audio)
                elif augtype == 'music':
                    audio  = self.augment_wav.add_noise('music', audio)
                elif augtype == 'speech':
                    audio  = self.augment_wav.add_noise('speech', audio)
                elif augtype == 'noise':
                    audio  = self.augment_wav.add_noise('noise', audio)
                elif augtype == 'speedup':
                    audio  = self.augment_wav.speed_up(audio)
                elif augtype == 'slowdown':
                    audio  = self.augment_wav.slow_down(audio)
                elif augtype == 'compand':
                    audio  = self.augment_wav.compand(audio)
            inp.append(audio)
        inp = np.concatenate(inp, axis=0)
        inp  = torch.FloatTensor(inp)
        if not self.add_file:
            return inp, lab
        return inp, lab, path

    def __len__(self):
        return len(self.datalst)


if __name__ == '__main__':
    import sys
    import time
    sys.path.append('/workspace/rwang')
    musan = '/workspace/datasets/musan_split'
    rir   = '/workspace/datasets/Room_Noise/RIRS_NOISES/simulated_rirs'
    print(musan, rir)
    from sugar.waveforms import AugmentWAV, loadWAV
    from sugar.data.voxceleb1 import idenset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    aug = AugmentWAV(musan, rir, 48000)
    train, val, test, spks = idenset(read_func=loadWAV)
    utts = LabeledUtterance(train, spks)
    aug_utts = AugmentedUtterance(train, spks, augment=aug, mode='v1')
    sampler = UniqueSampler(aug_utts.label_dict, n_utt_per_sample=1, max_utt_per_spk=100)
    dataloder = DataLoader(utts, batch_size=128, shuffle=False, sampler=None, num_workers=4)
    for dat in tqdm(dataloder, total=len(aug_utts)//dataloder.batch_size):
        # time.sleep(0.4)
        pass
    print(1)
