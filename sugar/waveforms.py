"""
Function to waveforms from a given file or np.ndarray.

Waveform | numpy, scipy

Author: Wang Rui
Version: v0.1
Create Date: 2020/9/27
"""
import glob
import os
import random
import sys
import torch
import torchaudio

import numpy as np
from scipy import signal
from scipy.io import wavfile

if any('jupyter' in arg for arg in sys.argv):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

torchaudio.set_audio_backend('sox_io')

def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def load_wav(path, sample_rate=16000):
    """
    Load a whole waveform.

        Return
        ------
            audio : np.ndarray
    """
    sr, audio = wavfile.read(path)
    assert sample_rate == sr
    return audio.astype(np.float32)


def loadWAV(path, num_samples=48000, mode_eval=False, num_eval=6, sample_rate=16000, min_samples=48000):
    """
    Read a wav file from the given path. A forced aligned waveform is 
        extracted by wrap padding (concatenate repeat data) or cropping 
        the original waveform. This operation is inspired from [1].

        [1] Zhang, C., Koishida, K., End-to-End Text-Independent Speaker 
        Verification with Triplet Loss on Short Utterances, Interspeech 
        2017. 1487–1491. https://doi.org/10.21437/Interspeech.2017-1608

        Requirement
        -----------
            scipy
            numpy
            random

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
            wave : 2-dim array
                A waveform with the given samples. If mode_eval is 
                false, the waveform is of (1, ?). If mode_eval is true, 
                the waveform is of 2-dim, as a batch of waveforms.
    """
    assert ~(min_samples is None and num_samples == 0), "Given min is None with 0 samples"
    sr, audio = wavfile.read(path)
    assert sr == sample_rate, "%d != %d" % (sr, sample_rate)

    audiosize = audio.shape[0]

    min_samples = min_samples if min_samples is not None and min_samples >= num_samples else num_samples
    if audiosize <= min_samples:
        shortage  = min_samples - audiosize + 1
        audio     = np.pad(audio, (0, shortage), mode="wrap")
        audiosize = audio.shape[0]

    wave = []
    if mode_eval is False:
        startsample = max(0, np.int64(random.random()*(audiosize-num_samples)))
        wave.append(audio[startsample: startsample + num_samples])
        wave = np.stack(wave, axis=0)
    else:
        if num_samples != 0:
            startsample = np.linspace(0, audiosize-num_samples, num=num_eval)
            
            for i_start in startsample:
                wave.append(audio[int(i_start): int(i_start)+num_samples])
            wave = np.stack(wave, axis=0)
        else:
            wave = np.stack([audio], axis=0)
    
    return wave.astype(np.float32)


def wav_segment(audio, num_samples=48000, noverlap=None):
    """Segment a waveform into the specific length.
    
        Parameter
        ---------
            audio : 1-dim/2-dim array
                A waveform.
            num_samples :
                The length of the resulted segments.
        
        Return
        ------
            wave : 2-dim array
                Multiple segments from the input with the specific length.
    """
    audio     = audio[0] if audio.ndim == 2 else audio
    shortage  = num_samples - (audio.shape[0] % num_samples)
    audio     = np.pad(audio, (0, shortage), mode="wrap")
    audiosize = audio.shape[0]

    start_i  = 0
    noverlap = num_samples if noverlap is None else noverlap
    wave     = []
    while start_i < audiosize:
        wave.append(audio[start_i: start_i + num_samples])
        start_i += noverlap

    wave = np.stack(wave, axis=0)
    return wave


def vad(wave, fs=16000, win_len=400, hop_len=160, n_fft=512,
        energy_threshold=0.12, low_freq=300, high_freq=3000):
    """
    Use signal energy to detect voice activity in numpy ndarray.
        Detects speech regions based on ratio between speech band energy 
        and total energy. Outputs are two arrat with the number of 
        frames where the first output is start frame and the second 
        output is to indicate voice activity.

        Parameter
        ---------
            wave : 1-dim array

        Return
        ------
            voiced_wave : 1-dim array
                a voiced waveform.
            voiced_sample : 1-dim array
                it indicate whether a sample is speech or not.
            energy : 2-dim array
                energy spectrogram of a total waveform.
    """

    f, t, Sxx = signal.stft((wave * 1.0).astype(np.float32), fs=fs,
                            window="hann", nfft=n_fft,
                            nperseg=win_len, noverlap=win_len-hop_len, 
                            padded=False, boundary=None, return_onesided=True)
    energy    = np.abs(Sxx)[1:, :]
    f         = f[1:]

    voiced_energy = np.sum(energy[(f>=low_freq) & (f<=high_freq), :], axis=0)
    full_energy   = np.sum(energy, axis=0)
    full_energy[full_energy == 0] = 2.220446049250313e-16

    voiced        = (voiced_energy / full_energy) >= energy_threshold
    voiced_sample = np.arange(0, voiced.shape[0]*hop_len).reshape(-1, hop_len)[voiced, :].flatten()

    if voiced[-1] is True and voiced.shape[0] * hop_len < wave.shape[0]:
        last_sample = np.arange(voiced.shape[0]*hop_len, wave.shape[0])
        voiced_sample = np.concatenate((voiced_sample, last_sample))
    
    voiced_wave = wave[voiced_sample]
    return voiced_wave, voiced_sample, energy


def utt2dur(utts):
    """
    Compute duration of utterances.
    
    Parameter
    ---------
        utterances : array or list
            a list of utterances of files
    
    Return
    ------
        durations : array
    """
    durs = np.empty(len(utts))
    task = "Utterance Durations"
    for i, wav in tqdm(enumerate(utts), total=len(utts), desc=task):
        sr, audio = wavfile.read(wav)
        durs[i]   = audio.shape[0] / sr
    return durs


class AugmentWAV(object):
    """
    Augment waveform via speech signal processing.
        Since it requires augmented corpus, this tool is implemented 
        as a class with multiple methods.
        This corpus include MUSAN and RIRs which has been split for
        accelerating reading.

        Method
        ------
            add_noise
            reverberate

        Requirement
        -----------
            scipy
            random

        It is modified from: https://github.com/clovaai/voxceleb_trainer
    """

    MAX_INT16 = 1 << 15

    def __init__(self, musan_path, rir_path, num_samples=32000, seed=None):
        """
        Class for waveform augmentation.

            Parameter
            ---------
                musan_path: str
                    It is generally MUSAN corpus, which also can be other dataset addititional.
                        The path of *.wav is as `music/fma/music-fma-0015/00000.wav`. It is 
                        create by spliting the original MUSAN as https://github.com/clovaai/voxceleb_trainer.
                        The parent path is `./musan_split` or `./musan`.
                rir_path: str
                    It is generally simulated RIR (Room Impluse Response) corpus, which also can
                        be other dataset convolved. The *.wav is as `smallroom/Room181/Room181-00001.wav`.
                        It refers to medium and small room as https://github.com/clovaai/voxceleb_trainer.
                        The parent path is `./Room_Noise/RIRS_NOISES/simulated_rirs`.
                seed: int
                    mannual seed of random.Random. Force the random process to be definite in order to take
                        advantage of repeatedly loading augmented utterances. 
        """
        self.num_samples = num_samples # It is the fixed length of inputs.
        # Paramter of noises
        self.noiselist  = None  # additional signals
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1]}
        # Parameters of RIR
        self.rir_files  = None  # convolved signals
        # Parameter of tempo
        self.ratespeed = {'speedup': [1.1, 1.2], 'slowdown': [0.8, 0.9]}
        # Parameters of compand: empty

        self.seed       = seed
        self.random     = random.Random(self.seed)

        pathlist = []
        # load file list of additional noises
        if musan_path is not None:
            augment_files  = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
            pathlist.extend(augment_files)
            self.noiselist = {}
            for file in augment_files:
                if not file.split('/')[-4] in self.noiselist:
                    self.noiselist[file.split('/')[-4]] = []
                self.noiselist[file.split('/')[-4]].append(file)

        # load file list of convolved responses
        if rir_path is not None:
            # RIR of medium and small room are used
            self.rir_files  = glob.glob(os.path.join(rir_path, 'mediumroom/*/*.wav')) + glob.glob(os.path.join(rir_path, 'smallroom/*/*.wav'))
            pathlist.extend(self.rir_files)

    def add_noise(self, noisecat, audio):
        """
        Add one or more noises to a speech signals.

            Parameter
            ---------
                noisecat: str
                    The type of additional noises, including {'noise', 'speech', 'music'}.
                audio: np.ndarray
                    A speech signal to be added noises.

            Return
            ------
                audio: np.ndarray
                    A speech signal added noises.
        """
        clean_db  = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        numnoise  = self.numnoise[noisecat]
        noiselist = self.random.sample(self.noiselist[noisecat], self.random.randint(numnoise[0],numnoise[1]))
        noises = np.empty((len(noiselist), self.num_samples), dtype=np.float32)
        for i, noise in enumerate(noiselist):
            noises[i,:] = loadWAV(noise, self.num_samples)[0] # load a noise
        noise_db = 10 * np.log10(np.mean(noises ** 2, axis=1, keepdims=True) + 1e-4)
        noise_snr = np.array([[self.random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])] for i in range(len(noiselist))])
        magnify_db = np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)).astype(np.float32)
        noiseaudio = np.mean(magnify_db * noises, axis=0, keepdims=True)
        return audio + noiseaudio

    def reverberate(self, audio):
        """
        Reverberate a speech signal via a simulated RIR chose randomly.

            Parameter
            ---------
                audio: np.ndarray
                    A speech signal to be reverberated.

            Return
            ------
                audio: np.ndarray
                    A reverberated speech signal.
        """
        rir_file = self.random.choice(self.rir_files)
        _, rir = wavfile.read(rir_file)
        rir    = np.expand_dims(rir.astype(np.float32), 0)
        energy = np.sqrt(np.sum(rir**2)).astype(np.float32)
        rir    = rir / energy # magnify as the energy of the simulated RIR signal
        return signal.fftconvolve(audio, rir, mode='full')[:, :self.num_samples]

    @torch.no_grad()
    def speed_up(self, audio, sr=16000):
        """
        Speed up audio using tempo in torchaudio sox effects.
            More details in Sox: 
                tempo [−q] [−m|−s|−l] factor [segment [search [overlap]]]
            Link: http://sox.sourceforge.net/sox.html

            Parameter
            ---------
                audio: np.ndarray
                    A speech signal to be speed/tempo up.
                sr: int
                    Sampling rate.

            Return
            ------
                audio: np.ndarray
                    A speed/tempo up speech signal.
        """
        speedup = self.random.choice(self.ratespeed['speedup'])
        effects = [['tempo', str(speedup)]]
        audio = torch.FloatTensor(audio / self.MAX_INT16)
        audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, sr, effects)
        audio = audio * self.MAX_INT16
        new_size = audio.shape[1]
        audio = np.pad(audio.numpy(), ((0, 0), (0, self.num_samples - new_size)), mode='wrap')
        return audio

    @torch.no_grad()
    def slow_down(self, audio, sr=16000):
        """
        Slow down audio using tempo in torchaudio sox effects.  
            More details in Sox: 
                tempo [−q] [−m|−s|−l] factor [segment [search [overlap]]]
            Link: http://sox.sourceforge.net/sox.html

            Parameter
            ---------
                audio: np.ndarray
                    A speech signal to be slow/tempo down.
                sr: int
                    Sampling rate.

            Return
            ------
                audio: np.ndarray
                    A slow/tempo down speech signal.
        """
        slowdown = self.random.choice(self.ratespeed['slowdown'])
        effects = [['tempo', str(slowdown)]]
        audio = torch.FloatTensor(audio / self.MAX_INT16)
        audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, sr, effects)
        audio = audio * self.MAX_INT16
        return audio.numpy()[:,:self.num_samples]

    @torch.no_grad()
    def compand(self, audio, sr=16000):
        """
        Compress and expand the waveform using compand in torchaudio sox effects.
            More details in Sox:
                compand attack1,decay1{,attack2,decay2} [soft-knee-dB:]in-dB1[,out-dB1]{,in-dB2,out-dB2} [gain [initial-volume-dB [delay]]]
            Link: http://sox.sourceforge.net/sox.html
        """
        effects = [["compand", "0,0.5", "6:-6,-3"]]
        audio = torch.FloatTensor(audio) / self.MAX_INT16
        audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, sr, effects)
        audio *= self.MAX_INT16
        return audio.numpy()


if __name__ == '__main__':

    # audio = np.random.rand(1, 32000)
    # rir = np.random.rand(1, 4800)
    # ans = [signal.fftconvolve(audio, rir, mode='same') for _ in tqdm(range(2000))]
    # import torch
    # audio = torch.FloatTensor(audio).view(1, 1, audio.shape[-1]).to('cuda:3')
    # rir = torch.FloatTensor(rir).view(1, 1, rir.shape[-1]).to('cuda:3')
    # print(torch.nn.functional.conv1d(audio, rir).shape)
    # ans = [torch.nn.functional.conv1d(audio, rir)[0] for _ in tqdm(range(3000))]
    # quit()
    def reverberate(self, audio):
        rir_file = self.random.choice(self.rir_files)
        _, rir = wavfile.read(rir_file)
        rir    = np.expand_dims(rir.astype(np.float32), 0)
        energy = np.sqrt(np.sum(rir**2)).astype(np.float32)
        rir    = rir / energy # magnify as the energy of the simulated RIR signal
        return signal.fftconvolve(audio, rir, mode='same') # fftconvolve(audio, rir, mode='same')

    musan_path='/workspace/datasets/MUSAN/musan_split'
    rir_path='/workspace/datasets/Room_Noise/RIRS_NOISES/simulated_rirs'
    aug = AugmentWAV(musan_path, rir_path, 32000)

    rng = ((np.random.rand(1, 32000) - 0.5) * (1<<15)).astype(np.float32)

    compand = [aug.compand(rng).shape for _ in tqdm(range(1000), desc='Compand')]
    slowdown = [aug.slow_down(rng).shape for _ in tqdm(range(1000), desc='Slowdown')]
    speedup = [aug.speed_up(rng).shape for _ in tqdm(range(1000), desc='Speedup')]
    rever = [aug.reverberate(rng).shape for _ in tqdm(range(2000), desc='Reverberate')] # True 660 it/s | False 450-650 it/s
    noise = [aug.add_noise('noise', rng).shape for _ in tqdm(range(5000), desc='Noisy')] # True 3200 it/s | False 2200-2700 it/s
    music = [aug.add_noise('music', rng).shape for _ in tqdm(range(5000), desc='Music')] # True 2900 it/s | False 750-2600 it/s
    speech = [aug.add_noise('speech', rng).shape for _ in tqdm(range(1000), desc='Babble')] # True 1000 it/s | False 370-950 it/s musan_split helps

    print(1)
