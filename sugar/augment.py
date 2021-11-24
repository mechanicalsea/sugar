"""
Augment speech dataset for speaker recognition via adding noise and 
    reverberation.

Reverberation is typically represented by convolution of the audio 
    signals with a room impulse response (RIR). We can simulate 
    far-filed speech using the following equation:
    $$
    x_r[t] = x[t] \ast h_s[t] + \sum_i{n_i[t] \ast h_i[t]} + d[t]
    $$
    while there is no $n_i[t]$ or $h_s[t]$, we can obtain augmented 
    speech with other speech as:
    $$
    x_a[t] = x[t] + d[t]
    $$

Author: Rui Wang

Function
--------
    1. reverberate monophonic speech using `wav_reverb` method.
    2. augment monophonic speech using `wav_augment` method.

Note
----
    This work is inspired by kaldi's wav-reverberate.cc and paper "A study on
        data augmentation of reverberant speech for robust speech recognition".
"""
import random
import numpy as np
from scipy import signal


__all__ = ["reverberate", "wav_reverb", "wav_augment", 
           "freq_mask", "time_mask"]


###############################
# Waveform-based Augmentation #
###############################

fs = 16000
shift_output = True
normalize = True


def compute_early_reverb_power(wave, rir, fs):
    """Computer the power of the early reverberated wave.
    """
    peak_index = rir.argmax()
    before_peak = int(0.001 * fs)
    after_peak = int(0.05 * fs)
    early_rir_start = min(0, peak_index - before_peak)
    early_rir_end = max(rir.shape[0], peak_index + after_peak)
    early_rir = rir[early_rir_start: early_rir_end]
    early_reverb = signal.fftconvolve(wave, early_rir, mode="full")
    early_power = np.dot(early_reverb, early_reverb) / early_reverb.shape[0]
    return early_power


def add_noise(wave, noise, fs, snr, start_time, duration, wave_power):
    """Add a noise to wave.
    """
    noise_power = np.dot(noise, noise) / noise.shape[0]
    scale_factor = np.sqrt(10**(-snr/10.0) * wave_power / noise_power)
    noise = noise * scale_factor
    offset = int(start_time * fs)
    add_length = min(wave.shape[0] - offset, int(duration * fs), noise.shape[0])
    if add_length > 0:
        wave[offset: offset + add_length] += noise[0: add_length]
    return wave


def do_reverb(wave, rir, fs):
    """
    wave : 1-dim array, int16
    rir  : 1-dim array, int16
    fs   : int
    """
    # normalize RIR
    rir = rir.astype("float32")
    rir = rir / np.max(np.abs(rir))
    # compute the power of early reverberated wave
    early_power = compute_early_reverb_power(wave, rir, fs)
    wave = signal.fftconvolve(wave, rir, mode="full")
    return wave, early_power


def reverberate(wave,
                rir=None,
                fs=fs,
                point_source_noises=None,
                isotropic_noises=None,
                shift_output=shift_output,
                normalize=normalize):
    """Reverberate and add noise to waveform (wave).
        The number of RIR of reverberation is 1 becasue of the static/fixed 
        room. The number of noises to add can be multiple becasuse foreground 
        and backgound noises can occur in a condition.

        Function
        --------
            reverberate : wave, rir, fs[, point_source_noises, isotropic_signals, 
                        shift_output, normalize]
            add_noise   : wave, fs, isotropic_signals[, shift_output, normalize]

        Parameter
        ---------
            wave : array, dim=1, int16
                Input signal, a monophonic waveform to be reverberated and possibly 
                added noise(s). It is in 16k sampling rate and 16-bit precision.
            rir : array, dim=1, int16
                A room impluse response (RIR) is used to reverberate. It is in 16k 
                sampling rate and 16-bit precision.
            fs : int
                Sampling rate of the given waveform and RIR.
            point_source_noises : dict
                point-source noises, e.g., 
                {
                    "signal": ..., # a list of 1-dim int16 array
                    "rir": ..., # a list of 1-dim int16 array
                    "duration": ..., # a list of float
                    "start_time": ..., # a list of float
                    "snr": ... # a list of float
                }
            isotropic_noises : dict
                isotropic noises, e.g., 
                {
                    "signal": ..., # a list of 1-dim int16 array
                    "duration": ..., # a list of float
                    "start_time": ..., # a list of float
                    "snr": ... # a list of float
                }

            normalize: bool
                If true, then after reverberating and possibly adding noise, scale 
                so that the signal energy is the same as the original input signal.
            shift_output : bool
                If true, the reverberated waveform will be shifted by the 
                amount of the peak position of the RIR and the length of 
                the output waveform will be equal to the input waveform. 
                If false, the length of the output waveform will be equal to 
                (original input length + rir length - 1). 

        Note
        ----
            signal : (a list of) 1-dim ndarray, int16
                A list of additive signals or a signal is used to add into the 
                input signal.
            duration : (a list of) float
                If nonzero, it specifies the duration (second) of the output 
                signal. If the duration is less than the length of the input 
                signal, the first duration of the input signal is trimmed, 
                otherwise, the signal will be repeated to fullfill the duration.
            start_time : (a list of) float
                A list of start times or a start time (second) refer to the offset 
                of the input signal.
            snr : (a list of) float
                A list of signal-to-noise (snr) or a snr is used to scale additive 
                signals.

        Return
        ------
            wave : array, dim=1, int16
                A speech signal after reverberation or adding noise(s).
    """
    if rir is None \
            and point_source_noises is None \
            and isotropic_noises is None:
        return wave
    # check size of point-source noises and isotropic noises
    if point_source_noises is not None \
        and len(point_source_noises["signal"]) != 0:
        assert len(point_source_noises["signal"]) == \
            len(point_source_noises["rir"]) == \
            len(point_source_noises["duration"]) == \
            len(point_source_noises["start_time"]) == \
            len(point_source_noises["snr"]), \
            "Unequal size of point-source noises: {}-{}-{}-{}-{}".format(
                len(point_source_noises["signal"]),
                len(point_source_noises["rir"]),
                len(point_source_noises["duration"]),
                len(point_source_noises["start_time"]),
                len(point_source_noises["snr"])
        )
    if isotropic_noises is not None \
        and len(isotropic_noises["signal"]) != 0:
        assert len(isotropic_noises["signal"]) == \
            len(isotropic_noises["duration"]) == \
            len(isotropic_noises["start_time"]) == \
            len(isotropic_noises["snr"]), \
            "Unequal size of isotropic noise(s): {}-{}-{}-{}".format(
                len(isotropic_noises["signal"]),
                len(isotropic_noises["duration"]),
                len(isotropic_noises["start_time"]),
                len(isotropic_noises["snr"])
        )
    wave = wave.astype("float32")
    # duration of output wave
    if shift_output is True:
        dur2len = len(wave)
    elif shift_output is False:
        dur2len = len(wave) + len(rir) - 1
    before_power = np.dot(wave, wave) / wave.shape[0]
    peak_index = 0
    early_power = before_power
    # reveberate wave using a rir
    if rir is not None:
        wave, early_power = do_reverb(wave, rir, fs)
        if shift_output is True:
            peak_index = rir.argmax()
    # add noises if possible
    if isotropic_noises is not None:
        for i in range(len(isotropic_noises["signal"])):
            inoise = isotropic_noises["signal"][i]
            iduration = isotropic_noises["duration"][i]
            istart_time = isotropic_noises["start_time"][i]
            isnr = isotropic_noises["snr"][i]
            wave = add_noise(wave, inoise.astype("float32"), fs,
                             isnr, istart_time, iduration, early_power)
    if point_source_noises is not None:
        for p in range(len(point_source_noises["signal"])):
            pnoise = point_source_noises["signal"][i]
            prir = point_source_noises["rir"][i]
            pduration = point_source_noises["duration"][i]
            pstart_time = point_source_noises["start_time"][i]
            psnr = point_source_noises["snr"][i]
            pwave = reverberate(pnoise, prir, fs=fs,
                                shift_output=shift_output, normalize=normalize)
            wave = add_noise(wave, pwave.astype("float32"), fs,
                             psnr, pstart_time, pduration, early_power)
    # normalize wave
    if normalize is True:
        # compute the power of wave after reverberation and possibly noise
        after_power = np.dot(wave, wave) / wave.shape[0]
        wave = wave * (np.sqrt(before_power / after_power))
    # extend wave if duration is larger than the length of wave
    if dur2len > wave.shape[0]:
        wave = np.pad(wave, pad_width=(
            0, dur2len - wave.shape[0]), mode="wrap")
    # shift the output wave by shift_output
    wave = wave[peak_index: peak_index + dur2len]
    return wave.astype("int16")


class number_cycle(object):
    def __init__(self, num):
        self.index = 0
        self.len = num
        self.num = np.arange(num, dtype="long")
        random.shuffle(self.num)

    def __next__(self):
        item = self.num[self.index]
        self.index = (self.index + 1) % self.len
        return item


def wav_reverb(wave, fs=16000,
               rirdict=None, noisedict=None,
               fg_snrs=None, bg_snrs=None,
               reverb_prob=1.0, pnoise_prob=1.0,
               inoise_prob=1.0, shift_output=True,
               max_pnoise_per_minute=2):
    """Reverberate waveform using RIRs.

        Parameter
        ---------
            wave: 1-dim array
            rirdict : dict 
                e.g., {"room_1", {"probability": 0.5, "rir": [...]}}
            noisedict : dict
                Noises include point-source noises and isotropic noises, where 
                point-source is manually classify as either a foreground or a 
                background noise. For example, 
                    {"point-source": [{"wave": ..., "type": ...}, ...],
                    "isotropic": [...]}
                where "type" contains "background" and "foreground".
        
        Example
        -------
            ```bash
            # Augment with musan_noise
            steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 
                --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
            ```
            import soundfile as sf
            wave, fs = sf.read(wav_file, dtype="int16")
            assert fs == 16000
            wave = wav_augment(wave, fs=16000, fg_utts=noise_utts, fg_snrs=[15,10,5,0], 
                               fg_interval=1, shift_output=True)
                
        Note
        ----
            Random operation is simplied.
    """
    assert rirdict is not None
    if fg_snrs is not None:
        fg_snrs_cycle = number_cycle(len(fg_snrs))
    if bg_snrs is not None:
        bg_snrs_cycle = number_cycle(len(bg_snrs))
    wave_dur = wave.shape[0] / fs
    room_list = list(rirdict.keys())
    room_prob = [rirdict[room_i]["probability"] for room_i in room_list]
    room_index = random.choices(room_list, weights=room_prob)[0]
    if random.random() < reverb_prob:
        wave_rir = random.choice(rirdict[room_index]["rir"])
    point_source_noises = {
        "signal": [],
        "rir": [],
        "duration": [],
        "start_time": [],
        "snr": []
    }
    if noisedict is not None and "point-source" in noisedict and random.random() < pnoise_prob:
        max_pnoise_recording = int(max_pnoise_per_minute * wave_dur / 60)
        if max_pnoise_recording >= 1:
            for k in range(random.randint(1, max_pnoise_recording)):
                pnoise = random.choice(noisedict["point-source"])
                point_source_noises["signal"].append(pnoise["wave"])
                prir = random.choice(rirdict[room_index]["rir"])
                point_source_noises["rir"].append(prir)
                if pnoise["type"] == "background":
                    duration = wave_dur
                    start_time = 0
                    snr = bg_snrs[next(bg_snrs_cycle)]
                elif pnoise["type"] == "foreground":
                    duration = pnoise["wave"].shape[0] / fs
                    start_time = round(random.random() * wave_dur, 2)
                    snr = fg_snrs[next(fg_snrs_cycle)]
                point_source_noises["duration"].append(duration)
                point_source_noises["start_time"].append(start_time)
                point_source_noises["snr"].append(snr)
    isotropic_noise = {
        "signal": [],
        "duration": [],
        "start_time": [],
        "snr": []
    }
    if noisedict is not None and "isotropic" in noisedict and random.random() < inoise_prob:
        if len(noisedict["isotropic"]) > 0:
            inoise = random.choice(noisedict["isotropic"])
            isnr = bg_snrs[next(bg_snrs_cycle)]
            isotropic_noise["signal"].append(inoise)
            isotropic_noise["duration"].append(wave_dur)
            isotropic_noise["start_time"].append(0)
            isotropic_noise["snr"].append(isnr)

    return reverberate(wave, wave_rir, fs=fs,
                       point_source_noises=point_source_noises,
                       isotropic_noises=isotropic_noise,
                       shift_output=shift_output, normalize=True)


def wav_augment(wave, fs=16000,
                fg_utts=None, bg_utts=None,
                fg_snrs=None, bg_snrs=None,
                fg_interval=1, bg_nums=[3, ],
                shift_output=True):
    """Augment speech with foreground or background noises.

        Parameter
        ---------
            fg_utts : a list of array
            bg_utts : a list of array
            fg_snrs : a list of float
            bg_snrs : a list of float
            fg_interval : float
            bg_nums : a list of int

        Example
        -------
            ```bash
            # Augment with musan_music
            steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" 
                --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
            ```
            import soundfile as sf
            wave, fs = sf.read(wav_file, dtype="int16")
            assert fs == 16000
            wave = wav_augment(wave, fs=16000, bg_utts=music_utts, bg_snrs=[15,10,8,5], 
                               bg_nums=[1], shift_output=True)
    """
    if fg_utts is None and bg_utts is None:
        return wave
    wave_dur = wave.shape[0] / fs
    additional_noises = {
        "signal": [],
        "duration": [],
        "start_time": [],
        "snr": []
    }
    if bg_utts is not None and len(bg_utts) > 0:
        num = random.choice(bg_nums)
        for i in range(num):
            noise = random.choice(bg_utts)
            snr = random.choice(bg_snrs)
            additional_noises["signal"].append(noise)
            additional_noises["duration"].append(wave_dur)
            additional_noises["start_time"].append(0)
            additional_noises["snr"].append(snr)
    if fg_utts is not None and len(fg_utts) > 0:
        total_noise_dur = 0
        while total_noise_dur < wave_dur:
            noise = random.choice(fg_utts)
            noise_dur = noise.shape[0] / fs
            start_time = total_noise_dur
            snr = random.choice(fg_snrs)
            additional_noises["signal"].append(noise)
            additional_noises["duration"].append(noise_dur)
            additional_noises["start_time"].append(start_time)
            additional_noises["snr"].append(snr)
            total_noise_dur += noise_dur + fg_interval
    return reverberate(wave, fs=fs,
                       isotropic_noises=additional_noises,
                       shift_output=shift_output,
                       normalize=True)



##################################
# Spectrogram-based Augmentation #
##################################


# features
def freq_mask(spec, F=10, num_masks=2, replace_with_zero=False):
    """Frequency Mask is intuitively described as mask certain frequency bands with 
        either the mean value of the spectrogram or zero.
    
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

# features
def time_mask(spec, T=40, num_masks=2, replace_with_zero=False):
    """Time Mask is intuitively described as mask certain time ranges with the mean 
        value of the spectrogram or zero.
        
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



if __name__ == "__main__":
    import soundfile as sf
    import sounddevice as sd
    wave, fs_w = sf.read("audio/arctic_a0010.wav", dtype="int16")
    rir, fs_r = sf.read("audio/Room001-00001.wav", dtype="int16")
    noise, fs_n = sf.read("audio/exercise_bike.wav", dtype="int16")
    assert fs_w == fs_r == fs_n
    fs = fs_w
    sd.play(wave, fs)
    sd.wait()
    wave1 = wav_reverb(wave, fs,
                      rirdict={"room_1": {"probability": 1.0, "rir": [rir]}},
                      noisedict={"point-source":
                                 [{"wave": noise, "type": "background"},
                                  {"wave": noise, "type": "foreground"}],
                                 "isotropic": [noise]},
                      fg_snrs=[20,], bg_snrs=[20,],
                      reverb_prob=1.0, pnoise_prob=0.0, inoise_prob=0.0,
                      max_pnoise_per_minute=20)

    print(wave1)
    sd.play(wave1, fs)
    sd.wait()
    sf.write("audio/reverb_arctic_a0010.wav", wave1, samplerate=fs)
    wave2 = wav_augment(wave, fs=16000,
                       fg_utts=[noise] * 5, bg_utts=[noise]*10,
                       fg_snrs=[10,], bg_snrs=[15,],
                       fg_interval=1, bg_nums=[1],
                       shift_output=True)
    print(wave2)
    sd.play(wave2, fs)
    sd.wait()
    sf.write("audio/noisy_arctic_a0010.wav", wave2, samplerate=fs)
