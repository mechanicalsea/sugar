'''
Descripttion: 
Version: 1.0.0
Author: Rui Wang
Date: 2021-01-29 08:04:22
LastEditTime: 2021-01-29 09:44:22
'''
from sugar.waveforms import AugmentWAV

musan_path = '/workspace/datasets/MUSAN/musan_split'
rir_path = '/workspace/datasets/Room_Noise/RIRS_NOISES/simulated_rirs'

def augset(musan_path=musan_path, rir_path=rir_path, num_samples=48000, seed=666):
    """
    Load corpus for speech augmentation, such as, MUSAN and simulated rirs.
    """
    aug_wav  = AugmentWAV(musan_path, rir_path, num_samples, seed)
    return aug_wav