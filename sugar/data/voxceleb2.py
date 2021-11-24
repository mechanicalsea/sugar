'''
Descripttion: 
Version: 1.0.0
Author: Rui Wang
Date: 2021-01-29 08:04:22
LastEditTime: 2021-01-29 09:48:32
'''
"""
Description: Dataset from VoxCeleb2
Author: Rui Wang
Date: 2021.01.21

The orginal format of VoxCeleb2 is `acc`, as tool in https://github.com/clovaai/voxceleb_trainer/blob/3bfd557fab5a3e6cd59d717f5029b3a20d22a281/dataprep.py#L118, they are converted to `wav`.

VoxCeleb2:
- From train_list.txt to training data.

File Directory: voxceleb2/{speaker_id}/{utterance_id}/*.wav

Indices List Templete:
- train_list.txt
    ```txt
    id00012 id00012/21Uxsk56VDQ/00001.wav
    id00012 id00012/21Uxsk56VDQ/00002.wav
    id00012 id00012/21Uxsk56VDQ/00003.wav
    id00012 id00012/21Uxsk56VDQ/00004.wav
    id00012 id00012/21Uxsk56VDQ/00005.wav
    ```
"""

import os
from sugar.database import Utterance

__all__ = ['veritrain']

trainlist='/workspace/datasets/voxceleb/Vox2/train_list.txt'
rootdir  = '/workspace/datasets/voxceleb/voxceleb2'

def veritrain(listfile=trainlist, rootdir=rootdir, num_samples=48000):
    """
    Load VoxCeleb2 training set from train_list.txt.

        Return
        ------
            train : Utterance
            spks : dict
    """
    trainlst = []
    spks = []

    with open(listfile, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace('\n', '')
            speaker, wav_path = line.split(' ')
            spks.append(speaker)
            wav_path = os.path.join(rootdir, wav_path)
            trainlst.append(wav_path)
    # create speaker dictionary
    spks = list(set(spks))
    spks.sort()
    spks = {spk: idx for idx, spk in enumerate(spks)}
    print('The number of speakers is', len(spks))
    # print('The "read" method of dataset is', read_func)
    # convert list to Utterance
    train = Utterance(trainlst, num_samples)
    return train, spks