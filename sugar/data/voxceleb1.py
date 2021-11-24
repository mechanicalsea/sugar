"""
Description: Dataset from VoxCeleb1
Author: Rui Wang
Date: 2020.11.26

VoxCeleb1: 
- From iden_split.txt to the training and validation set.
- From iden_split.txt to evaulate the identification task.
- From veri_test2.txt, list_test_all2.txt, and list_test_hard2.txt to 
    evaluate the corresponding verification task.

File Directory: voxceleb1/{speaker_id}/{utterance_id}/*.wav

Indices List: 
    - Speaker Identification: iden_split.txt
    - Speaker Verification: veri_test2.txt, list_test_hard2.txt, list_test_all2.txt

List Template:
    - iden_split.txt
        | 3    | id10003/na8-QEFmj44/00003.wav |
        | 1    | id10003/tCq2LcKO6xY/00002.wav |
        | 1    | id10003/K5zRxtXc27s/00001.wav |
    - veri_test2.txt, list_test_hard2.txt, list_test_all2.txt
        | 1    | id10001/Y8hIVOBuels/00001.wav | id10001/1zcIwhmdeo4/00001.wav |
        | 0    | id10001/Y8hIVOBuels/00001.wav | id10943/vNCVj7yLWPU/00005.wav |
        | 1    | id10001/Y8hIVOBuels/00001.wav | id10001/7w0IBEWc9Qw/00004.wav |
"""
import os
from sugar.database import Utterance
from sugar.database import VerificationTrials

__all__ = ['idenset', 'veriset', 'veritrain']

idenlist = '/workspace/datasets/voxceleb/Vox1/iden_split.txt'
rootdir  = '/workspace/datasets/voxceleb/voxceleb1'

verilist = '/workspace/datasets/voxceleb/Vox1/veri_split.txt' # create as the official mentioned

veritest2 = '/workspace/datasets/voxceleb/Vox2/veri_test2.txt'
veriall2 = '/workspace/datasets/voxceleb/Vox2/list_test_all2.txt'
verihard2 = '/workspace/datasets/voxceleb/Vox2/list_test_hard2.txt'

def idenset(listfile=idenlist, rootdir=rootdir, read_func=None, num_samples=48000, num_eval=2, is_xvector=True):
    """
    Load VoxCeleb1 training set, and identification test set.
        iden_split.txt to the training, validaton, and test sets.

        Return
        ------
            train : Utterance
            val : Utterance
            test : Utterance
            spks : dict
            is_xvector : bool
                'x-vector' or 'd-vector'
    """
    TRAIN_TYPE = 1
    VAL_TYPE = 2
    TEST_TYPE = 3
    datalst = []
    spks = []
    with open(listfile, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace('\n', '')
            split_type, wav_path = line.split(' ')
            spks.append(wav_path.split('/')[0])
            split_type = int(split_type)
            wav_path = os.path.join(rootdir, wav_path)
            datalst.append([split_type, wav_path])
    # create speaker dictionary
    spks = list(set(spks))
    spks.sort()
    spks = {spk: idx for idx, spk in enumerate(spks)}
    print('The "read" method of dataset is', read_func)
    # split train part
    trainlst = [wav for t, wav in datalst if t == TRAIN_TYPE]
    # split validation part
    vallst = [wav for t, wav in datalst if t == VAL_TYPE]
    # split test part
    testlst = [wav for t, wav in datalst if t == TEST_TYPE]
    # convert list to Utterance
    train = Utterance(trainlst, num_samples)
    val = Utterance(vallst, num_samples)
    test = Utterance(testlst, num_samples if not is_xvector else 0, True, num_eval)
    return train, val, test, spks


def veriset(test2=veritest2, all2=veriall2, hard2=verihard2, rootdir=rootdir, num_samples=48000, num_eval=2):
    """
    Load VoxCeleb1 verification test set.
        1. veri_test2.txt to test2, 
        2. list_test_hard2.txt to hard2,
        3. list_test_all2.txt to all2.
        These list file has the same root directory.

        Return
        ------
            veri_test2 : VerificationTrials
            veri_all2 : VerificationTrials
            veri_hard2 : VerificationTrials
            wav_files : Utterance

        Notes
        -----
            Utterance and VerificationTrials are both Dataset.
            VerificationTrials: 
                {'label': int, 'enroll': str, 'test': str}
            Utterance:
                {'file': str, 'input': Tensor}
            By try-and-error, 2 x 5-second segments is better within 20 seconds. 
    """
    # load multiple trials list
    veri_test2 = VerificationTrials(test2, num_samples, True, num_eval, rootdir)
    veri_hard2 = VerificationTrials(hard2, num_samples, True, num_eval, rootdir) if hard2 is not None and os.path.exists(hard2) else None
    veri_all2 = VerificationTrials(all2, num_samples, True, num_eval, rootdir) if all2 is not None and os.path.exists(all2) else None 
    # create unique file list, and then convert to be a Utterance
    wav_files = set()
    for veri in [veri_test2, veri_hard2, veri_all2]:
        if veri is not None:
            wav_files |= set(veri.files)
    wav_files = list(wav_files)
    wav_files = Utterance(wav_files, num_samples, True, num_eval)
    return veri_test2, veri_all2, veri_hard2, wav_files 


def veritrain(listfile=verilist, rootdir=rootdir, num_samples=48000):
    train, _, _, spks = idenset(listfile, rootdir, num_samples)
    return train, spks

if __name__ == '__main__':
    veri_test2, veri_all2, veri_hard2, wav_files = veriset()
    
    print('# of utterances in Vox1-O: {:,}'.format(len(veri_test2.files)))
    print('# of utterances in Vox1-E: {:,}'.format(len(veri_all2.files)))
    print('# of utterances in Vox1-H: {:,}'.format(len(veri_hard2.files)))
    print('# of utterances in Vox1-E/H: {:,}'.format(len(set(veri_all2.files) | set(veri_hard2.files))))

    print('# of speakers in Vox1-O: {:,}'.format(len(set([utt.split('/')[-3] for utt in veri_test2.files]))))
    print('# of speakers in Vox1-E: {:,}'.format(len(set([utt.split('/')[-3] for utt in veri_all2.files]))))
    print('# of speakers in Vox1-H: {:,}'.format(len(set([utt.split('/')[-3] for utt in veri_hard2.files]))))
    print('# of speakers in Vox1-E/H: {:,}'.format(len(set([utt.split('/')[-3] for utt in veri_all2.files]) | \
                                            set([utt.split('/')[-3] for utt in veri_hard2.files]))))
    print(1)
