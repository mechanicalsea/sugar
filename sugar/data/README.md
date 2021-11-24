# Berif Introduction of data Directory

The implementation of a series of `*.py` in data are used to build easy-to-use the loader of datasets. Here are examples as follows:

1. `voxceleb1.py` is used to build loaders of VoxCeleb1, specifically, training set, evaluation set for speaker identification task, and evaluation set for speaker verification task.
2. `voxceleb2.py` is used to build loaders of VoxCeleb2, specifically, only training set.
3. `cnceleb.py` is used to buld loaders of CN-Celeb1 and CN-Celeb2. (*There are something to be added in the future.*)

Note that evaluation set can be functionally seen as validation set or test set.

These implementation is based on some basic packages that consist of `database.py`, `waveforms.py`, and `features.py`.

## Added

- `veri_split`: it is created as `veri_test2` and `iden_split`. The utterances of speakers not in `veri_test2` and not in `iden_split` marked `3` or `1` are set to training data. The utterances of speakers in `iden_split` marked `2` are set to validate data. The utterances of speakers in `veri_test2` are set to test data as marked `3`.
  - 1: training data
  - 2: validate data
  - 3: test data  
- `iden_train`, `iden_dev`, and `iden_test`: splited from `iden_split` by using `iden_list.py`
  
## Officially Offered

- `iden_split`: voxceleb1 identification data list.
  - 1: training data
  - 2: validate data
  - 3: test data
- `veri_test2`, `list_test_all2`, `list_test_hard2`: voxceleb1 verification trials.
  - `veri_test2`: speakers here are as `iden_split`.3
  - `list_test_all2` and `list_test_hard2`: speakers are overlapped with `iden_split`.1/2
- `train_list`: voxceleb2 development data
- `test_list`: voxceleb1 test used in Clova Baseline

## Dataset

- `voxceleb1`: VoxCeleb1 wav files.
- `voxceleb2`: VoxCeleb2 wav files.
- `voxceleb1_cd`: voxsrc2020 wav files for evaluation offline.
- `voxsrc2020`: voxsrc 2020 wav files for evaluation online.
- `voxsrc2019`: voxsrc 2019 wav files for evaluation online.
