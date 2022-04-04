# Sugar

Efficient Speech Processing Tookit for Automatic Speaker Recognition

| **[HuggingFace](https://huggingface.co/mechanicalsea/efficient-tdnn)** |

## What's New

![supernet](./tutorials/EfficientTDNN/supernet.png)

- [EfficientTDNN: Efficient Architecture Search for Speaker Recognition](https://arxiv.org/abs/2103.13581)

## Models and Checkpoints

| Model | Training Dataset | Link |
|:---:|:---:|:---:|
| EfficientTDNN | VoxCeleb2 | HuggingFace: [mechanicalsea/efficient-tdnn](https://huggingface.co/mechanicalsea/efficient-tdnn) |

## Requirements and Installation

- PyTorch version >= 1.7.1
- Python version >= 3.7.9
- To install sugar:

```bash
git clone https://github.com/mechanicalsea/sugar.git
cd sugar
pip install --editable .
```

We provide pre-trained models for extracting speaker embeddings via [huggingface](https://huggingface.co/mechanicalsea/efficient-tdnn#compute-your-speaker-embeddings).

## Tutorials

- EfficientTDNN
  - [the evalution of a subnet](./tutorials/EfficientTDNN/EfficientTDNN.ipynb)
  - [evaluation of different subnets from a given supernet](./tutorials/EfficientTDNN/SubnetEvaluation.ipynb)
  - [search on the trained supernet](./tutorials/EfficientTDNN/TDNN-NAS.ipynb)

## EfficientTDNN Training Scripts

An example of training the TDNN supernet is given as follows.

```sh
task=''            # training stage, e.g., supernet, supernet_kernel, supernet_kernel_depth, ...
cycle_step=34120   # a half cycle of learning rate scheduler, e.g., GPU 1: 68248, GPU 2: 34120, GPU 4: 17056
second=48000       # largest stage: 32000, other stages: 48000
initial_weights='' # required to these stages except for the largest stage
logdir=''
python scripts/train_vox2_veri.py --distributed --augment \
      --task supernet_width1 --epochs 64 --cycle-step ${cycle_step} --second ${second} \
      --initial-weights ${initial_weights} \
      --logdir ${logdir}
```

## Citing EfficientTDNN

Please, cite EfficientTDNN if you use it for your research or business.

```bibtex
@article{rwang-efficienttdnn-2021,
  title={{EfficientTDNN}: Efficient Architecture Search for Speaker Recognition},
  author={Rui Wang and Zhihua Wei and Haoran Duan and Shouling Ji and Yang Long and Zhen Hong},
  journal={arXiv preprint arXiv:2103.13581},
  year={2021},
  eprint={2103.13581},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2103.13581}
}
```

## Contact Information

For help or issues using EfficientTDNN models, please submit a GitHub issue.

For other communications related to EfficientTDNN, please contact Rui Wang (`rwang@tongji.edu.cn`).
