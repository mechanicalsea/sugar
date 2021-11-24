# Sugar: Efficient Speech Processing Tookit for Automatic Speaker Recognition

Sugar is developed based on Pytorch and Numpy for the task of speaker recognition (or speaker diarisation).

Wish everyone who is interested in speaker recognition can enjor as people like sugar.

Thanks for promising open source softwares and tools, such as [Kaldi](https://github.com/kaldi-asr/kaldi), [ASV-Subtools](https://github.com/Snowdar/asv-subtools), etc.

| Copyright | Author |
|:---------:|:------:|
| MIT | Rui Wang |

---

[TOC]

---

## Introduction

In Sugar, Numpy is used to extract acoustic features and scoring in the back-end, and PyTorch is used to build network-like model freely and train it with a pleasant style. Also, some useful functions and tools, as presented in Kaldi, are reproduced for creating Pythonic programming.

The project is illustrated as follows. It can be helpful to have some insights into Sugar.

### Project

Sugar contains five parts:

- **Core**: This part includes all python codes that achieve classes and methods.
- **Demo**: This part is a set of notebooks that present helpful tutorials.
- **Source**: This part demonstrates where and when those core codes and those demos comes from that can be sourced.
- **Docs**: A detailed reference that is better than README.
- **Resources**: A list of files, e.g., pretrained weights, well-organized datasets.

As mentioned above, everyone, who is interested in automatic speaker recognition, can pass guidance notebook, download necessary files, use appropriate codes, and understand doing works.

Note that all codes are commented as Numpy comments.

### Task Workflow

The basic task workflow is provided from a data-driven task, so it is not complex if you have idea of machine learning and deep learning project. Also, the task workflow can help your find where is need to be replaced or improved.

[此处应该有图片，来描述 task workflow]

### Data Pipeline

Based on machine learning and deep learning project, the data pipeline is key to understanding how the system works. A lot of functions in Kaldi are reproduced from pythonic programming.

[此处应该有图片，来描述 data pipeline]

Of course, the computational efficiency is going to be improved during the development cycle of the project.

## Package Category

### 1. Basic Package

- database
- waveform
- features
- vectors
- scores

### 2. Model-dependent Package

- models
- architectures
- loss
- optimizer
- scheduler

### 3. Task-dependent Package

- data
- works

## Support List

- Multi-GPU Training Solution

    > Currently, no solutions.

- Multi-CPU Implementation

  - [ ] Parallel For-List Function

- Feature Extraction

  - [ ] MFCC
  - [ ] PLP
  - [ ] Mel-scale FBank
  - [ ] Spectrogram

- Front-end Operations

  - [ ] Reverberate with RIRs
  - [ ] Augment wtih Noise and Music
  - [ ] SpecAugment via Time Wrap, Time Mask, and Frequency Mask
  - [ ] Mixup
  - [ ] Cepstral Mean Subtraction (CMS), Cepstral Variance Normalization (CVN), and Cepstral Mean and Variance Normalization (CMVN)

- Dataset Workflow

  - [ ] A Interface to Load a List of Inputs' Files
  - [ ] A Interface to Load batch-style Inputs
  - [ ] Batch Sampler to Custom Requirements
  - [ ] Multi-CPU Support

- Speaker Modeling

  - Model Architecture
  - [ ] Extended-TDNN (E-TDNN)
    - [ ] Factorized-TDNN (F-TDNN)
    - [ ] ResNet34SE
    - [ ] ResNet34SE Aggregating Former Feature Layers
  - Pooling Layer (Sub Model)
  - [ ] Statistics Pooling
    - [ ] Self-Attentive Pooling
    - [ ] Multi-Head Attentive Pooling

- Loss Function

  - [ ] Softmax Loss
  - [ ] AM-Softmax Loss (CosFace)
  - [ ] AAM-Softmax Loss (ArcFace)
  - [ ] Center Loss
  - [ ] Generalized End-to-End Loss (GE2E)

- Learning System

  - [ ] Classification Objective
  - [ ] Metric Learning Objective

- Optimizer

  - [ ] 自然随机梯度下降 Natural Stochastic Gradient Descent (NSGD)
    > Povey, D., Zhang, X., Khudanpur, S., 2015. Parallel training of DNNs with natural gradient and parameter averaging, in: 3rd International Conference on Learning Representations, ICLR 2015 - Workshop Track Proceedings.

- Training Strategy

  - [ ] Softmax Loss followed by AAM-Softmax Loss (ArcFace)
  - [ ] Progressively Increasing Learning Rate for Learning Process
  - [ ] Progressively Decreasing Learning Rate for Learning Process

- Vector Normalization and Compensation

  - [ ] Length Normalization

  - [ ] Linear Discriminant Analysis (LDA)

    > 1. Scikit-learn 文档：<https://scikit-learn.org/stable/modules/lda_qda.html#linear-and-quadratic-discriminant-analysis>
    > 2. Scikit-learn API：<https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis>

  - [ ] Nuisance Attribute Projection (NAP)

  - [ ] Within-Class Covariance Normalization (WCCN)

- Back-end Operations
  - [ ] Similarity Estimator: Cosine, Norm2, PLDA LLR
  - [ ] Basic Classifier: Logistic Regression
  - [ ] Score Normalization: T-Norm, Z-Norm, S-Norm, AS-Norm
  - [ ] Calibration and Fusion
  - [ ] Metric: EER, minDCF

- Practical Tools

    > No idea. Refer to Kaldi tools.

## How to Start

Workspace directory: `/workspace/projects/sugar`

## Tutorial Demo

## How to Post a Issue

## How to Make a Contribution

## Acknowledgement

Inspired by works for VoxSRC 2020, I was determined to make a contribution via developping open source software for automatic speaker recogniton.

More experiments, more "hungry".
