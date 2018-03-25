# CURE-TSR

The overall goal of this project is to analyze the robustness of data-driven algorithms under diverse challenging conditions where trained models can possibly be depolyed. To achieve this goal, we introduced a large-sacle (>2,000,000 images) recognition dataset (CURE-TSR) which is among the most comprehensive dataset with controlled synthetic challenging conditions. Also, this repository contains codes to reproduce the benchmarking result for CNN presented in our NIPS workshop paper. For detailed information, please refer to our paper [CURE-TSR: Challenging Unreal and Real Environments for Traffic Sign Recognition](https://arxiv.org/abs/1712.02463) and [website](https://ghassanalregib.com/cure-tsr).

## Dataset
<p align="center">
<img src="./figs/signtype.png">
<img src="./figs/challtype.png">
</p> 


In order to receive  the download link, please fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSfjG211OENp4_QKFh86wLtFh-sa4HwkKq4hoWcAVKXN2QyICw/viewform) to submit your information and agree the conditions of use. These information will be kept confidential and will not be released to anybody outside the OLIVE administration team.


## Requirements
- Tested on Linux 14.04
- CUDA, CuDNN
- Anaconda (or virtualenv)
- PyTorch (www.pytorch.org)
- Optionally, tensorflow-cpu for tensorboard


## Usage

```
usage: train.py [-h] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR]
                [--momentum M] [--weight-decay W] [--print-freq N]
                [--resume PATH] [-e]
                DIR

CURE-TSR Training and Evaluation

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
```

- Training example:
```
python train.py --lr 0.001 ./CURE-TSR
```
- Testing example: You need to change the variable 'testdir' to test trained models on different challenging conditions. 

```
python train.py -e --resume  ./checkpoints/checkpoint.pth.tar
```

## Output example

```
*** Start Training *** 

Epoch: [55][0/29]       Time 0.258 (0.258)      Data 0.251 (0.251)      Loss 0.1454 (0.1454)  Prec@1 95.312 (95.312)  Prec@5 99.609 (99.609)
Epoch: [55][10/29]      Time 0.024 (0.048)      Data 0.021 (0.044)      Loss 0.1117 (0.1493)  Prec@1 96.875 (96.165)  Prec@5 99.609 (99.751)
Epoch: [55][20/29]      Time 0.120 (0.043)      Data 0.116 (0.039)      Loss 0.1565 (0.1480)  Prec@1 94.922 (96.112)  Prec@5 100.000 (99.814)

*** Start Testing *** 

Test: [0/14]    Time 0.227 (0.227)      Loss 1.2593 (1.2593)    Prec@1 66.406 (66.406)        Prec@5 94.922 (94.922)
Test: [10/14]   Time 0.005 (0.037)      Loss 2.2871 (0.9604)    Prec@1 62.109 (78.871)        Prec@5 87.109 (94.602)
 * Prec@1 81.254 Prec@5 94.991
```

## Citation

If you use CURE-TSR dataset and these codes, please consider citing:

```
@article{CURETSR,
Author = {D. Temel and G. Kwon* and M.Prabhuhankar* and G. AlRegib},
Journal = {Advances in Neural Information Processing Systems (NIPS) Machine Learning for Intelligent Transportations Systems Workshop},
Title = {{CURE-TSR: Challenging unreal and real environments for traffic sign recognition}},
note = {(*: equal contribution), \url{https://ghassanalregib.com/cure-tsr/}},
Year = {2017},
}
```
