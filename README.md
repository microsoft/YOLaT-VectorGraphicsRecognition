# :fire: [NIPS2021, TPAMI2024] YOLaT & YOLaT++-VectorGraphicsRecognition

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2111.03281)

This repository is the official PyTorch implementation of our NeurIPS-2021 [paper](https://arxiv.org/abs/2111.03281): Recognizing Vector Graphics without Rasterization.

## Environments
```sh
conda create -n your_env_name python=3.8
conda activate your_env_name
sh deepgcn_env_install.sh 
```

## YOLaT 
### 1. Data Preparation

#### Floorplans
a) Download and unzip the [Floorplans dataset](http://mathieu.delalandre.free.fr/projects/sesyd/symbols/floorplans.html) to the dataset folder: `data/FloorPlansGraph5_iter`

b) Run the following scripts to prepare the dataset for training/inference.

```sh
cd utils
python svg_utils/build_graph_bbox.py
```
#### Diagrams
a) Download and unzip the [Diagrams dataset](http://mathieu.delalandre.free.fr/projects/sesyd/symbols/diagrams.html) to the dataset folder: `data/diagrams`

b) Run the following scripts to prepare the dataset for training/inference.
```sh
cd utils
python svg_utils/build_graph_bbox_diagram.py
```

### 2. Training & Inference
#### Floorplans
```sh
cd cad_recognition
CUDA_VISIBLE_DEVICES=0 python -u train.py --batch_size 4 --data_dir data/FloorPlansGraph5_iter --phase train --lr 2.5e-4 --lr_adjust_freq 9999999999999999999999999999999999999 --in_channels 5 --n_blocks 2 --n_blocks_out 2 --arch centernet3cc_rpn_gp_iter2  --graph bezier_cc_bb_iter --data_aug true  --weight_decay 1e-5 --postname run182_2 --dropout 0.0 --do_mixup 0 --bbox_sampling_step 10
```
#### Diagrams
```sh
cd cad_recognition
CUDA_VISIBLE_DEVICES=0 python -u train.py --batch_size 4 --data_dir data/diagrams --phase train --lr 2.5e-4 --lr_adjust_freq 9999999999999999999999999999999999999 --in_channels 5 --n_blocks 2 --n_blocks_out 2 --arch centernet3cc_rpn_gp_iter2  --graph bezier_cc_bb_iter --data_aug true  --weight_decay 1e-5 --postname run182_2 --dropout 0.0 --do_mixup 0 --bbox_sampling_step 5
```

## YOLaT++



## Citation
BibTex:
```
@inproceedings{jiang2021recognizing,
title={{Recognizing Vector Graphics without Rasterization}},
author={Jiang, Xinyang and Liu, Lu and Shan, Caihua and Shen, Yifei and Dong, Xuanyi and Li, Dongsheng},
booktitle={Proceedings of Advances in Neural Information Processing Systems (NIPS)},
volume={34},
number={},
pages={},
year={2021}}

@inproceedings{yolat24,
title={{Hierarchical Recognizing Vector Graphics and A New Chart-based Vector Graphics Dataset}},
author={Shuguang Dou, Xinyang Jiang, Lu Liu, Lu Ying, Caihua Shan, Yifei Shen, Xuanyi Dong, Yun Wang, Dongsheng Li, Cairong Zhao},
booktitle={Proceedings of Advances in Neural Information Processing Systems (NIPS)},
volume={},
number={},
pages={},
year={2024}}

```  
Please do consider :star2: star our project to share with your community if you find this repository helpful!

# Related Dataset
[Benchmark for VG-based Detection and Chart Understanding](https://github.com/Vill-Lab/2024-TPAMI-VGDCU)

