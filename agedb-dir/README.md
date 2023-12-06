# ConR on AgeDB-DIR
This repository contains the implementation of __ConR__ on *AgeDB-DIR* dataset. 

The imbalanced regression framework and LDS+FDS are based on the public repository of [Gong et al., ICML 2022](https://github.com/BorealisAI/ranksim-imbalanced-regression). 



## Installation

#### Prerequisites

1. Download AgeDB dataset from [here](https://ibug.doc.ic.ac.uk/resources/agedb/) and extract the zip file (you may need to contact the authors of AgeDB dataset for the zip password) to folder `./data` 

2. We use the standard train/val/test split file (`agedb.csv` in folder `./data`) provided by Yang et al.(ICML 2021), which is used to set up balanced val/test set. To reproduce the results in the paper, please directly use this file. You can also generate it using

```bash
python data/create_agedb.py
python data/preprocess_agedb.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- tensorboard_logger
- numpy, pandas, scipy, tqdm, matplotlib, PIL, wget

## Code Overview

#### Main Files

- `train.py`: main training and evaluation script
- `create_agedb.py`: create AgeDB raw meta data
- `preprocess_agedb.py`: create AgeDB-DIR meta file `agedb.csv` with balanced val/test set

#### Main Arguments

- `--data_dir`: data directory to place data and meta file
- `--reweight`: cost-sensitive re-weighting scheme to use
- `--loss`: training loss type
- `--conr`: wether to use ConR or not.
- `-w`: distance threshold (default 1.0) 
- `--beta`: the scale of ConR loss (default 4.0)
- `-t`: temperature(default 0.2)
- `-e`: pushing power scale(default 0.01)
## Getting Started

### 1. Train baselines

To use Vanilla model

```bash
python train.py --batch_size 64 --lr 2.5e-4
```



### 2. Train a model with ConR
##### batch size 64, learning rate 2.5e-4

```bash
python train.py --batch_size 64 --lr 2.5e-4 --conr -w 1.0 --beta 4.0 -e 0.01
```



### 3. Evaluate and reproduce

If you do not train the model, you can evaluate the model and reproduce our results directly using the pretrained weights from the anonymous links below.

```bash
python train.py --evaluate [...evaluation model arguments...] --resume <path_to_evaluation_ckpt>
```





