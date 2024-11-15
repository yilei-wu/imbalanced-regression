The imbalanced regression is based on the public repository of [Yang et al., ICML 2021](https://github.com/YyzHarry/imbalanced-regression/tree/main/imdb-wiki-dir). 


## Installation

#### Prerequisites

1. Download GloVe word embeddings (840B tokens, 300D vectors) using

```bash
python glove/download_glove.py
```

2. We use the standard file (`./glue_data/STS-B`) provided by Yang et al.(ICML 2021), which is used to set up balanced STS-B-DIR dataset. To reproduce the results in the paper, please directly use this file. If you want to try different balanced splits, you can delete the folder `./glue_data/STS-B` and run

```bash
python glue_data/create_sts.py
```


#### Dependencies

The required dependencies for this task are quite different to other three tasks, so it's better to create a new environment for this task. If you use conda, you can create the environment and install dependencies using the following commands:

```bash
conda create -n sts python=3.6
conda activate sts
# PyTorch 0.4 (required) + Cuda 9.2
conda install pytorch=0.4.1 cuda92 -c pytorch
# other dependencies
pip install -r requirements.txt
# The current latest "overrides" dependency installed along with allennlp 0.5.0 will now raise error. 
# We need to downgrade "overrides" version to 3.1.0
pip install overrides==3.1.0
```


## Getting Started

### 1. running 

```
python train.py --dfr --w1 1e-4 --w2 1e-2 --w3 1e-4 --temp 0.1
```