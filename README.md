# Improve Representation for Imbalanced Regression through Geometric Constraints (CVPR2025)

<div align="center">
<img src="./SRL.png" width="800px" alt="Illustration of our geometric constraint-based approach"/>
</div>


##### This repository contains the code and datasets for **Improving Representation for Imbalanced Regression through Geometric Constraints**. This work aims to tackle the unique challenges of imbalanced regression by introducing advanced representation techniques that leverage geometric constraints to improve model performance across varying distributions.

---


## Usage

Please navigate to the sub-folder to run experiments on different datasets. An example dataset is provided as follows.

- [STS-B-DIR (sentence similarity regression)](./sts-b-dir)


## Pretrained Weight

- [STS-B-DIR (sentence similarity regression)](https://drive.google.com/file/d/1f1BJWWXNHZUoUBYcxQaFt7kslxzYX_7R/view?usp=sharing)
- [IMDB-WIKI-DIR (age estimation)](https://drive.google.com/file/d/1yTlDQOpWFGIfhAl8nMZ2_tFE3n00FLrc/view?usp=sharing)
- [AgeDB-DIR (age estimation)](https://drive.google.com/file/d/1G5LWUVnT7cDf4h6wnbEwuwa_Hh6VQrkc/view?usp=drive_link)

---

## File Structure

The repository is organized as follows:

```
imbalanced-regression/
├── sts-b-dir/             # STS-B dataset for semantic textual similarity regression
│   ├── preprocess.py      # Preprocessing and data preparation for STS-B
│   ├── dfr.py             # Implementation of Decoupled Feature Representation (DFR) methods
│   ├── evaluate.py        # Evaluation scripts for model performance
│   ├── models.py          # Model architectures for the regression tasks
│   ├── tasks.py           # Task-specific configurations and operations
│   ├── trainer.py         # Training and evaluation pipelines
│   ├── train.py           # Script to initiate the training process
│   └── glue_data/         # Directory containing raw and preprocessed STS-B 
```

---


### Running (STS-B-DIR)

1. Download GloVe word embeddings (840B tokens, 300D vectors) using

```bash
python glove/download_glove.py
```

2. We use the standard file (`./glue_data/STS-B`) provided by Yang et al.(ICML 2021), which is used to set up balanced STS-B-DIR dataset. To reproduce the results in the paper, please directly use this file. If you want to try different balanced splits, you can delete the folder `./glue_data/STS-B` and run

```bash
python glue_data/create_sts.py
```

3. The required dependencies for this task are quite different to other three tasks, so it's better to create a new environment for this task. If you use conda, you can create the environment and install dependencies using the following commands:

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

4. running
```
python train.py --dfr --w1 1e-4 --w2 1e-2 --w3 1e-4 --temp 0.1
```



---

## Acknowledgment

Our codebase builds heavily on [DIR](https://github.com/YyzHarry/imbalanced-regression) and [RankSim](https://github.com/BorealisAI/ranksim-imbalanced-regression).

---


## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```


