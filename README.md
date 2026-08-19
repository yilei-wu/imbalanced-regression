

<div align="center">
<h1>🧶 Improving Representation for Imbalanced Regression <br> through Geometric Constraints </h1>
</div>


> **Improve Representation for Imbalanced Regression through Geometric Constraints _(CVPR 2025)_** <br>
> Zijian Dong<sup>1*</sup>, Yilei Wu<sup>1*</sup>, Chongyao Chen<sup>2*</sup>, Yingtian Zou<sup>1</sup>, Yichi Zhang<sup>1</sup>, Juan Helen Zhou<sup>1</sup> <br>
> <sup>1</sup>National University of Singapore, <sup>2</sup>Duke University, <sup>*</sup>Equal contribution
> <a href="https://arxiv.org/abs/2503.00876"><img src="https://img.shields.io/badge/Paper-Arxiv-darkred.svg" alt="Paper"></a>


<div align="center">
<img src="./SRL.png" width="800px" alt="Illustration of our geometric constraint-based approach"/>
</div>

## 💡 Introduction

Our paper addresses representation learning for imbalanced regression by introducing two geometric constraints: **enveloping loss**, which encourages representations to uniformly occupy a hypersphere's surface, and **homogeneity loss**, which ensures evenly spaced representations along a continuous trace. Unlike classification-based methods that cluster features into distinct groups, our approach preserves the continuous and ordered nature essential for regression tasks. We integrate these constraints into a **Surrogate-driven Representation Learning (SRL)** framework. Experiments on several datasets demonstrate significant performance improvements, especially in regions with limited data.

## 🔧 Usage

An example dataset is provided as follows.

- [STS-B-DIR (sentence similarity regression)](./sts-b-dir)
- [IMDB-WIKI-DIR (age estimation)](./imdb-wiki-dir)

## 💻 Pretrained Weights

We provide our model weights trained on [DIR benchmark datasets](https://github.com/YyzHarry/imbalanced-regression):

- [STS-B-DIR (sentence similarity regression)](https://drive.google.com/file/d/1f1BJWWXNHZUoUBYcxQaFt7kslxzYX_7R/view?usp=sharing)
- [IMDB-WIKI-DIR (age estimation)](https://drive.google.com/file/d/1ihFPTDf43J4AYPU2B-8DE4b8vm8AChD2/view?usp=sharing)
- [AgeDB-DIR (age estimation)](https://drive.google.com/file/d/1G5LWUVnT7cDf4h6wnbEwuwa_Hh6VQrkc/view?usp=drive_link)

## 📂 File Structure

The repository is organized as follows:

```
imbalanced-regression/
├── sts-b-dir/             # STS-B dataset for semantic textual similarity regression
│   ├── preprocess.py      # Preprocessing and data preparation for STS-B
│   ├── dfr.py             # Method implementation
│   ├── evaluate.py        # Evaluation scripts for model performance
│   ├── models.py          # Model architectures for the regression tasks
│   ├── tasks.py           # Task-specific configurations and operations
│   ├── trainer.py         # Training and evaluation pipelines
│   ├── train.py           # Script to initiate the training process
│   └── glue_data/         # Directory containing raw and preprocessed STS-B 
├── imdb-wiki-dir/         # IMDB-WIKI dataset for age estimation
│   ├── datasets.py        # Preprocessing and data preparation for IMDB-WIKI
│   ├── data               # dataset directory
│   ├── dfr.py             # Method implementation
│   ├── resnet.py          # Network implementation
│   ├── evaluate.py        # Evaluation pipelines
│   └── utils.py/          # Directory containing utility functions
├── agedb-dir/             # AgeDB dataset for age estimation
│   ├── evaluate.py        # Evaluation pipelines
│   ├── data               # dataset directory
│   ├── dfr.py             # Method implementation
│   ├── resnet.py          # Network implementation
│   ├── dataset.py         # Preprocessing and data preparation for IMDB-WIKI
│   └── utils.py/          # Directory containing utility functions

```

## 🧑🏻‍💻 Running (STS-B-DIR)

1. Download GloVe word embeddings (840B tokens, 300D vectors) using

```bash
python glove/download_glove.py
```

2. We use the standard file (`./glue_data/STS-B`) provided by [DIR](https://github.com/YyzHarry/imbalanced-regression), which is used to set up balanced STS-B-DIR dataset. To reproduce the results in the paper, please directly use this file. If you want to try different balanced splits, you can delete the folder `./glue_data/STS-B` and run

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

4. training

```bash
python train.py --dfr --w1 1e-4 --w2 1e-2 --w3 1e-4 --temp 0.1
```

## 🧑🏻‍ Evaluating

```bash
python evaluate.py --evaluate --resume <path_to_evaluation_ckpt> #agedb-dir & imdb-wiki-dir
```
```bash
python evaluate.py --evaluate --eval_model <path_to_evaluation_ckpt> #sts-b-dir
```


---

## Acknowledgment

Our codebase was built on [DIR](https://github.com/YyzHarry/imbalanced-regression) and [RankSim](https://github.com/BorealisAI/ranksim-imbalanced-regression). Thanks for their wonderful work!

---

## Citation

If you find this repository useful in your research, please consider giving a star ⭐️ and a citation:

```
@inproceedings{dong2025improve,
  title={Improve Representation for Imbalanced Regression through Geometric Constraints},
  author={Dong, Zijian and Wu, Yilei and Chen, Chongyao and Zou, Yingtian and Zhang, Yichi and Zhou, Juan Helen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
