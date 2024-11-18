# Imbalanced Regression

<div align="center">
<img src="./SRL.png" width="800px" alt="Illustration of our geometric constraint-based approach"/>
</div>

##### This repository contains the code and datasets for **Improving Representation for Imbalanced Regression through Geometric Constraints**. This work aims to tackle the unique challenges of imbalanced regression by introducing advanced representation techniques that leverage geometric constraints to improve model performance across varying distributions.
---

---
## Usage

Please go into the sub-folder to run experiments for different datasets. 

- [STS-B-DIR (sentence similarity regression)](./sts-b-dir)

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

## Acknowledgment

Our codebase builds heavily on [DIR](https://github.com/YyzHarry/imbalanced-regression) and [RankSim](https://github.com/BorealisAI/ranksim-imbalanced-regression).

---

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

---
