# Imbalanced Regression

##### This repository contains the code and datasets for **Improving Representation for Imbalanced Regression through Geometric Constraints**. This work aims to tackle the unique challenges of imbalanced regression by introducing advanced representation techniques that leverage geometric constraints to improve model performance across varying distributions.
---
## Features

- Implements multiple strategies for handling imbalanced regression tasks effectively.
- Provides preprocessing and evaluation scripts tailored to datasets such as AgeDB, IMDb-Wiki, and STS-B.
- Contains utilities for generating simulated datasets and performing ranking-based regression tasks.

---

## Installation

Clone the repository:

   ```bash
   git clone https://github.com/yilei-wu/imbalanced-regression.git
   cd imbalanced-regression
   ```


---
## Usage

Please go into the sub-folder to run experiments for different datasets. 

- [IMDB-WIKI-DIR (age estimation)](./imdb-wiki-dir)
- [AgeDB-DIR (age estimation)](./agedb-dir)
- [STS-B-DIR (sentence similarity regression)](./sts-b-dir)
- [OL-DIR (operater learning)](./OL)

---

## File Structure

The repository is organized as follows:

```
imbalanced-regression/
├── OL/                    # Ordinal Learning: scripts and utilities for ordinal regression
│   ├── main.py            # Main script for training and evaluation
│   ├── dfr.py             # Deep feature regression module
│   ├── OrdinalEntropy.py  # Ordinal loss functions
│   ├── train.npz          # Training data for experiments
│   ├── test.npz           # Test data for experiments
│   └── ...                # Additional helper scripts and configs
│
├── agedb-dir/             # AgeDB dataset for age regression tasks
│   ├── datasets.py        # Dataset handling for AgeDB
│   ├── preprocess_agedb.py # Preprocessing script for AgeDB
│   ├── data/              # Raw and processed AgeDB data files
│   └── README.md          # Instructions and details for AgeDB usage
│
├── imdb-wiki-dir/         # IMDb-Wiki dataset for age regression
│   ├── datasets.py        # Dataset handling for IMDb-Wiki
│   ├── preprocess_imdb_wiki.py # Preprocessing script for IMDb-Wiki
│   ├── data/              # Raw and processed IMDb-Wiki data files
│   └── README.md          # Usage details for IMDb-Wiki
│
├── sts-b-dir/             # STS-B dataset for semantic textual similarity regression
│   ├── preprocess.py      # Preprocessing and data preparation for STS-B
│   ├── tasks.py           # Task-specific configurations and operations
│   ├── trainer.py         # Training and evaluation scripts
│   └── glue_data/         # Raw and preprocessed STS-B dataset files
│
└── loss.py                # Loss functions for regression tasks
```

---

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

---

Let me know if you need further customizations or specific sections added!
