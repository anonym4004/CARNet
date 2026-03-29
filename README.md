# CARNet

CARNet is an attention-free model for multivariate time series forecasting that captures cross-variate dependencies and global periodic patterns efficiently. It integrates cycle-conditioned aggregation with Multihead Core Aggregation to deliver strong performance while maintaining linear-complexity modeling.


## Dataset

Due to file size limitations, the datasets are hosted externally and can be downloaded from the link below:

[Download datasets](https://drive.google.com/drive/folders/1_iLs9_T4RmZMYa0q8Eqy5fSP1l1X5D9Z)

After downloading, place the files in the dataset directory expected by the experiment scripts.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
``` 


## Reproducing Experiments

After installing the dependencies, you can reproduce an experiment by running:

```bash
bash ./scripts/CARNet/ECL.sh
```
