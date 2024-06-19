## MELEP: A Novel Predictive Measure of Transferability in Multi-Label ECG Diagnosis

This repository contains code and resources related to the implementation of the paper `MELEP: A Novel Predictive Measure of Transferability in Multi-Label ECG Diagnosis`, published in the Journal of Healthcare Informatics Research (2024).

## Installation

```
git clone https://github.com/cuongvng/melep-ecg.git
cd melep-ecg
pip install -r requirements.txt
```

## Instructions

### Dataset Acquisition
The paper used publicly available datasets: PTB-XL, CPSC2018, Georgia, and Chapman-Shaoxing-Ningbo (CSN). Raw datasets can be found [here](https://physionet.org/content/challenge-2021/1.0.3/#files). Processed PTB-XL and CSN datasets for experiments can be found at the [corresponding folder](./data/).

### Pretraining Models
We provided pre-trained models used for our experiments at [this link](https://1drv.ms/f/s!AvPo7TVfqxSKi9l8W9JvgFh9MU08Cg?e=7Tigmw). 
Please follow instructions in [this repo](https://github.com/cuongvng/transfer-learning-ecg-diagnosis) if you want to pre-train those models from scratch.

### Running experiments

After installing required packages, download datasets and models, run the corresponding scripts to reproduce our experiments. For example:

```
python ptbxl_resnet_transfer.py
```


## Citation & Acknowledgements

If you find this work helpful, please cite:

```
@article{nguyen2024melep,
  title={MELEP: A Novel Predictive Measure of Transferability in Multi-label ECG Diagnosis},
  author={Nguyen, Cuong V and Duong, Hieu Minh and Do, Cuong D},
  journal={Journal of Healthcare Informatics Research},
  pages={1--17},
  year={2024},
  publisher={Springer}
}
```


## License
This project is licensed under the [CC-BY-4.0 license](LICENSE).
