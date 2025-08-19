# Dynamic Collaborative Low-Rank Tuning for Efficient Model Adaptation

This repository includes the reference code for paper: 

**Dynamic Collaborative Low-Rank Tuning for Efficient Model Adaptation**

## Install Requirements
```
conda create -n DCLT python=3.10
conda activate DCLT
pip install -r requirements.txt
```
## Train
```
cd trainer
python train.py
```

## Acknowledgment
This repository is built upon the excellent work LoR-VP. We sincerely thank the authors for releasing their code and making their research publicly available.
```bibtex
@inproceedings{
jin2025lorvp,
title={LoR-{VP}: Low-Rank Visual Prompting for Efficient Vision Model Adaptation},
author={Can Jin and Ying Li and Mingyu Zhao and Shiyu Zhao and Zhenting Wang and Xiaoxiao He and Ligong Han and Tong Che and Dimitris N. Metaxas},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=5btFIv2PNb}
}
```
