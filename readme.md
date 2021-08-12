# Signed Graph Neural Networks

This repository offers Pytorch implementations for [Signed Graph Attention Networks](./readme_sigat.md) and [SDGNN: Learning Node Representation for Signed Directed Networks](./readme_sdgnn.md)


## Overview

Two sociological theories (ie balance theory and status theory) play a vital role in the analysis and modeling of directed networks.

How to integrate these sociological theories with graph neural networks and efficiently model signed directed graph is the key issue of this project

## Installation

```
pip install -r requirements.txt
```

## Run Example

### SiGAT

```python
python sigat.py
```

### SDGNN
```
python sdgnn.py
```

## Bibtex
Please cite our paper if you use this code in your own work:

```
@inproceedings{huang2019signed,
  title={Signed graph attention networks},
  author={Huang, Junjie and Shen, Huawei and Hou, Liang and Cheng, Xueqi},
  booktitle={International Conference on Artificial Neural Networks},
  pages={566--577},
  year={2019},
  organization={Springer}
}
```

```
@inproceedings{huang2021sdgnn,
  title={SDGNN: Learning Node Representation for Signed Directed Networks},
  author={Huang, Junjie and Shen, Huawei and Hou, Liang and Cheng, Xueqi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={1},
  pages={196--203},
  year={2021}
}
```
