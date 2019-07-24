## Signed Graph Attention Network

## Overview
This paper is accepted at ICANN2019.

<div align=center>
 <img src="./imgs/SiGAT.png" alt="Sigat" align=center/>
</div>

> We provide a Pytorch implementation of Signed Graph Attention Network, which incorporates graph motifs into GAT to capture two well-known theories in signed network research, i.e., balance theory and status theory.

## Requirements

The script has been tested running under Python 3.6.3, with the following packages installed (along with their dependencies):

```
pip install -r requirements.txt
```

## Run the Code

```
python sigat.py
```

## Parameters

```
--epochs                INT         Number of SiGAT training epochs.     Default is 100. 
--seed                  INT         Random seed value.                   Default is 13.
--learning-rate         FLOAT       Learning rate.                       Default is 0.0005.  
--weight-decay          FLOAT       Weight decay.                        Default is 10^-5. 

```


## Run Example

```
python sigat_demo.py
```

## Cite
Please cite our paper if you use this code in your own work:


## Acknowledgement

> Some codes adapted from [paper](https://dl.acm.org/citation.cfm?id=1772756)

