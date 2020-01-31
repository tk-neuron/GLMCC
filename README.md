# Python implementation of GLMCC

## Overview

This is **unofficial** Python implementation of GLMCC (generalized linear model for spike cross-correlogram).

Original paper: [Kobayashi et al., Nature Communications 2019](https://www.nature.com/articles/s41467-019-12225-2)  
Original code: [github](https://github.com/NII-Kobayashi/GLMCC)

## Requirement

* Numpy 1.16.1
* Scipy 1.1.0
* Matplotlib 3.1.1

## Getting Started

You can test with the sample dataset (`datasets/`)

To estimate all the connectivity on the sample dataset, just run:

```shell
python3 estimation.py
```

For your original dataset, you need to customize `params.py`.  This file configures file paths and GLMCC settings.  For details, please see the code.

To plot cross-correlogram and fitted GLMCC for a particular pair of two neurons, just run:

```shell
python3 glmplot.py
```

Please make sure to edit `params.py` before running.

## Details
There are two script files that you can run: `estimation.py` and  `glmplot.py`.

1. `estimation.py`  
this code estimates all connectivity on the dataset, and returns connectivity values in csv.
output file's name is `./results/J_[end time in sec]_[firing threshold]Hz.csv`.   
csv format: `reference id, target id, J_ij, J_ji, J_ij threshold, J_ji threshold, max log posterior`

2. `glmplot.py`  
this code estimates single connectivity between reference id and target id specified in `params.py`, and
draws a figure of CC and fitted GLM.  The graph is saved in `./results/cc_from[reference]to[target].png`.

