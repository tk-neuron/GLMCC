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

Make sure to edit `params.py` before running.



