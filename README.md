# Python implementation of GLMCC

## Overview

This is unofficial Python3 implementation of GLMCC (generalized linear model for spike cross-correlogram).

Original paper: Kobayashi, R., Kurita, S., Kurth, A. et al. Reconstructing neuronal circuitry from parallel spike trains. Nat Commun 10, 4468 (2019). [https://doi.org/10.1038/s41467-019-12225-2](https://www.nature.com/articles/s41467-019-12225-2)  
Original code: [github](https://github.com/NII-Kobayashi/GLMCC)



## Requirement

* Numpy >= 1.16.1
* Scipy >= 1.1.0
* Matplotlib >= 3.1.1



## Examples

In `./examples`, there is an example jupyter notebook `example.ipynb` that explains how GLMCC works with sample spike train data `sample_data.csv`.

Below is bokeh visualization of fitted GLMCC (example code is in `./examples/bokeh_visualization.ipynb`).

![Bokeh Visualization](https://user-images.githubusercontent.com/38024515/98433558-85ea1700-210b-11eb-8a9e-e737062bfb8f.png)
