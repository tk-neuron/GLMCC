# Python implementation of GLMCC

## Overview
Python3 implementation of GLMCC (generalized linear model for spike cross-correlogram).   
Note that this code is not reviewed by original authors. This code is a refactored version and can be used in a sklearn-like style.

<img src="https://user-images.githubusercontent.com/38024515/196653782-52a53d7b-9268-4446-8031-a41f3a45122e.png" width="500px">

### Credit
* Original paper: Kobayashi, R., Kurita, S., Kurth, A. et al. Reconstructing neuronal circuitry from parallel spike trains. Nat Commun 10, 4468 (2019). [https://doi.org/10.1038/s41467-019-12225-2](https://www.nature.com/articles/s41467-019-12225-2)  
* Original code: https://github.com/NII-Kobayashi/GLMCC

## Setup
* clone this repository to your local environment.
* At the root of this repository, run `pip install .`.
* After successfully installing GLMCC, you can use the module by ordinary import.

## Usage
```python
from glmcc import GLMCC  # model
from glmcc import spiketime_relative

spiketrains = {
  1: [],
  2: [],
  // ... //
}  # prepare your spiketrain data [ms]

# relative spiketime (target - reference)
t_sp = spiketime_relative(spiketime_tar=spiketrains[TARGET_NEURON_ID], 
                          spiketime_ref=spiketrains[REFERENCE_NEURON_ID], window_size=50.0)

glm = GLMCC(delay=1.0)  # tune synaptic delay [ms]
glm.fit(t_sp)
glm.summary()  # print fitting summary

print(glm.theta[-2], glm.theta[-1])  # estimated synaptic weights
```

For details, please take a look at the notebook in [`examples`](https://github.com/tk-neuron/GLMCC/blob/master/examples/example.ipynb) directory with sample data.
