# flypylib

## overview

This code provides an implementation of a system for object detection
(training and inference), for the use of detecting pre-synaptic sites
(T-bars), as described in:

Fully-Automatic Synapse Prediction and Validation on a Large Data Set\
Gary B. Huang, Louis K. Scheffer, Stephen M. Plaza\
<https://arxiv.org/abs/1604.03075>, 2016

The overall pipeline is the same as presented in the paper, but the
underlying voxel prediction network has been improved and replaced
with a U-Net style CNN.

## installation

via conda / [miniconda](http://conda.pydata.org/miniconda.html)

```
# linux:
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# activate conda
CONDA_ROOT=`conda info --root`
source ${CONDA_ROOT}/bin/activate root

# install flypylib, and optionally ipython and matplotlib
conda create -n <NAME> -c flyem-forge -c conda-forge flypylib ipython matplotlib
```

## usage

See
[scripts/fpl_fib25_example.py](https://github.com/janelia-flyem/flypylib/blob/master/scripts/fpl_fib25_example.py)
for a small example of training and inference, using [FlyEM's medulla
seven column data](http://emdata.janelia.org/#/repo/medulla7column).
