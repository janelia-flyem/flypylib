# flypylib

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
