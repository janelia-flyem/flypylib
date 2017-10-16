#!/usr/bin/env python

"""flypylib - A library for synapse/object detection in
flyem-compatible biological images.

This package exports the following subpackages:
    fplerroranalysis
    fplmodels
    fplnetwork
    fplobjdetect
    fplsynapses
They have their own documentation; "pydoc flypylib.fplmodels"
will get you the fplmodels docs.

Please see
https://github.com/janelia-flyem/flypylib
for the current sources. This software is most easily setup using
the conda distribution of Python. 
"""

from .fplnetwork import FplNetwork
