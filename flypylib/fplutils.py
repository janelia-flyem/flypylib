import numpy as np
from collections import namedtuple

szyx = namedtuple(
    'szyx', 'size z y x')

def to3d(vv):
    if(np.size(vv) == 1):
        vv = (vv,vv,vv)
    return vv

def set_filter(radius):
    flt_sz = 2*radius+1

    cc = np.mgrid[-radius:radius+1,-radius:radius+1,-radius:radius+1]
    dd = np.sqrt( cc[0,:,:,:]**2 + cc[1,:,:,:]**2 + cc[2,:,:,:]**2 )
    return dd<=radius

def roi_from_txt(filename):
    with open(filename,'r') as f_in:
        substacks = f_in.read().splitlines()
    roi = []
    for ss in substacks:
        roi.append(szyx(*[int(nn) for nn in ss.split(',')]))
    return [roi,]
