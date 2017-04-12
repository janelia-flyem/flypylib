import numpy as np

def to3d(vv):
    if(np.size(vv) == 1):
        vv = (vv,vv,vv)
    return vv

def set_filter(radius):
    flt_sz = 2*radius+1

    cc = np.mgrid[-radius:radius+1,-radius:radius+1,-radius:radius+1]
    dd = np.sqrt( cc[0,:,:,:]**2 + cc[1,:,:,:]**2 + cc[2,:,:,:]**2 )
    return dd<=radius
