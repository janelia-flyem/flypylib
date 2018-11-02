import skimage.exposure
import numpy as np
import h5py
from collections import namedtuple

szyx = namedtuple(
    'szyx', 'size z y x')

def to3d(vv):
    if(np.size(vv) == 1):
        vv = (vv,vv,vv)
    return vv

def set_filter(radius, return_dist=False):
    flt_sz = 2*radius+1

    cc = np.mgrid[-radius:radius+1,-radius:radius+1,-radius:radius+1]
    dd = np.sqrt( cc[0,:,:,:]**2 + cc[1,:,:,:]**2 + cc[2,:,:,:]**2 )

    if return_dist:
        return dd<=radius, dd
    return dd<=radius

def roi_from_txt(filename):
    with open(filename,'r') as f_in:
        substacks = f_in.read().splitlines()
    roi = []
    for ss in substacks:
        roi.append(szyx(*[int(nn) for nn in ss.split(',')]))
    return [roi,]


def clahe(im_in, kernel_size, fn_out=None, zero_mean=False):
    if(isinstance(im_in, str)):
        im_in = h5py.File(im_in,'r')['/main'][:]

    if(im_in.dtype != 'float32'):
        im_in = im_in.astype('float32')

    im_out = np.zeros(im_in.shape, 'float32')

    im_min = np.min(im_in)
    im_max = np.max(im_in)
    for ii in range(im_in.shape[0]):
        im_slice = im_in[ii,:,:]
        im_slice = (im_slice - im_min)/(im_max - im_min)
        im_clahe = skimage.exposure.equalize_adapthist(
            im_slice, kernel_size)

        im_out[ii,:,:] = im_clahe.astype('float32')

    if zero_mean:
        im_out = im_out - 0.5

    if fn_out is not None:
        hh = h5py.File(fn_out,'w')
        hh.create_dataset('/main', im_out.shape,
                          dtype='float32', compression='gzip')
        hh['/main'][:] = im_out
        hh.close()

    return im_out
