"""functions for

- working with voxel-wise object labels and predictions,
- generating point annotations from voxel predictions,
- and computing object-level performance as precision/recall

"""

from flypylib import fplutils, fplsynapses, fplnetwork
from diced import DicedStore, DicedException
from libdvid import DVIDNodeService, ConnectionMethod, DVIDException
import numpy as np
import h5py
from scipy import ndimage
import pulp
import math
import multiprocessing
from collections import namedtuple
import os, sys, pickle, time

PR_Result = namedtuple(
    'PR_Result', 'num_tp tot_pred tot_gt pp rr match')
szyx = namedtuple(
    'szyx', 'size z y x')

def gen_batches(train_data, context_sz, batch_sz, is_mask=False):
    """generator function that yields training batches

    extract training patches and labels for training with keras
    fit_generator

    Args:
        train_data (tuple of tuple of str): for each inner tuple, first string gives filename for training image hdf5 file, and second string gives filename prefix for labels and mask hdf5 files
        context_sz (tuple of int): tuple specifying size of training patches
        batch_sz   (int): number of examples in batch to yield

    """

    n_per_class = int(round(batch_sz/2))
    context_rr  = tuple(int(round(cc/2)) for cc in context_sz)

    ims  = []
    lls  = []
    mms  = []
    locs = []
    for tr in train_data:
        ims.append(h5py.File(tr[0],'r')['/main'][:])
        lls.append(h5py.File('%slabels.h5' % tr[1], 'r')['/main'][:])
        mms.append(h5py.File('%smask.h5'   % tr[1], 'r')['/main'][:])

        mms[-1][:context_rr[0],:,:] = 0
        mms[-1][:,:context_rr[1],:] = 0
        mms[-1][:,:,:context_rr[2]] = 0
        mms[-1][-context_rr[0]:,:,:] = 0
        mms[-1][:,-context_rr[1]:,:] = 0
        mms[-1][:,:,-context_rr[2]:] = 0

        locs_iter = [None, None]
        for cc in range(2):
            locs_iter[cc] = ( (lls[-1]==cc) & (mms[-1]==1) ).nonzero()
        locs.append(locs_iter)

        if (is_mask):
            # set the label of the ignored regions to 2
            idx = (mms[-1] == 0).nonzero()
            lls[-1][idx] = 2

    train_idx   = 0
    n_train     = len(train_data)
    data        = np.zeros(
        (batch_sz, context_sz[0], context_sz[1], context_sz[2], 1),
        dtype='float32')
    if (is_mask):
        labels = np.zeros((batch_sz, 6, 6, 6, 1), dtype='uint8')
    else:
        labels = np.zeros( (batch_sz, 1, 1, 1, 1), dtype='uint8' )

    while True:
        im = ims[train_idx]
        ll = lls[train_idx]
        mm = mms[train_idx]

        for cc in range(2):
            n_possible = len(locs[train_idx][cc][0])
            if n_possible == 0: # just use last iterations
                continue
            locs_idx   = np.random.choice(n_possible,
                                          n_per_class, True)

            for ii in range(n_per_class):
                locs_idx_ii = locs_idx[ii]
                xx_ii       = locs[train_idx][cc][0][locs_idx_ii]
                yy_ii       = locs[train_idx][cc][1][locs_idx_ii]
                zz_ii       = locs[train_idx][cc][2][locs_idx_ii]

                example_idx = ii*2+cc
                data[example_idx,:,:,:,0] = im[
                    xx_ii-context_rr[0]:xx_ii+context_rr[0],
                    yy_ii-context_rr[1]:yy_ii+context_rr[1],
                    zz_ii-context_rr[2]:zz_ii+context_rr[2]]
                if (is_mask):
                    labels[example_idx, :, :, :, 0] = ll[
                        xx_ii-3:xx_ii+3,yy_ii-3:yy_ii+3,zz_ii-3:zz_ii+3]
                else:
                    labels[example_idx,0]   = ll[xx_ii,yy_ii,zz_ii]

        # data augmentation
        aug_rot = np.floor(4*np.random.rand(batch_sz))
        aug_ref = np.floor(2*np.random.rand(batch_sz))
        aug_fpz = np.floor(2*np.random.rand(batch_sz))
        for ii in range(batch_sz):
            if(aug_rot[ii]):
                data[ii,:,:,:,0] = np.rot90(
                    data[ii,:,:,:,0], aug_rot[ii], (1,2) )
                if (is_mask):
                    labels[ii,:,:,:,0] = np.rot90(labels[ii,:,:,:,0], aug_rot[ii], (1,2))
            if(aug_ref[ii]):
                data[ii,:,:,:,0] = np.flip(
                    data[ii,:,:,:,0],2)
                if (is_mask):
                    labels[ii,:,:,:,0] = np.flip(labels[ii,:,:,:,0], 2)
            if(aug_fpz[ii]):
                data[ii,:,:,:,0] = np.flip(
                    data[ii,:,:,:,0],0)
                if (is_mask):
                    labels[ii,:,:,:,0] = np.flip(labels[ii,:,:,:,0], 0)

        yield data, labels
        train_idx = (train_idx + 1) % n_train

def voxel2obj(pred, obj_min_dist, smoothing_sigma,
              volume_offset=(0,0,0), buffer_sz=0, thd=0):
    """convert voxel-wise predictions to object predictions

    apply smoothing and non-maxima suppression to dense voxel-wise
    predictions to generate point-wise object predictions

    Args:
        pred            (array): 3D array of voxel-wise predictions
        obj_min_dist    (float): minimum allowable distance between predictions, for non-maxima suppression
        smoothing_sigma (float): sigma to use for Gaussian smoothing
        volume_offset   (tuple, optional): if given, shift prediction locations by (x,y,z) tuple

    Returns:
        array: NxM boolean array indicating matches between prediction and groundtruth objects

    """

    buffer_sz = fplutils.to3d(buffer_sz)

    if(isinstance(pred, str)):
        pred = h5py.File(pred,'r')
        pred = pred['/main'][:]

    pred_sz = pred.shape
    pred    = np.pad(pred, obj_min_dist, 'constant')

    pred = ndimage.filters.gaussian_filter(
        pred, smoothing_sigma, truncate=2.0)

    pred[:obj_min_dist,:,:] = 0
    pred[:,:obj_min_dist,:] = 0
    pred[:,:,:obj_min_dist] = 0
    pred[-obj_min_dist:,:,:] = 0
    pred[:,-obj_min_dist:,:] = 0
    pred[:,:,-obj_min_dist:] = 0

    thresh = np.maximum(np.percentile(pred,97), thd)
    locs   = (pred > thresh).nonzero()
    inds   = np.ravel_multi_index(locs, pred.shape)

    pred_flat       = pred.view()
    pred_flat.shape = (-1)

    dist_flt = np.logical_not(fplutils.set_filter(obj_min_dist))
    is_valid = np.ones( pred.shape, dtype='bool' )
    is_valid_flat = is_valid.view()
    is_valid_flat.shape = (-1)

    obj_pred = []

    while(inds.size > 0):
        vals    = pred_flat[inds]
        max_ind = np.argmax(vals)
        max_val = vals[max_ind]
        if(max_val <= 0):
            break

        zz,yy,xx = np.unravel_index(inds[max_ind], pred.shape)
        obj_pred.append([xx,yy,zz,max_val])

        is_valid[zz-obj_min_dist:zz+obj_min_dist+1,
                 yy-obj_min_dist:yy+obj_min_dist+1,
                 xx-obj_min_dist:xx+obj_min_dist+1] &= dist_flt

        inds = np.delete(inds, np.logical_not(
            is_valid_flat[inds]).nonzero())

    if obj_pred:
        obj_pred = np.asarray(obj_pred)
    else:
        obj_pred = np.zeros( (0,4) )
    obj_pred[:,:3] -= obj_min_dist

    min_bound = np.asarray([[
        buffer_sz[0],buffer_sz[1],buffer_sz[2],-np.inf ]])
    obj_pred  = np.delete(obj_pred,
                          np.any(obj_pred <  min_bound, 1).nonzero(),
                          axis=0)
    max_bound = np.asarray([[
        pred_sz[0]-buffer_sz[0],
        pred_sz[1]-buffer_sz[1],
        pred_sz[2]-buffer_sz[2], np.inf ]])
    obj_pred  = np.delete(obj_pred,
                          np.any(obj_pred >= max_bound, 1).nonzero(),
                          axis=0)

    obj_shift = np.array([volume_offset + (0,)])
    obj_pred  += obj_shift

    obj_out = { 'locs': obj_pred[:,:3],
                'conf': obj_pred[:,3] }
    return obj_out

def obj_match(dists, allow_mult=False):
    """compute object matching

    sets up and solves an integer program for matching predicted
    object locations to ground-truth object locations

    Args:
        dists (array): NxM array of distances between N prediction locations and M groundtruth locations, after subtracting distance threshold for counting prediction/groundtruth pair as a match (ie negative values indicate possible matches)

    Returns:
        array: NxM boolean array indicating matches between prediction and groundtruth objects

    """

    match = pulp.LpProblem('obj match', pulp.LpMinimize)

    n_pred, n_gt = dists.shape

    use_var = (dists < 0)

    match_names  = []
    match_costs = {}
    match_const_pred = [ list() for ii in range(n_pred)]
    match_const_gt   = [ list() for ii in range(n_gt)]

    for ii in range(n_pred):
        for jj in range(n_gt):
            if use_var[ii,jj]:
                match_names.append('x_%d_%d' % (ii, jj))
                match_costs[match_names[-1]] = dists[ii,jj]
                match_const_pred[ii].append(match_names[-1])
                match_const_gt[  jj].append(match_names[-1])

    match_vars = pulp.LpVariable.dicts(
        'var', match_names, 0, 1, pulp.LpInteger)
    match += pulp.lpSum([match_costs[ii]*match_vars[ii] for
                         ii in match_names]), 'obj match cost'

    if not allow_mult:
        for ii in range(n_pred):
            if match_const_pred[ii]:
                match += pulp.lpSum(match_vars[jj] for
                                    jj in match_const_pred[ii]
                                    ) <= 1, 'pred %d' % ii

    for ii in range(n_gt):
        if match_const_gt[ii]:
            match += pulp.lpSum(match_vars[jj] for
                                jj in match_const_gt[ii]
                                ) <= 1, 'gt %d' % ii

    match.solve()

    obj_matches = np.zeros( (n_pred,n_gt), dtype='bool')

    for ii in range(n_pred):
        for jj in range(n_gt):
            if use_var[ii,jj]:
                obj_matches[ii,jj] = match_vars[
                    'x_%d_%d' % (ii, jj)].varValue

    return obj_matches

def obj_pr(predict_locs, groundtruth_locs, dist_thresh,
           predict_lbls=None, groundtruth_lbls=None,
           allow_mult=False):
    """compute precision/recall

    given predicted and groundtruth object locations, computes
    precision-recall

    Args:
        predict_locs     (array): Nx3 numpy array containing prediction locations
        groundtruth_locs (array): Mx3 numpy array containing groundtruth locations
        dist_thresh      (float): distance threshold for counting a prediction and groundtruth location pair to be a match

    Returns:
        tuple: namedtuple with fields ``'pp'``: precision, ``'rr'``: recall, ``'num_tp'``: number of true positives, ``'tot_pred'``: number of predictions, and ``'tot_gt'``: number of groundtruth, and ``'match'``: as NxM numpy matrix indicating prediction/groundtruth matches

    """

    if( (predict_locs.shape[0] == 0) |
        (groundtruth_locs.shape[0] == 0) ): # check for empty cases
        tot_pred = predict_locs.shape[0]
        tot_gt   = groundtruth_locs.shape[0]

        pp = 0
        rr = 0
        if(tot_pred == 0):
            pp = 1
        if(tot_gt   == 0):
            rr = 1

        return PR_Result(num_tp=0, tot_pred=tot_pred, tot_gt=tot_gt,
                         pp=pp, rr=rr, match=None)

    pred     = predict_locs.reshape(     (-1, 1,3) )
    gt       = groundtruth_locs.reshape( ( 1,-1,3) )

    dists    = np.sqrt( ((pred-gt)**2).sum(axis=2) )
    dists   -= dist_thresh

    if predict_lbls is not None:
        lbl_constraint = (
            predict_lbls.reshape( (-1,1) ) !=
            groundtruth_lbls.reshape( (1,-1) ) ).astype('float32')
        dists += (dist_thresh+1.) * lbl_constraint

    match    = obj_match(dists, allow_mult=allow_mult)

    num_tp   = match.sum()
    pd_mult  = np.maximum(match.sum(axis=1) - 1,0).sum()
    tot_pred = match.shape[0] + pd_mult
    result   = PR_Result(
        num_tp=num_tp, tot_pred=tot_pred, tot_gt=match.shape[1],
        pp=num_tp/match.shape[0], rr=num_tp/match.shape[1],
        match=match)

    return result

def obj_pr_curve(predict, groundtruth, dist_thresh, thresholds,
                 predict_lbls=None, groundtruth_lbls=None,
                 allow_mult=False):
    """compute precision/recall curve

    given predicted and groundtruth object locations, computes
    precision-recall curve

    Args:
        predict     (dict or str): predictions, with locations in Nx3 numpy array as predict['locs'], and confidence values in N numpy array as predict['conf']; if string, will load predictions from json file specified by string
        groundtruth (dict): groundtruth, with locations in Mx3 numpy array as groundtruth['locs']; if string, will load predictions from json file specified by string
        dist_thresh (float): distance threshold for counting a prediction and groundtruth location pair to be a match
        thresholds  (array): T numpy array of thresholds to apply to confidence values, at which precision/recall will be computed

    Returns:
        tuple: namedtuple with fields ``'pp'``: precision, ``'rr'``: recall, ``'num_tp'``: number of true positives, ``'tot_pred'``: number of predictions, ``'tot_gt'``: number of groundtruth, with each value as T numpy array

    """

    if(isinstance(predict,str)):
        predict = obj_load_from_json(predict)
    if(isinstance(groundtruth,str)):
        groundtruth = obj_load_from_json(groundtruth)

    predict_locs     = predict['locs']
    predict_conf     = predict['conf']
    groundtruth_locs = groundtruth['locs']

    n_thd    = thresholds.size
    num_tp   = np.zeros( (n_thd,) )
    tot_pred = np.zeros( (n_thd,) )
    tot_gt   = np.zeros( (n_thd,) )
    pp       = np.zeros( (n_thd,) )
    rr       = np.zeros( (n_thd,) )

    predict_lbls_iter = None

    for ii in range(thresholds.size):
        predict_idx = (predict_conf >= thresholds[ii])
        predict_locs_iter = predict_locs[predict_idx,:]
        if predict_lbls is not None:
            predict_lbls_iter = predict_lbls[predict_idx]

        mm = obj_pr(predict_locs_iter, groundtruth_locs,
                    dist_thresh,
                    predict_lbls_iter, groundtruth_lbls,
                    allow_mult=allow_mult)
        num_tp[  ii] = mm.num_tp
        tot_pred[ii] = mm.tot_pred
        tot_gt[  ii] = mm.tot_gt
        pp[      ii] = mm.pp
        rr[      ii] = mm.rr

    result = PR_Result(num_tp=num_tp, tot_pred=tot_pred, tot_gt=tot_gt,
                       pp=pp, rr=rr, match=None)
    return result


def aggregate_pr(results):
    dim      = results[0].num_tp.shape
    num_tp   = np.zeros(dim)
    tot_pred = np.zeros(dim)
    tot_gt   = np.zeros(dim)

    for rr in results:
        num_tp   += rr.num_tp
        tot_pred += rr.tot_pred
        tot_gt   += rr.tot_gt

    pp = num_tp / (tot_pred + 10e-8)
    rr = num_tp / (tot_gt + 10e-8)

    result = PR_Result(num_tp=num_tp, tot_pred=tot_pred, tot_gt=tot_gt,
                       pp=pp, rr=rr, match=None)
    return result

def _get_labels(seg, tt):
    tt_ind = tt['locs'].astype(int)
    return seg[tt_ind[:,2],tt_ind[:,1],tt_ind[:,0]]

def _evaluate_substacks_worker(pred, obj_min_dist, smoothing_sigma,
                               volume_offset, buffer_sz,
                               json_fn, thds, seg_fn,
                               allow_mult, qq):
    out  = voxel2obj(pred, obj_min_dist, smoothing_sigma,
                     volume_offset, buffer_sz)
    gt   = fplsynapses.load_from_json(json_fn, pred.shape, buffer_sz)

    lbls_pd = None
    lbls_gt = None
    if seg_fn is not None:
        seg = h5py.File(seg_fn, 'r')['/main'][:]
        lbls_pd = _get_labels(seg, out)
        lbls_gt = _get_labels(seg, gt)

    rr   = obj_pr_curve(out, gt, obj_min_dist, thds,
                        lbls_pd, lbls_gt,
                        allow_mult=allow_mult)

    qq.put({json_fn: rr})

def evaluate_substacks(network, substacks, thds,
                       obj_min_dist=27, smoothing_sigma=5,
                       volume_offset=(0,0,0), buffer_sz=5,
                       allow_mult=False):
    qq  = multiprocessing.Queue()
    pps = []
    oo  = {}
    for ss in substacks:
        pred = network.infer(ss[0])
        if len(ss)<3:
            seg_fn = None
        else:
            seg_fn = ss[2]
        pp = multiprocessing.Process(
            target=_evaluate_substacks_worker,
            args=(pred, obj_min_dist, smoothing_sigma,
                  volume_offset, buffer_sz, ss[1], thds, seg_fn,
                  allow_mult, qq))
        pps.append(pp)
        pp.start()

    for pp in pps:
        oo.update(qq.get())

    results = []
    for ss in substacks:
        results.append(oo[ss[1]])

    return aggregate_pr(results), results

def get_out_sz(in_sz):
    """compute the output size of a unet-style net

    Args:
        in_sz: size of an input subvolume
    """
    bottleneck_sz = int(math.floor(math.floor((in_sz - 2) / 2) - 2) / 2)
    out_sz = (bottleneck_sz * 2 - 2) * 2 - 2
    return out_sz

def gen_volume(train_data, context_sz, batch_sz, ratio):
    """generator function that yields training batches

    extract training patches and labels for training with keras
    fit_generator

    Args:
        train_data (tuple of tuple of str): for each inner tuple, first string gives filename for training image hdf5 file, and second string gives filename prefix for labels and mask hdf5 files
        context_sz (tuple of int): tuple specifying size of training patches
        batch_sz   (int): number of examples in batch to yield
        ratio      (float): ratio of negative samples

    """
    #n_per_class = round(batch_sz/2)
    context_rr  = tuple(round(cc/2) for cc in context_sz)

    ims  = []
    lls  = []
    mms  = []
    locs = []
    for tr in train_data:
        ims.append(h5py.File(tr[0],'r')['/main'][:])
        lls.append(h5py.File('%slabels.h5' % tr[1], 'r')['/main'][:])
        mms.append(h5py.File('%smask.h5'   % tr[1], 'r')['/main'][:])

        mms[-1][:context_rr[0],:,:] = 0
        mms[-1][:,:context_rr[1],:] = 0
        mms[-1][:,:,:context_rr[2]] = 0
        mms[-1][-context_rr[0]:,:,:] = 0
        mms[-1][:,-context_rr[1]:,:] = 0
        mms[-1][:,:,-context_rr[2]:] = 0

        locs_iter = [None, None]
        for cc in range(2):
            locs_iter[cc] = ( (lls[-1]==cc) & (mms[-1]==1) ).nonzero()
        locs.append(locs_iter)

        '''
        locs_iter = ((lls[-1] == 1) & mms[-1] == 1).nonzero()
        locs.append(locs_iter)
        '''

        # set the label of the ignored regions to 2
        idx = (mms[-1] == 0).nonzero()
        lls[-1][idx] = 2

    train_idx   = 0
    n_train     = len(train_data)
    data        = np.zeros(
        (batch_sz, context_sz[0], context_sz[1], context_sz[2], 1),
        dtype='float32')

    #out_sz = tuple(get_out_sz(cc) for cc in context_sz)
    out_sz = (6, 6, 6)
    out_rr = tuple(round(cc/2) for cc in out_sz)
    labels = np.zeros((batch_sz, out_sz[0], out_sz[1], out_sz[2], 1), dtype='uint8')
    #masks = np.zeros((batch_sz, out_sz[0], out_sz[1], out_sz[2], 1), dtype='uint8')

    while True:

        # random sampling
        example_idx = 0
        '''
        im = ims[train_idx]
        ll = lls[train_idx]
        mm = mms[train_idx]
        label_idx = 1 # positive samples
        if np.random.uniform(0,1) < ratio:
            label_idx = 0 # negative samples

        n_possible = len(locs[train_idx][label_idx][0])
        if (n_possible == 0):
            label_idx = 0
            n_possible = len(locs[train_idx][label_idx][0])
        locs_idx = np.random.choice(n_possible, batch_sz, True)
        '''
        for ii in range(batch_sz):
            im = ims[train_idx]
            ll = lls[train_idx]
            mm = mms[train_idx]

            label_idx = 1 # positive samples
            if np.random.uniform(0,1) < ratio:
                label_idx = 0 # negative samples

            n_possible = len(locs[train_idx][label_idx][0])
            if (n_possible == 0):
                label_idx = 0
                n_possible = len(locs[train_idx][label_idx][0])
            locs_idx = np.random.choice(n_possible, batch_sz, True)

            locs_idx_ii = locs_idx[ii]
            xx_ii = locs[train_idx][label_idx][0][locs_idx_ii]
            yy_ii = locs[train_idx][label_idx][1][locs_idx_ii]
            zz_ii = locs[train_idx][label_idx][2][locs_idx_ii]
            data[example_idx,:,:,:,0] = im[
                xx_ii-context_rr[0]:xx_ii+context_rr[0],
                yy_ii-context_rr[1]:yy_ii+context_rr[1],
                zz_ii-context_rr[2]:zz_ii+context_rr[2]]
            labels[example_idx, :, :, :, 0] = ll[
                xx_ii-out_rr[0]:xx_ii+out_rr[0],
                yy_ii-out_rr[1]:yy_ii+out_rr[1],
                zz_ii-out_rr[2]:zz_ii+out_rr[2]]
            '''
            masks[example_idx, :, :, :, 0] = mm[
                xx_ii-out_rr[0]:xx_ii+out_rr[0],
                yy_ii-out_rr[1]:yy_ii+out_rr[1],
                zz_ii-out_rr[2]:zz_ii+out_rr[2]]
            '''
            example_idx = example_idx + 1
            train_idx = (train_idx + 1) % n_train

        # data augmentation
        aug_rot = np.floor(4*np.random.rand(batch_sz))
        aug_ref = np.floor(2*np.random.rand(batch_sz))
        aug_fpz = np.floor(2*np.random.rand(batch_sz))
        for ii in range(batch_sz):
            if(aug_rot[ii]):
                data[ii,:,:,:,0] = np.rot90(
                    data[ii,:,:,:,0], aug_rot[ii], (1,2) )
                labels[ii,:,:,:,0] = np.rot90(
                    labels[ii,:,:,:,0], aug_rot[ii], (1,2))
            if(aug_ref[ii]):
                data[ii,:,:,:,0] = np.flip(
                    data[ii,:,:,:,0],2)
                labels[ii,:,:,:,0] = np.flip(
                    labels[ii,:,:,:,0], 2)
            if(aug_fpz[ii]):
                data[ii,:,:,:,0] = np.flip(
                    data[ii,:,:,:,0],0)
                labels[ii,:,:,:,0] = np.flip(
                    labels[ii,:,:,:,0], 0)

        yield data, labels
        #train_idx = (train_idx + 1) % n_train

def gen_volume2(train_data, context_sz, batch_sz, ratio,
                noise_aug=[0,0]):
    """generator function that yields training batches

    extract training patches and labels for training with keras
    fit_generator

    Args:
        train_data (tuple of tuple of str): for each inner tuple, first string gives filename for training image hdf5 file, and second string gives filename prefix for labels and mask hdf5 files
        context_sz (tuple of int): tuple specifying size of training patches
        batch_sz   (int): number of examples in batch to yield
        ratio      (float): ratio of negative samples

    """
    context_rr  = tuple(round(cc/2) for cc in context_sz)

    ims  = []
    lls  = []
    mms  = []
    locs = [[[], [], [], []],
            [[], [], [], []]]
    train_idx = 0

    use_weighted_sampling = False
    if len(train_data[0])>2:
        use_weighted_sampling = True
        for cc in range(2):
            locs[cc].append([])

    for tr in train_data:
        ims.append(h5py.File(tr[0],'r')['/main'][:])
        lls.append(h5py.File('%slabels.h5' % tr[1], 'r')['/main'][:])
        mms.append(h5py.File('%smask.h5'   % tr[1], 'r')['/main'][:])

        mms[-1][:context_rr[0],:,:] = 0
        mms[-1][:,:context_rr[1],:] = 0
        mms[-1][:,:,:context_rr[2]] = 0
        mms[-1][-context_rr[0]:,:,:] = 0
        mms[-1][:,-context_rr[1]:,:] = 0
        mms[-1][:,:,-context_rr[2]:] = 0

        if use_weighted_sampling:
            ww = h5py.File(tr[2],'r')['/main'][:]

        for cc in range(2):
            if use_weighted_sampling:
                locs_iter = ( (lls[-1]==cc) & (mms[-1]==1) &
                              (ww > 0)).nonzero()
                ww_locs_iter = ww[locs_iter]
            else:
                locs_iter = ( (lls[-1]==cc) & (mms[-1]==1) ).nonzero()
            for locs_iter_ii in locs_iter:
                locs_iter_ii = locs_iter_ii.astype('int16')
            locs_ss   = train_idx * np.ones(locs_iter[0].shape,
                                            dtype='int16')
            locs_iter = (locs_ss,) + locs_iter
            if use_weighted_sampling:
                locs_iter = locs_iter + (ww_locs_iter,)

            for ll in range(len(locs_iter)):
                locs[cc][ll].append(locs_iter[ll])

        # set the label of the ignored regions to 2
        idx = (mms[-1] == 0).nonzero()
        lls[-1][idx] = 2
        train_idx += 1

    for cc in range(2):
        for ll in range(len(locs[cc])):
            locs[cc][ll] = np.concatenate(locs[cc][ll])
    if use_weighted_sampling:
        for cc in range(2):
            locs[cc][4] /= np.sum(locs[cc][4].astype('float32'))

    train_idx   = 0
    n_train     = len(train_data)
    data        = np.zeros(
        (batch_sz, context_sz[0], context_sz[1], context_sz[2], 1),
        dtype='float32')

    #out_sz = tuple(get_out_sz(cc) for cc in context_sz)
    out_sz = (6, 6, 6)
    out_rr = tuple(round(cc/2) for cc in out_sz)
    labels = np.zeros((batch_sz, out_sz[0], out_sz[1], out_sz[2], 1), dtype='uint8')


    n_tot_neg = len(locs[0][0])
    n_tot_pos = len(locs[1][0])
    outer_batches  = 100
    outer_batch_sz = outer_batches * batch_sz
    n_neg          = round(ratio*outer_batch_sz)
    n_pos          = outer_batch_sz - n_neg
    while True:
        if use_weighted_sampling:
            neg_idx = np.random.choice(n_tot_neg, n_neg, True,
                                       locs[0][4])
            pos_idx = np.random.choice(n_tot_pos, n_pos, True,
                                       locs[1][4])
        else:
            neg_idx = np.random.choice(n_tot_neg, n_neg, True)
            pos_idx = np.random.choice(n_tot_pos, n_pos, True)
        all_idx = np.random.permutation(outer_batch_sz)

        sample_idx = 0
        for b_idx in range(outer_batches):
            example_idx = 0

            for ii in range(batch_sz):
                locs_idx = all_idx[sample_idx]
                if(locs_idx < n_neg):
                    label_idx = 0
                    locs_idx  = neg_idx[locs_idx]
                else:
                    label_idx = 1
                    locs_idx -= n_neg
                    locs_idx  = pos_idx[locs_idx]

                train_idx = locs[label_idx][0][locs_idx]
                im = ims[train_idx]
                ll = lls[train_idx]
                mm = mms[train_idx]

                xx_ii = locs[label_idx][1][locs_idx]
                yy_ii = locs[label_idx][2][locs_idx]
                zz_ii = locs[label_idx][3][locs_idx]
                data[example_idx,:,:,:,0] = (
                    (noise_aug[1]+1)*np.random.randn() *
                    im[
                        xx_ii-context_rr[0]:xx_ii+context_rr[0],
                        yy_ii-context_rr[1]:yy_ii+context_rr[1],
                        zz_ii-context_rr[2]:zz_ii+context_rr[2]]
                    ) + noise_aug[0]*np.random.randn()
                labels[example_idx, :, :, :, 0] = ll[
                    xx_ii-out_rr[0]:xx_ii+out_rr[0],
                    yy_ii-out_rr[1]:yy_ii+out_rr[1],
                    zz_ii-out_rr[2]:zz_ii+out_rr[2]]

                example_idx = example_idx + 1
                sample_idx  += 1

            # data augmentation
            aug_rot = np.floor(4*np.random.rand(batch_sz))
            aug_ref = np.floor(2*np.random.rand(batch_sz))
            aug_fpz = np.floor(2*np.random.rand(batch_sz))
            for ii in range(batch_sz):
                if(aug_rot[ii]):
                    data[ii,:,:,:,0] = np.rot90(
                        data[ii,:,:,:,0], aug_rot[ii], (1,2) )
                    labels[ii,:,:,:,0] = np.rot90(
                        labels[ii,:,:,:,0], aug_rot[ii], (1,2))
                if(aug_ref[ii]):
                    data[ii,:,:,:,0] = np.fliplr(
                        data[ii,:,:,:,0])
                    labels[ii,:,:,:,0] = np.fliplr(
                        labels[ii,:,:,:,0])
                if(aug_fpz[ii]):
                    data[ii,:,:,:,0] = np.flipud(
                        data[ii,:,:,:,0])
                    labels[ii,:,:,:,0] = np.flipud(
                        labels[ii,:,:,:,0])

            yield data, labels
            #train_idx = (train_idx + 1) % n_train

def write_sampling_weights(train_data, network, fn_prefix,
                           l0_thresh, l1_thresh):
    train_data_aug = []
    idx = 0
    for tr in train_data:
        voxel_loss = network.voxel_loss(
            tr[0], tr[1], l0_thresh, l1_thresh)
        ww_fn = '%s%02d.h5' % (fn_prefix, idx)
        hh = h5py.File(ww_fn,'w')
        hh.create_dataset('/main', voxel_loss.shape,
                          dtype='float32', compression='gzip')
        hh['/main'][:] = voxel_loss
        hh.close()
        train_data_aug.append(list(tr) + [ww_fn,])
        idx += 1
    return train_data_aug

def full_roi_inference(data_source, dvid_uuid, dvid_roi,
                       network, thd, working_dir,
                       image_normalize,
                       obj_min_dist=27, smoothing_sigma=5,
                       buffer_sz=35, partition_size=16,
                       local_cache_dir=None,
                       roi_force_file=False,
                       instance_name='grayscale'):
    """ Given a trained network and a data source, return predictions
        within the provided region. This also caches results to disk so
        subsequent calls might be faster.

        Parameters:
            data_source     - string containing an URL that might be a DVID server
                              or a GoogleStore bucket (gs:// prefix)
            dvid_uuid       - String containing the UUID of the requested volume
            dvid_roi        - If data_source is DVID, used to keep track of ROIs already
                              processed by prior invocations of this function. Otherwise
                              this is a filename used to load ROIs from a textfile.
            network         - Pass in an already-trained trained fplnetwork object
            thd             - Threshold used by voxel2obj in postprocessing
            working_dir     - A temporary directory needed to store/cache some computation
            image_normalize - An array of parameters used in normalization.
                              The first element is the mean value,
                              The second is the unit standard deviation,
                              and an optional third value can be provided, if this subunit has a different
                              mean, to provide an interpolation factor between that mean and the global mean
            obj_min_dist    - Minimum acceptable distance for predictions.
            smoothing_sigma - Value used for smoothing function
            buffer_sz       - How many units (in each dimension) to buffer around the image to
                              ensure there is data to sample from
            partition_size  - If data_source is DVID, used to request the data partition containing
                              the location and size of ROIs. Otherwise ignored.
            local_cache_dir - An optional parameter that can identify a directory used to cache
                              the remote data to the local system. Can speed things up, but can also
                              use a lot of disk. Be careful when using this.
            roi_force_file  - Boolean, forces load of ROIs from file even with a DVID
                              data_source. Useful if your DVID is readonly.
            instance_name   - If you're doing inference on a data instance not named grayscale,
                              you may specify a different name here
"""
    try:
        os.makedirs(working_dir)
    except OSError:
        if not os.path.isdir(working_dir):
            raise
    norm_dir = '%s/norm' % working_dir
    try:
        os.makedirs(norm_dir)
    except OSError:
        if not os.path.isdir(norm_dir):
            raise

    if data_source[:5] == 'gs://' or roi_force_file: # Always when using a DICED google bucket
        has_dvid_roi = False
        roi = roi_from_txt(dvid_roi)
    else:
        has_dvid_roi = True
        dvid_node = DVIDNodeService(data_source, dvid_uuid,
                                    'fpl','fpl')
        roi = dvid_node.get_roi_partition(dvid_roi, partition_size)

    locs = []
    conf = []

    # skip previously processed substacks, prepare pool worker args
    fri_get_image_args = []
    num_processed = 0
    for rr in roi[0]:
        ff = fri_filename(working_dir, rr)
        if os.path.isfile(ff):
            with open(ff, 'rb') as f_in:
                obj = pickle.load(f_in)
            locs.append(obj['locs'])
            conf.append(obj['conf'])
            num_processed += 1
            continue
        rr2 = szyx(rr.size,rr.z,rr.y,rr.x)
        fri_get_image_args.append(
            [rr2, data_source, dvid_uuid, image_normalize, buffer_sz,
             local_cache_dir, norm_dir, instance_name])
    print('already processed: %d' % num_processed)
    print('to process: %d' % len(fri_get_image_args))

    n_done  = 0
    max_at_once = 1000
    fri_im_args_batch = []
    for ii in range(0, len(fri_get_image_args), max_at_once):
        fri_im_args_batch.append(
            fri_get_image_args[ii:min(ii+max_at_once,
                                      len(fri_get_image_args))] )

    for bb in fri_im_args_batch:
        qq_pre  = multiprocessing.Queue()
        qq_post = multiprocessing.Queue()

        pp_pre  = multiprocessing.Process(
            target=fri_get_image_generator,
            args=(bb, qq_pre))
        pp_pre.start()

        for ii in range(len(bb)):
            ss = qq_pre.get()
            pred = network.infer(ss[0])
            pp_obj = multiprocessing.Process(
                target=fri_postprocess,
                args=(pred, working_dir, obj_min_dist, smoothing_sigma,
                      ss[1], buffer_sz, thd, qq_post))
            pp_obj.start()
            n_done += 1
            sys.stdout.write('\r%d' % n_done)
            sys.stdout.flush()

        for ii in range(len(bb)):
            ff = qq_post.get()
            with open(ff, 'rb') as f_in:
                obj = pickle.load(f_in)
                locs.append(obj['locs'])
                conf.append(obj['conf'])

    locs = np.concatenate(locs)
    conf = np.concatenate(conf)

    if has_dvid_roi:
        # filter by roi
        pts    = np.fliplr(locs).astype('int').tolist()
        in_roi = dvid_node.roi_ptquery(dvid_roi, pts)
        in_roi = np.array(in_roi)
        locs   = locs[in_roi]
        conf   = conf[in_roi]

    obj = { 'locs': locs,
            'conf': conf }
    with open('%s/all.p' % working_dir, 'wb') as f_out:
        pickle.dump(obj, f_out)
    return obj

def fri_get_image_generator(get_image_args, qq):
    if not get_image_args:
        return

    dvid_server     = get_image_args[0][1]
    dvid_uuid       = get_image_args[0][2]
    instance_name   = get_image_args[0][7]

    if dvid_server[:5] == 'gs://': # DICED google bucket
        using_diced = True
        if dvid_server.find(',')>=0:
            store = DicedStore(*dvid_server.split(','))
        else:
            store = DicedStore(dvid_server)

        repo = store.open_repo(uuid=dvid_uuid)
        dvid_node = repo.get_array(instance_name)
    else:
        using_diced = False
        dvid_node = DVIDNodeService(dvid_server, dvid_uuid,
                                    'fpl','fpl')
    for gg in get_image_args:
        while qq.qsize() >= 2:
            time.sleep(10)
        qq.put(fri_get_image(gg, dvid_node, using_diced, instance_name))

    if using_diced:
        store._shutdown_store()

def fri_get_image(substack_info, dvid_node, using_diced, instance_name='grayscale'):
    substack        = substack_info[0]
    image_normalize = substack_info[3]
    buffer_sz       = substack_info[4]
    local_cache_dir = substack_info[5]
    norm_dir        = substack_info[6]

    image_sz = substack.size + 2*buffer_sz
    image_offset = [substack.z - buffer_sz,
                    substack.y - buffer_sz,
                    substack.x - buffer_sz]

    if local_cache_dir is not None: # return if file exists
        cfn = fri_cachename(local_cache_dir, substack, buffer_sz)
        if os.path.isfile(cfn):
            print('reading from disk')
            image = h5py.File(cfn,'r')['/main'][:]
            return (image, substack)

    if using_diced:
        while True:
            try:
                image = dvid_node[
                    image_offset[0]:(image_offset[0]+image_sz),
                    image_offset[1]:(image_offset[1]+image_sz),
                    image_offset[2]:(image_offset[2]+image_sz)]
            except (DicedException,DVIDException) as e:
                print(e)
                time.sleep(150)
                continue
            break

    else:
        image = dvid_node.get_gray3D(instance_name,
                                     [image_sz, image_sz, image_sz],
                                     image_offset)

    im_raw_mn  = np.mean(image);
    im_raw_std = np.std( image);
    idx = ((image < 200) & (image > 1))
    if np.sum(idx)>0:
        im_flt_mn  = np.mean(image[idx])
        im_flt_std = np.std( image[idx])
    else:
        im_flt_mn  = image_normalize[0]
        im_flt_std = image_normalize[1]

    if len(image_normalize) < 3:
        global_frac = 1.
    else:
        global_frac = image_normalize[2]

    mn_use = global_frac * image_normalize[0] + (
        1-global_frac) * im_flt_mn

    image = (image.astype('float32') -
             mn_use) / image_normalize[1]

    norm_fn = '%s/%d_%d_%d_%d.txt' % (
        norm_dir, substack.size,
        substack.z, substack.y, substack.x)
    with open(norm_fn,'w') as f_out:
        f_out.write('%d,%d,%d,%d,%d,%g,%g,%g,%g,%g,%g,%g,%g\n' %
                    (substack.size, buffer_sz,
                     substack.z, substack.y, substack.x,
                     image_normalize[0], image_normalize[1],
                     global_frac, mn_use,
                     im_flt_mn, im_flt_std, im_raw_mn, im_raw_std))

    if local_cache_dir is not None: # write out image
        cfn = fri_cachename(local_cache_dir, substack, buffer_sz)
        hh = h5py.File(cfn,'w')
        hh.create_dataset('/main', image.shape,
                          dtype='float32', compression='gzip')
        hh['/main'][:] = image
        hh.close()

    return (image, substack)

def fri_postprocess(pred, working_dir, obj_min_dist, smoothing_sigma,
                    substack, buffer_sz, thd, qq):
    out = voxel2obj(pred, obj_min_dist, smoothing_sigma,
                    (substack.x - buffer_sz,
                     substack.y - buffer_sz,
                     substack.z - buffer_sz),
                    buffer_sz, thd)
    ff = fri_filename(working_dir, substack)
    with open(ff, 'wb') as f_out:
        pickle.dump(out, f_out)
    qq.put(ff)

def fri_filename(working_dir, substack):
    return '%s/%d_%d_%d_%d.p' % (working_dir, substack.size,
                                 substack.z, substack.y, substack.x)
def fri_cachename(cache_dir, substack, buffer_sz):
    return '%s/%d_%d_%d_%d_%d.h5' % (
        cache_dir, substack.size,
        substack.z, substack.y, substack.x, buffer_sz)

def gen_full_tab_roi(filename, store_name, repo_uuid,
                     crop=[None,None,None],
                     n_splits=1, step_size=512):
    store  = DicedStore(store_name)
    repo   = store.open_repo(uuid=repo_uuid)
    imdata = repo.get_array('grayscale')

    extents = imdata.get_extents()
    store._shutdown_store()

    for ii in range(len(extents)):
        if crop[ii] is None:
            crop[ii] = [extents[ii].start, extents[ii].stop]

    file_idx  = 0
    f_out     = open('%s_%02d.txt' % (filename, file_idx),'w')
    count_idx = 0

    max_count = int(np.ceil( (
        np.ceil( (crop[0][1]-crop[0][0])/float(step_size) ) *
        np.ceil( (crop[1][1]-crop[1][0])/float(step_size) ) *
        np.ceil( (crop[2][1]-crop[2][0])/float(step_size) ) ) /
                             n_splits ))

    for zz in range(crop[0][0],crop[0][1],step_size):
        for yy in range(crop[1][0],crop[1][1],step_size):
            for xx in range(crop[2][0],crop[2][1],step_size):
                f_out.write('%d,%d,%d,%d\n' %
                            (step_size,zz,yy,xx))

                count_idx += 1
                if count_idx==max_count:
                    count_idx=0
                    file_idx += 1
                    f_out.close()
                    f_out = open('%s_%02d.txt' %
                                 (filename, file_idx),'w')

def roi_from_txt(filename):
    with open(filename,'r') as f_in:
        substacks = f_in.read().splitlines()
    roi = []
    for ss in substacks:
        roi.append(szyx(*[int(nn) for nn in ss.split(',')]))
    return [roi,]
