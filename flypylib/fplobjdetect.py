"""functions for

- working with voxel-wise object labels and predictions,
- generating point annotations from voxel predictions,
- and computing object-level performance as precision/recall

"""

from flypylib import fplutils
import numpy as np
import h5py
from scipy import ndimage
import pulp
import math

def gen_batches(train_data, context_sz, batch_sz, is_mask=False):
    """generator function that yields training batches

    extract training patches and labels for training with keras
    fit_generator

    Args:
        train_data (tuple of tuple of str): for each inner tuple, first string gives filename for training image hdf5 file, and second string gives filename prefix for labels and mask hdf5 files
        context_sz (tuple of int): tuple specifying size of training patches
        batch_sz   (int): number of examples in batch to yield

    """

    n_per_class = round(batch_sz/2)
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

        example_idx = 0
        for cc in range(2):
            n_possible = len(locs[train_idx][cc][0])
            locs_idx   = np.random.choice(n_possible,
                                          n_per_class, True)

            for ii in range(n_per_class):
                locs_idx_ii = locs_idx[ii]
                xx_ii       = locs[train_idx][cc][0][locs_idx_ii]
                yy_ii       = locs[train_idx][cc][1][locs_idx_ii]
                zz_ii       = locs[train_idx][cc][2][locs_idx_ii]

                data[example_idx,:,:,:,0] = im[
                    xx_ii-context_rr[0]:xx_ii+context_rr[0],
                    yy_ii-context_rr[1]:yy_ii+context_rr[1],
                    zz_ii-context_rr[2]:zz_ii+context_rr[2]]
                if (is_mask):
                    labels[example_idx, :, :, :, 0] = ll[
                        xx_ii-3:xx_ii+3,yy_ii-3:yy_ii+3,zz_ii-3:zz_ii+3]
                else:
                    labels[example_idx,0]   = ll[xx_ii,yy_ii,zz_ii]

                example_idx = example_idx + 1

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
                    labels[ii,:,:,:,0] = np.flip([labels[ii,:,:,:,0]], 0)

        yield data, labels
        train_idx = (train_idx + 1) % n_train

def voxel2obj(pred, obj_min_dist, smoothing_sigma,
              volume_offset, buffer_sz):
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

    thresh = np.percentile(pred,97)
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

    obj_pred        = np.asarray(obj_pred)
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

    obj_out = { 'locs': obj_pred[:,:3],
                'conf': obj_pred[:,3] }
    return obj_out

def obj_match(dists):
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

def obj_pr(predict_locs, groundtruth_locs, dist_thresh):
    """compute precision/recall

    given predicted and groundtruth object locations, computes
    precision-recall

    Args:
        predict_locs     (array): Nx3 numpy array containing prediction locations
        groundtruth_locs (array): Mx3 numpy array containing groundtruth locations
        dist_thresh      (float): distance threshold for counting a prediction and groundtruth location pair to be a match

    Returns:
        dict: dictionary containing keys ``'pp'``: precision, ``'rr'``: recall, ``'num_tp'``: number of true positives, ``'tot_pred'``: number of predictions, and ``'tot_gt'``: number of groundtruth, and ``'match'``: as NxM numpy matrix indicating prediction/groundtruth matches

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

        result = { 'match'    : None,
                   'num_tp'   : 0,
                   'tot_pred' : tot_pred,
                   'tot_gt'   : tot_gt,
                   'pp'       : pp,
                   'rr'       : rr }
        return result

    pred     = predict_locs.reshape(     (-1, 1,3) )
    gt       = groundtruth_locs.reshape( ( 1,-1,3) )

    dists    = np.sqrt( ((pred-gt)**2).sum(axis=2) )
    dists   -= dist_thresh

    match    = obj_match(dists)

    result   = { 'match'    : match,
                 'num_tp'   : match.sum(),
                 'tot_pred' : match.shape[0],
                 'tot_gt'   : match.shape[1] }

    result['pp'] = result['num_tp'] / result['tot_pred']
    result['rr'] = result['num_tp'] / result['tot_gt']

    return result

def obj_pr_curve(predict, groundtruth, dist_thresh, thresholds):
    """compute precision/recall curve

    given predicted and groundtruth object locations, computes
    precision-recall curve

    Args:
        predict     (dict or str): predictions, with locations in Nx3 numpy array as predict['locs'], and confidence values in N numpy array as predict['conf']; if string, will load predictions from json file specified by string
        groundtruth (dict): groundtruth, with locations in Mx3 numpy array as groundtruth['locs']; if string, will load predictions from json file specified by string
        dist_thresh (float): distance threshold for counting a prediction and groundtruth location pair to be a match
        thresholds  (array): T numpy array of thresholds to apply to confidence values, at which precision/recall will be computed

    Returns:
        dict: dictionary containing keys ``'pp'``: precision, ``'rr'``: recall, ``'num_tp'``: number of true positives, ``'tot_pred'``: number of predictions, ``'tot_gt'``: number of groundtruth, with each value as T numpy array

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

    for ii in range(thresholds.size):
        predict_locs_iter = predict_locs[
            predict_conf >= thresholds[ii],:]
        mm = obj_pr(predict_locs_iter, groundtruth_locs,
                    dist_thresh)
        num_tp[  ii] = mm['num_tp']
        tot_pred[ii] = mm['tot_pred']
        tot_gt[  ii] = mm['tot_gt']
        pp[      ii] = mm['pp']
        rr[      ii] = mm['rr']

    result = { 'num_tp'   : num_tp,
               'tot_pred' : tot_pred,
               'tot_gt'   : tot_gt,
               'pp'       : pp,
               'rr'       : rr }
    return result


def aggregate_pr(results):
    dim   = results[0]['num_tp'].shape
    total = { 'num_tp'   : np.zeros(dim),
              'tot_pred' : np.zeros(dim),
              'tot_gt'   : np.zeros(dim) }

    for rr in results:
        total['num_tp']   += rr['num_tp']
        total['tot_pred'] += rr['tot_pred']
        total['tot_gt']   += rr['tot_gt']

    total['pp'] = total['num_tp'] / total['tot_pred']
    total['rr'] = total['num_tp'] / total['tot_gt']

    return total
