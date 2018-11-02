"""functions for working with synapse data, json, dvid """

from flypylib import fplutils
from flyem_syn_eval import eval
from libdvid import DVIDNodeService, ConnectionMethod, DVIDConnection
from libdvid._dvid_python import DVIDException
import numpy as np
import h5py
import json, os, time

def load_from_json(fn, vol_sz=None, buffer=None):
    """read synapse data from json file

    """

    if os.path.isfile(fn):
        with open(fn) as json_file:
            data = json.load(json_file)
    else:
        data = json.loads(fn)

    locs = []
    conf = []
    err  = []

    if((isinstance(data, dict) and
        'data' in data.keys())): # Raveler format

        for syn in data['data']:
            locs.append(syn['T-bar']['location'])
            conf.append(syn['T-bar']['confidence'])
    elif data is not None: # assume dvid format
        if len(data)==1 and isinstance(data[0],list):
            data = data[0]
        for syn in data:
            if syn['Kind'] != 'PreSyn':
                continue

            if 'conf' in syn['Prop']:
                cc = float(syn['Prop']['conf'])
            else:
                cc = 1.0

            if 'err' in syn['Prop']:
                err.append(float(syn['Prop']['err']))
            else:
                err.append(None)

            locs.append(syn['Pos'])
            conf.append(cc)

    locs  = np.asarray(locs)
    conf  = np.asarray(conf)
    err   = np.asarray(err)

    if locs.size > 0 and buffer is not None and buffer != 0:
        assert vol_sz is not None, \
            'to apply buffer, must also supply volume size'
        buffer = fplutils.to3d(buffer)
        vol_sz = fplutils.to3d(vol_sz)

        idx = (locs[:,0] < buffer[0]).nonzero()[0]
        idx = np.union1d(idx, (locs[:,1] < buffer[1]).nonzero()[0])
        idx = np.union1d(idx, (locs[:,2] < buffer[2]).nonzero()[0])
        idx = np.union1d(
            idx, (locs[:,0] >= vol_sz[0] - buffer[0]).nonzero()[0])
        idx = np.union1d(
            idx, (locs[:,1] >= vol_sz[1] - buffer[1]).nonzero()[0])
        idx = np.union1d(
            idx, (locs[:,2] >= vol_sz[2] - buffer[2]).nonzero()[0])

        locs = np.delete(locs, idx, axis=0)
        conf = np.delete(conf, idx, axis=0)

    tbars = { 'locs': locs, 'conf': conf, 'err': err }
    return tbars

def tbars_to_json_format(tbars_np, json_file=None, user_name='$fpl',
                         labels=None):
    tbars_json = []
    locs = tbars_np['locs']
    conf = tbars_np['conf']
    for ii in np.arange(conf.size):
        props = { 'conf' : '%.03f' % conf[ii],
                  'user' : user_name }
        tt = { 'Kind': 'PreSyn',
               'Pos' : locs[ii,:].astype('int').tolist(),
               'Prop': props }
        if labels is not None:
            tt['body ID'] = str(labels[ii])
        tbars_json.append(tt)

    if json_file is not None: # write out to file
        with open(json_file,'w') as f_out:
            json.dump(tbars_json, f_out)
    return tbars_json

def tbars_to_json_format_raveler(tbars_np, json_file=None):
    tbars_json = []
    locs = tbars_np['locs']
    conf = tbars_np['conf']
    for ii in np.arange(conf.size):
        tt = { 'confidence': '%.03f' % conf[ii],
               'location': locs[ii,:].astype('int').tolist() }
        tbars_json.append( { 'T-bar': tt } )
    tbars_json = { 'data': tbars_json }

    if json_file is not None: # write out to file
        with open(json_file,'w') as f_out:
            json.dump(tbars_json, f_out)
    return tbars_json

def tbars_push_dvid(tbars_json, dvid_server, dvid_uuid, dvid_annot):
    dvid_node = DVIDNodeService(dvid_server, dvid_uuid,
                                'fpl', 'fpl')
    dvid_conn = DVIDConnection(dvid_server, 'fpl', 'fpl')

    # create annotation if necessary
    try:
        dvid_node.custom_request('%s/info' % dvid_annot,
                                 None, ConnectionMethod.GET)
    except DVIDException as e:
        post_json = json.dumps({
            'typename': 'annotation',
            'dataname': dvid_annot})
        status, body, error_message = dvid_conn.make_request(
            '/repo/%s/instance' % dvid_uuid,
            ConnectionMethod.POST, post_json.encode('utf-8'))

    num_at_once = 100000
    n_tot       = len(tbars_json)
    for ii in range(0, n_tot, num_at_once):
        jj = np.minimum(ii+num_at_once, n_tot)
        data = json.dumps(tbars_json[ii:jj])
        oo = dvid_node.custom_request('%s/elements' % dvid_annot,
                                      data.encode('utf-8'),
                                      ConnectionMethod.POST)


def roi_to_substacks(dvid_server, dvid_uuid, dvid_roi,
                     dvid_annotations, partition_size,
                     data_dir=None, substack_ids=None,
                     image_normalize=None, buffer_size=None,
                     radius_use=None, radius_ign=None,
                     seg_name=None):

    dvid_node = DVIDNodeService(dvid_server, dvid_uuid,
                                os.getenv('USER'), 'fpl')

    if os.path.isfile(dvid_roi): # read from file
        print('reading roi %s from file' % dvid_roi)
        roi = fplutils.roi_from_txt(dvid_roi)
    else:
        roi = dvid_node.get_roi_partition(dvid_roi, partition_size)

    if substack_ids is None: # print statistics
        for ii in range(len(roi[0])):
            ss  = roi[0][ii]

            rr3 = dvid_node.get_roi3D(
                dvid_roi, (ss.size,ss.size,ss.size), (ss.z,ss.y,ss.x))
            frac_in = np.sum(rr3)/float(rr3.size)

            # grab annotations in substack, filter by roi, print num
            tt = get_substack_annotations(
                dvid_node, dvid_roi, dvid_annotations, ss)
            print('%03d\t%.03f\t%3d' % (ii, frac_in, tt['conf'].size))

    else: # write out substacks
        if not hasattr(substack_ids, '__len__'):
            substack_ids = [substack_ids,]

        for ii in substack_ids:
            print(ii)
            ss = roi[0][ii]

            # get image with buffer
            image_sz = ss.size + 2*buffer_size
            image_offset = [ss.z - buffer_size,
                            ss.y - buffer_size,
                            ss.x - buffer_size]
            image = dvid_node.get_gray3D(
                'grayscale', [image_sz, image_sz, image_sz],
                image_offset)
            image = (image.astype('float32') -
                     image_normalize[0]) / image_normalize[1]
            image_fn = '%s/%03d_im%g_%g_bf%d.h5' % (
                data_dir, ii, image_normalize[0], image_normalize[1],
                buffer_size)
            hh = h5py.File(image_fn,'w')
            hh.create_dataset('/main', image.shape,
                              dtype='float32', compression='gzip')
            hh['/main'][:] = image
            hh.close()

            # get annotations, convert to local coordinates
            tt = get_substack_annotations(
                dvid_node, dvid_roi, dvid_annotations, ss)
            if tt['conf'].size > 0:
                tt['locs'] -= np.fliplr(np.asarray([image_offset]))
            json_fn = '%s/%03d_bf%d_synapses.json' % (
                data_dir, ii, buffer_size)
            tbars_to_json_format_raveler(tt, json_fn)

            # get labels/mask, account for ROI and buffer
            mask = dvid_node.get_roi3D(
                dvid_roi, [image_sz, image_sz, image_sz],
                image_offset)
            hh = h5py.File('%s/%03d_bf%d_roimask.h5' % (
                data_dir, ii, buffer_size), 'w')
            hh.create_dataset('/main', mask.shape,
                              dtype='uint8', compression='gzip')
            hh['/main'][:] = mask
            hh.close()

            prefix = '%s/%03d_ru%d_ri%d_bf%d' % (
                data_dir, ii, radius_use, radius_ign, buffer_size)
            write_labels_mask(tt, mask, radius_use, radius_ign,
                              buffer_size, prefix)

            # get segmentation
            if seg_name is not None:
                seg = dvid_node.get_labels3D(
                    seg_name, [image_sz, image_sz, image_sz],
                    image_offset)

                seg_fn = '%s/%03d_seg_bf%d.h5' % (
                    data_dir, ii, buffer_size)
                hh = h5py.File(seg_fn, 'w')
                hh.create_dataset('/main', seg.shape,
                                  dtype='uint64', compression='gzip')
                hh['main'][:] = seg
                hh.close()


def get_substack_annotations(dvid_node, dvid_roi, dvid_annotations, ss):
    synapses_json = dvid_node.custom_request(
        '%s/elements/%d_%d_%d/%d_%d_%d' % (
            dvid_annotations, ss.size, ss.size, ss.size,
            ss.x, ss.y, ss.z), None,
        ConnectionMethod.GET)
    tt = load_from_json(synapses_json.decode())
    # filter by roi
    if dvid_roi is not None and tt['conf'].size > 0:
        pts = np.fliplr(tt['locs']).astype('int').tolist()
        in_roi = np.array(dvid_node.roi_ptquery(dvid_roi, pts))
        tt['locs'] = tt['locs'][in_roi]
        tt['conf'] = tt['conf'][in_roi]
    return tt

def write_labels_mask(tbars, roi_mask, radius_use, radius_ign,
                      buffer_size, prefix):
    radius_use_flt = fplutils.set_filter(radius_use)
    if radius_ign is not None:
        radius_ign_flt = 1 - fplutils.set_filter(radius_ign)
    else:
        radius_ign = 0

    mask   = np.copy(roi_mask);
    labels = np.zeros( mask.shape, dtype='uint8' )
    for jj in range(tbars['locs'].shape[0]):
        xx = tbars['locs'][jj,0]
        yy = tbars['locs'][jj,1]
        zz = tbars['locs'][jj,2]

        if radius_ign > 0:
            mask[(zz-radius_ign):(zz+radius_ign+1),
                 (yy-radius_ign):(yy+radius_ign+1),
                 (xx-radius_ign):(xx+radius_ign+1)
            ] = np.logical_and(
                mask[(zz-radius_ign):(zz+radius_ign+1),
                     (yy-radius_ign):(yy+radius_ign+1),
                     (xx-radius_ign):(xx+radius_ign+1)],
                radius_ign_flt)

        mask[(zz-radius_use):(zz+radius_use+1),
             (yy-radius_use):(yy+radius_use+1),
             (xx-radius_use):(xx+radius_use+1)
        ] = np.logical_or(
            mask[(zz-radius_use):(zz+radius_use+1),
                 (yy-radius_use):(yy+radius_use+1),
                 (xx-radius_use):(xx+radius_use+1)],
            radius_use_flt)
        labels[(zz-radius_use):(zz+radius_use+1),
               (yy-radius_use):(yy+radius_use+1),
               (xx-radius_use):(xx+radius_use+1)
        ] = np.logical_or(
            labels[(zz-radius_use):(zz+radius_use+1),
                   (yy-radius_use):(yy+radius_use+1),
                   (xx-radius_use):(xx+radius_use+1)],
            radius_use_flt)

    mask[:buffer_size, :,:] = 0
    mask[-buffer_size:,:,:] = 0
    mask[:,:buffer_size, :] = 0
    mask[:,-buffer_size:,:] = 0
    mask[:,:,:buffer_size ] = 0
    mask[:,:,-buffer_size:] = 0

    hh = h5py.File('%s_labels.h5' % prefix, 'w')
    hh.create_dataset('/main', labels.shape,
                      dtype='uint8', compression='gzip')
    hh['/main'][:] = labels
    hh.close()
    hh = h5py.File('%s_mask.h5' % prefix, 'w')
    hh.create_dataset('/main', mask.shape,
                      dtype='uint8', compression='gzip')
    hh['/main'][:] = mask
    hh.close()


def delete_tbar_psds(dvid_node, dvid_annot, coord):
    syn_json = dvid_node.custom_request(
        '%s/elements/1_1_1/%d_%d_%d' %
        (dvid_annot, coord[0], coord[1], coord[2]),
        None, ConnectionMethod.GET)

    assert syn_json != b'null', 'no T-bar at location'

    syn = json.loads(syn_json.decode())[0]

    if 'Rels' in syn:
        for psd in syn['Rels']:
            dvid_node.custom_request('%s/element/%d_%d_%d' % (
                dvid_annot,
                psd['To'][0], psd['To'][1], psd['To'][2]),
                                     None, ConnectionMethod.DELETE)

    dvid_node.custom_request('%s/element/%d_%d_%d' % (
        dvid_annot, syn['Pos'][0], syn['Pos'][1], syn['Pos'][2]),
                             None, ConnectionMethod.DELETE)

def mv_annotation(dvid_node, dvid_annot, coord_from, coord_to,
                  retry=False):
    syn_json = dvid_node.custom_request(
        '%s/elements/1_1_1/%d_%d_%d' %
        (dvid_annot, coord_from[0], coord_from[1], coord_from[2]),
        None, ConnectionMethod.GET)
    assert syn_json != b'null', 'no annotation at from location'

    syn_json = dvid_node.custom_request(
        '%s/elements/1_1_1/%d_%d_%d' %
        (dvid_annot, coord_to[0], coord_to[1], coord_to[2]),
        None, ConnectionMethod.GET)

    if not retry:
        assert syn_json == b'null', 'existing annotation at to location'
    elif syn_json != b'null':
        for xx in range(2,10):
            syn_json = dvid_node.custom_request(
                '%s/elements/1_1_1/%d_%d_%d' %
                (dvid_annot, coord_to[0]+xx, coord_to[1], coord_to[2]),
                None, ConnectionMethod.GET)
            if syn_json == b'null':
                coord_to[0] += xx
                break
            syn_json = dvid_node.custom_request(
                '%s/elements/1_1_1/%d_%d_%d' %
                (dvid_annot, coord_to[0]-xx, coord_to[1], coord_to[2]),
                None, ConnectionMethod.GET)
            if syn_json == b'null':
                coord_to[0] -= xx
                break

    syn_json = dvid_node.custom_request(
        '%s/elements/1_1_1/%d_%d_%d' %
        (dvid_annot, coord_to[0], coord_to[1], coord_to[2]),
        None, ConnectionMethod.GET)

    assert syn_json == b'null', 'existing annotation at to location'

    dvid_node.custom_request(
        '%s/move/%d_%d_%d/%d_%d_%d' %
        (dvid_annot, coord_from[0], coord_from[1], coord_from[2],
         coord_to[0], coord_to[1], coord_to[2]),
        None, ConnectionMethod.POST)

def rm_tbar_multi_pred(tbars_in, dvid_node, segm_name,
                       neighbor_thresh=30):
    tbars_copy = eval.Tbars(tbars_in['locs'],tbars_in['conf'])
    if segm_name:
        ll = eval.get_labels(dvid_node, segm_name, tbars_copy)
    else:
        ll = np.zeros(tbars_in['conf'].shape)

    pos = tbars_in['locs']

    dists = np.sqrt( (
        (pos.reshape((-1,1,3)) - pos.reshape((1,-1,3)))**2).sum(
            axis=2) )

    has_neighbor = np.sum( (dists>0) & (dists<neighbor_thresh),
                           axis=1 )
    tt_idx = np.argsort( -tbars_in['conf'] )

    rm_idx = np.zeros( tt_idx.shape, 'bool' )
    mv_idx = np.zeros( tt_idx.shape, 'bool' )
    mv_loc = np.zeros( pos.shape, 'int' )

    for ii in tt_idx:
        if rm_idx[ii]:
            continue
        if not has_neighbor[ii]:
            continue

        candidates = (
            (dists[ii,:]>0) & (dists[ii,:]<neighbor_thresh) &
            (ll == ll[ii]) &
            (np.logical_not(rm_idx)) )

        jj = np.nonzero(candidates)[0]

        if jj.size > 0:
            mv_idx[ii] = True

            old_loc = pos[ii,:]
            candidates[ii] = True
            updates = 0
            while True:
                # new location by interpolation
                ww = tbars_in['conf'] * candidates
                ww = ww / np.sum(ww)

                mv_loc[ii,:] = np.round(
                    np.sum( ww.reshape( (-1,1) ) * pos, axis=0) )
                if np.array_equal(old_loc, mv_loc[ii,:]):
                    break

                updates += 1

                old_loc = mv_loc[ii,:].copy()
                new_dists = np.sqrt( np.sum(
                    (pos - mv_loc[ii,:])**2, axis=1) )

                candidates = (
                    (new_dists < neighbor_thresh) &
                    (ll == ll[ii]) &
                    (np.logical_not(rm_idx)) )

            jj = np.nonzero(candidates)[0]
            rm_idx[jj] = True

            # if updates > 1:
            #     print(pos[jj,:])
            #     print(old_loc)

    rm_idx[mv_idx] = False

    return rm_idx, mv_idx, mv_loc


def get_labels(dvid_node, segm_name, tbars_in, get_supervoxels=False):
    n_tot = tbars_in['conf'].shape[0]
    n_at_once = 3000
    ll = np.zeros(n_tot,'uint64')

    sv_str = ''
    if get_supervoxels:
        sv_str = '?supervoxels=true'

    for ii in range(0,n_tot,n_at_once):
        if ii>0:
            print('[{}]'.format(ii), end='', flush=True)
        jj = np.minimum(ii+n_at_once, n_tot)
        while True:
            try:
                ll_iter = dvid_node.custom_request(
                    '%s/labels%s' % (segm_name, sv_str),
                    json.dumps(
                        tbars_in['locs'][ii:jj,:].round().astype('int'
                        ).tolist()).encode(),
                    ConnectionMethod.GET)
                break
            except DVIDException as e:
                time.sleep(60)
        ll_iter = json.loads(ll_iter.decode())
        ll[ii:jj] = np.asarray(ll_iter)

    if n_tot>n_at_once:
        print()
    return ll


def set_syncs(dvid_server, dvid_uuid, dvid_annot,
              dvid_segm=None, dvid_labelsz=None):
    dvid_node = DVIDNodeService(dvid_server, dvid_uuid,
                                os.getenv('USER'), 'fpl')
    dvid_conn = DVIDConnection(dvid_server, os.getenv('USER'), 'fpl')

    # create annotation if necessary
    try:
        dvid_node.custom_request('%s/info' % dvid_annot,
                                 None, ConnectionMethod.GET)
    except DVIDException as e:
        post_json = json.dumps({
            'typename': 'annotation',
            'dataname': dvid_annot})
        status, body, error_message = dvid_conn.make_request(
            '/repo/%s/instance' % dvid_uuid,
            ConnectionMethod.POST, post_json.encode('utf-8'))

    if dvid_labelsz is not None:
        # create labelsz if necessary
        try:
            dvid_node.custom_request('%s/info' % dvid_labelsz,
                                     None, ConnectionMethod.GET)
        except DVIDException as e:
            post_json = json.dumps({
                'typename': 'labelsz',
                'dataname': dvid_labelsz})
            status, body, error_message = dvid_conn.make_request(
                '/repo/%s/instance' % dvid_uuid,
                ConnectionMethod.POST, post_json.encode('utf-8'))

    # sync synapses to segmentation
    if dvid_segm is not None:
        syn_sync_json = json.dumps({'sync': dvid_segm})
        dvid_node.custom_request('%s/sync' % dvid_annot,
                                 syn_sync_json.encode('utf-8'),
                                 ConnectionMethod.POST)

    # sync labelsz to synapses
    if dvid_labelsz is not None:
        lsz_sync_json = json.dumps({'sync': dvid_annot})
        dvid_node.custom_request('%s/sync' % dvid_labelsz,
                                 lsz_sync_json.encode('utf-8'),
                                 ConnectionMethod.POST)
