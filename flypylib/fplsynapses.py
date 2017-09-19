"""functions for working with synapse data, json, dvid """

from flypylib import fplutils
from libdvid import DVIDNodeService, ConnectionMethod, DVIDConnection
from libdvid._dvid_python import DVIDException
import numpy as np
import h5py
import json, os

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

    if((isinstance(data, dict) and
        'data' in data.keys())): # Raveler format

        for syn in data['data']:
            locs.append(syn['T-bar']['location'])
            conf.append(syn['T-bar']['confidence'])
    elif data is not None: # assume dvid format
        for syn in data:
            if syn['Kind'] != 'PreSyn':
                continue

            if 'conf' in syn['Prop']:
                cc = float(syn['Prop']['conf'])
            else:
                cc = 1.0

            locs.append(syn['Pos'])
            conf.append(cc)

    locs  = np.asarray(locs)
    conf  = np.asarray(conf)

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

    tbars = { 'locs': locs, 'conf': conf }
    return tbars

def tbars_to_json_format(tbars_np, json_file=None):
    tbars_json = []
    locs = tbars_np['locs']
    conf = tbars_np['conf']
    for ii in np.arange(conf.size):
        props = { 'conf' : '%.03f' % conf[ii] }
        tt = { 'Kind': 'PreSyn',
               'Pos' : locs[ii,:].astype('int').tolist(),
               'Prop': props }
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

    data = json.dumps(tbars_json)
    oo = dvid_node.custom_request('%s/elements' % dvid_annot,
                                  data.encode('utf-8'),
                                  ConnectionMethod.POST)

def roi_to_substacks(dvid_server, dvid_uuid, dvid_roi,
                     dvid_annotations, partition_size,
                     data_dir=None, substack_ids=None,
                     image_normalize=None, buffer_size=None,
                     radius_use=None, radius_ign=None):

    dvid_node = DVIDNodeService(dvid_server, dvid_uuid,
                                os.getenv('USER'), 'fpl')
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

        radius_use_flt = fplutils.set_filter(radius_use)
        if radius_ign is not None:
            radius_ign_flt = 1 - fplutils.set_filter(radius_ign)
        else:
            radius_ign = 0

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

            labels = np.zeros( mask.shape, dtype='uint8' )
            for jj in range(tt['locs'].shape[0]):
                xx = tt['locs'][jj,0]
                yy = tt['locs'][jj,1]
                zz = tt['locs'][jj,2]

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

            prefix = '%s/%03d_ru%d_ri%d_bf%d' % (
                data_dir, ii, radius_use, radius_ign, buffer_size)
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


def get_substack_annotations(dvid_node, dvid_roi, dvid_annotations, ss):
    synapses_json = dvid_node.custom_request(
        '%s/elements/%d_%d_%d/%d_%d_%d' % (
            dvid_annotations, ss.size, ss.size, ss.size,
            ss.x, ss.y, ss.z), None,
        ConnectionMethod.GET)
    tt = load_from_json(synapses_json.decode())
    # filter by roi
    if tt['conf'].size > 0:
        pts = np.fliplr(tt['locs']).astype('int').tolist()
        in_roi = np.array(dvid_node.roi_ptquery(dvid_roi, pts))
        tt['locs'] = tt['locs'][in_roi]
        tt['conf'] = tt['conf'][in_roi]
    return tt
