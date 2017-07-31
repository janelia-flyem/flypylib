"""functions for working with synapse data, json, dvid """

from flypylib import fplutils
from libdvid import DVIDNodeService, ConnectionMethod, DVIDConnection
from libdvid._dvid_python import DVIDException
import json
import numpy as np

def load_from_json(fn, vol_sz=None, buffer=None):
    """read synapse data from json file

    """

    with open(fn) as json_file:
        data = json.load(json_file)

    locs = []
    conf = []
    if(('data' in data.keys()) and len(data['data'])>0 and
       ('T-bar' in data['data'][0].keys())): # Raveler format

        for syn in data['data']:
            locs.append(syn['T-bar']['location'])
            conf.append(syn['T-bar']['confidence'])

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
            ConnectionMethod.POST, post_json)

    data = json.dumps(tbars_json)
    dvid_node.custom_request('%s/elements' % dvid_annot,
                             data, ConnectionMethod.POST)
