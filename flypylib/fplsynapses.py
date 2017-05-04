"""functions for working with synapse data, json, dvid """

from flypylib import fplutils
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
