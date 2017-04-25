"""functions for working with synapse data, json, dvid """

import json
import numpy as np

def load_from_json(fn):
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
    tbars = { 'locs': locs, 'conf': conf }
    return tbars
