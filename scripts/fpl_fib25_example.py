#!/usr/bin/env python

from flypylib import fplobjdetect, fplmodels, fplnetwork, fplsynapses
from diced import DicedStore, DicedException
import matplotlib.pyplot as plt
import numpy as np
import h5py
import urllib, os

######
# fpl_fib25_example.py
#
# This is a test program that uses publically available data to demonstrate
# the use of the flypylib library.

# number of available gpus on our system
n_gpu = 4

# How many steps to do in an epoch, and how many epochs to do
epochs = 5
epoch_steps = 1000

# specify directory for data. Right now this uses a relative path, so it'll be
# under where the program is run.
data_dir = 'fib25_data'
# We'll use this to store data about where we're getting our data from
dsource = {
    'url': 'gs://flyem-public-connectome', # Where our data is stored
    'repo': '8225', # The repo under there
    'array': 'grayscale' # And the array under there
    }
# A source of curated synapses for the above data source, in json format
synsource = 'http://emdata.janelia.org/api/node/8225/.files/key/synapse.json'

#####################

# This loads the voxel data from where it is stored
def load_data(data):
    store   = DicedStore(data['url'])
    repo    = store.open_repo(uuid=data['repo'])
    full_im = repo.get_array(data['array'])
    return full_im

# This loads a list of synapses for the data above, for training and evaluation
def load_synapses(syndata):
    full_syn_json = urllib.request.urlopen(syndata).read().decode()
    return fplsynapses.load_from_json(full_syn_json)

def ensure_directory(ddir):
    # Make sure the directory we're loading data into exists.
    # This will not clear out the dir if it already exists. Maybe
    # it should?
    try:
        os.makedirs(data_dir)
    except OSError:
        if not os.path.isdir(data_dir):
            raise

###############
# Filtering

def prep_region(full_image, full_synapses, offset, numeric_prefix, normalization, radius, vol_sz, buffer_sz):
    # This takes a region (identified by the 3D offset where it starts) and prepares from the
    # original full dataset a filtered on-disk dataset and synapse list for later analysis.
        im = full_image[
            (offset[2] - buffer_sz):(
                offset[2] + vol_sz + buffer_sz),
            (offset[1] - buffer_sz):(
                offset[1] + vol_sz + buffer_sz),
            (offset[0] - buffer_sz):(
                offset[0] + vol_sz + buffer_sz)]
        im = (im.astype('float32') - normalization['sub']) / normalization['div']
        hh = h5py.File('%s/%d_im.h5' % (data_dir, numeric_prefix), 'w')
        hh.create_dataset('/main', im.shape,
                          dtype='float32', compression='gzip')
        hh['/main'][:] = im
        hh.close()

# Filter the synapse list down to the training and test regions
        idx = ((full_synapses['locs'][:,0] >= offset[0]) &
               (full_synapses['locs'][:,1] >= offset[1]) &
               (full_synapses['locs'][:,2] >= offset[2]) &
               (full_synapses['locs'][:,0]  < offset[0] + vol_sz) &
               (full_synapses['locs'][:,1]  < offset[1] + vol_sz) &
               (full_synapses['locs'][:,2]  < offset[2] + vol_sz))

        tbars = { 'locs' : full_synapses['locs'][idx,:],
                  'conf' : full_synapses['conf'][idx] }
        tbars['locs'] -= (np.asarray( [offset] ) - buffer_sz)

        json_fn = '%s/%d_synapses.json' % (data_dir, numeric_prefix)
        fplsynapses.tbars_to_json_format_raveler(tbars, json_fn)

        mask = np.ones(im.shape, dtype='uint8')
        prefix = '%s/%d' % (data_dir, numeric_prefix)
        fplsynapses.write_labels_mask(tbars, mask, radius['use'], radius['ign'],
                                      buffer_sz, prefix)

##################
# Plotting

# Given aggregate data for training and test, make a nice combined plot
def plot_results(mm_train_agg, mm_test_agg):
    plt.figure()
    plt.plot(mm_train_agg.rr, mm_train_agg.pp, 'b-')
    plt.plot(mm_test_agg.rr,  mm_test_agg.pp,  'r-')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(['train', 'test'], loc='lower left')
    plt.grid(linestyle='dashed')
    plt.show()



####

def main():
## set up training, test data ------------------------------
    ensure_directory(data_dir)
    radius = {
        'use': 7,
        'ign': 15
        }

# This uses distinct regions of the same underlying public data
# for training and test purposes, respectively.
# x,y,z ordering
    training_offsets = [3232, 2944, 1984]
    test_offsets     = [3744, 2944, 1984]

    image_normalize = {
        'sub': 128.0,
        'div': 33.0
    }

    vol_sz    = 400 # This number of units in each direction defines the region
    buffer_sz = 30  # plus 30 extra for the algorithms to find feature edges in

    full_im  = load_data(dsource)
    full_syn = load_synapses(synsource)
# Filter the image data into training and test regions.
# "0" is the path component for data used for training
# "1" is the path component for data used to test our trained network
    prep_region(full_im, full_syn, training_offsets, 0, image_normalize, radius, vol_sz, buffer_sz)
    prep_region(full_im, full_syn, test_offsets,     1, image_normalize, radius, vol_sz, buffer_sz)

## train network -------------------------------------------
    model       = fplmodels.unet_like2;
    network     = fplnetwork.FplNetwork(model)

    batch_size  = 64
    train_shape = network.rf_size

    train_data = [ [ '%s/0_im.h5' % data_dir, '%s/0_' % data_dir] ]
    train_eval = [ [ '%s/0_im.h5' % data_dir, '%s/0_synapses.json' % data_dir] ]
    test_eval  = [ [ '%s/1_im.h5' % data_dir, '%s/1_synapses.json' % data_dir] ]
    generator = fplobjdetect.gen_volume2(train_data, train_shape, batch_size * n_gpu, 0.5)

    if n_gpu > 1:
        network.make_train_parallel(n_gpu, batch_size, train_shape)

    network.train(generator, epoch_steps, epochs,
                  '%s/log.csv' % data_dir,
                  '%s/epoch' % data_dir)
    filename = '%s/net.p' % data_dir
    network.save_network(filename)

## run inference, plot precision/recall --------------------
    if n_gpu > 1:
        network.make_infer_parallel(n_gpu)

    mm_train_agg, mm_train = fplobjdetect.evaluate_substacks(
        network, train_eval, np.arange(0.6, 0.96, 0.02),
        obj_min_dist=27, smoothing_sigma=5,
        buffer_sz=buffer_sz)
    mm_test_agg, mm_test = fplobjdetect.evaluate_substacks(
        network, test_eval, np.arange(0.6, 0.96, 0.02),
        obj_min_dist=27, smoothing_sigma=5,
        buffer_sz=buffer_sz)

    # Make a nice plot of how the network did
    plot_results(mm_train_agg, mm_test_agg)

main()
