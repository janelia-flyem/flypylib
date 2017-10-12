from flypylib import fplobjdetect, fplmodels, fplnetwork, fplsynapses
from diced import DicedStore, DicedException
import numpy as np
import h5py
import urllib, os

# specify directory for data
data_dir = 'fib25_data'
# number of available gpus
n_gpu = 4

## set up training, test data ------------------------------
try:
    os.makedirs(data_dir)
except OSError:
    if not os.path.isdir(data_dir):
        raise

radius_use = 7
radius_ign = 15

# x,y,z ordering
offsets = [
    [3232, 2944, 1984],
    [3744, 2944, 1984]]
vol_sz = 400
buffer_sz = 30
n_offsets = len(offsets)

store = DicedStore("gs://flyem-public-connectome")
repo = store.open_repo(uuid="8225")
full_im = repo.get_array("grayscale")

image_normalize = [128.,33.]

syn_url = 'http://emdata.janelia.org/api/node/8225/.files/key/synapse.json'
full_syn_json = urllib.request.urlopen(syn_url).read().decode()
full_syn = fplsynapses.load_from_json(full_syn_json)

for ii in range(n_offsets):
    im = full_im[
        (offsets[ii][2]-buffer_sz):(
            offsets[ii][2]+vol_sz+buffer_sz),
        (offsets[ii][1]-buffer_sz):(
            offsets[ii][1]+vol_sz+buffer_sz),
        (offsets[ii][0]-buffer_sz):(
            offsets[ii][0]+vol_sz+buffer_sz)]
    im = (im.astype('float32') -
          image_normalize[0]) / image_normalize[1]
    hh = h5py.File('%s/%d_im.h5' % (data_dir,ii), 'w')
    hh.create_dataset('/main', im.shape,
                      dtype='float32', compression='gzip')
    hh['/main'][:] = im
    hh.close()

    idx = ((full_syn['locs'][:,0] >= offsets[ii][0]) &
           (full_syn['locs'][:,1] >= offsets[ii][1]) &
           (full_syn['locs'][:,2] >= offsets[ii][2]) &
           (full_syn['locs'][:,0] < offsets[ii][0] + vol_sz) &
           (full_syn['locs'][:,1] < offsets[ii][1] + vol_sz) &
           (full_syn['locs'][:,2] < offsets[ii][2] + vol_sz))

    tbars = { 'locs' : full_syn['locs'][idx,:],
              'conf' : full_syn['conf'][idx] }
    tbars['locs'] -= (np.asarray( [offsets[ii]] ) - buffer_sz)

    json_fn = '%s/%d_synapses.json' % (data_dir,ii)
    fplsynapses.tbars_to_json_format_raveler(tbars, json_fn)

    mask = np.ones(im.shape, dtype='uint8')
    prefix = '%s/%d' % (data_dir, ii)
    fplsynapses.write_labels_mask(tbars, mask, radius_use, radius_ign,
                                  buffer_sz, prefix)


## train network -------------------------------------------
model       = fplmodels.unet_like2;
network     = fplnetwork.FplNetwork(model)

batch_size  = 64
train_shape = network.rf_size

train_data = [
    [ '%s/0_im.h5' % data_dir, '%s/0_' % data_dir]
    ]
train_eval = [
    [ '%s/0_im.h5' % data_dir, '%s/0_synapses.json' % data_dir]
    ]
test_eval = [
    [ '%s/1_im.h5' % data_dir, '%s/1_synapses.json' % data_dir]
    ]
generator = fplobjdetect.gen_volume2(
    train_data, train_shape, batch_size*n_gpu, 0.5)
if n_gpu > 1:
    network.make_train_parallel(n_gpu, batch_size, train_shape)

network.train(generator, 1000, 5)

## run inference, plot precision/recall --------------------
filename = '%s/net-%d.p' % (out_dir, ee)
network.save_network(filename)

if n_gpu > 1:
    network.make_infer_parallel(n_gpu)

mm_train_agg, mm_train = fplobjdetect.evaluate_substacks(
    network, train_eval, np.arange(0.6,0.96,0.02),
    obj_min_dist=27, smoothing_sigma=5,
    buffer_sz=buffer_sz)
mm_test_agg, mm_test = fplobjdetect.evaluate_substacks(
    network, test_eval, np.arange(0.6,0.96,0.02),
    obj_min_dist=27, smoothing_sigma=5,
    buffer_sz=buffer_sz)

plt.figure()
plt.plot(mm_train_agg.rr, mm_train_agg.pp, 'b-')
plt.plot(mm_test_agg.rr,  mm_test_agg.pp,  'r-')
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(['train', 'test'], loc='lower left')
plt.grid(linestyle='dashed')
plt.show()
