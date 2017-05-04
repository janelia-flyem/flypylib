from flypylib import fplobjdetect, fplmodels, fplnetwork, fplsynapses
import numpy as np
import matplotlib.pyplot as plt
import os

# choose a net architecture
#   possible models: baseline_model, vgg_like, resnet_like, unet_like
model       = fplmodels.vgg_like2;
network     = fplnetwork.FplNetwork(model)

n_gpu       = 4
batch_size  = 64
train_shape = network.rf_size


is_mask = (model==fplmodels.unet_like)
network.make_train_parallel(n_gpu, batch_size, train_shape)

base_dir  = '/groups/flyem/data/synapse_training'
if not os.path.isdir(base_dir):
    base_dir = '/data'

train_idx = (28, 29, 35, 40)
test_idx  = (54, 58, 60, 61, 65, 69, 70, 75, 95, 103)

train_dir   = '%s/cx1_0_0' % base_dir
train_data = []
train_eval = []
for ii in train_idx:
    train_image = '%s/cx1_%03d_mn135_std48_image.h5' % (train_dir,ii)
    train_data.append( (
        train_image,
        '%s/cx1_%03d_ru7_ri15_mn135_std48_im_' % (train_dir,ii) ))
    train_eval.append( [
        train_image,
        '%s/cx1_%03d_synapses.json' % (train_dir,ii) ] )
generator  = fplobjdetect.gen_batches(
    train_data, network.rf_size, batch_size*n_gpu, is_mask)

test_dir   = '%s/cx1_0_1' % base_dir
test_data  = []
for ii in test_idx:
    test_image = '%s/cx1_%03d_mn135_std48_image.h5' % (test_dir,ii)
    test_json  = '%s/cx1_%03d_synapses.json'        % (test_dir,ii)
    test_data.append( [test_image, test_json] )

for ee in range(1):
    network.train(generator, 1000, 10)
    network.make_infer_parallel(4)

    mm_train_agg, mm_train = fplobjdetect.evaluate_substacks(
        network, train_eval, np.arange(0.6,0.96,0.02),
        obj_min_dist=27, smoothing_sigma=5,
        volume_offset=None, buffer_sz=15)
    mm_test_agg, mm_test = fplobjdetect.evaluate_substacks(
        network, test_data, np.arange(0.6,0.96,0.02),
        obj_min_dist=27, smoothing_sigma=5,
        volume_offset=None, buffer_sz=15)

    plt.figure()
    plt.plot(mm_train_agg.rr, mm_train_agg.pp, 'b-')
    plt.plot(mm_test_agg.rr,  mm_test_agg.pp,  'r-')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend(['train', 'test'], loc='lower left')
    plt.show()

# network.save('vgg2.p')
