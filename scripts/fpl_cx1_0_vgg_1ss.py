from flypylib import fplobjdetect, fplmodels, fplnetwork, fplsynapses
import numpy as np
import matplotlib.pyplot as plt

# choose a net architecture
#   possible models: baseline_model, vgg_like, resnet_like, unet_like
model = fplmodels.vgg_like;

network = fplnetwork.FplNetwork(model)
is_mask = (model==fplmodels.unet_like)

base_dir  = '/groups/flyem/data/synapse_training'
train_idx = (28,)
test_idx  = (54, 58, 60, 61, 65, 69, 70, 75, 95, 103)

train_dir   = '%s/cx1_0_0' % base_dir
train_data = []
for ii in train_idx:
    train_data.append( (
        '%s/cx1_%03d_mn135_std48_image.h5'     % (train_dir,ii),
        '%s/cx1_%03d_ru7_ri15_mn135_std48_im_' % (train_dir,ii) ))
generator  = fplobjdetect.gen_batches(
    train_data, network.rf_size, 32, is_mask)

network.train(generator, 1000, 10)
network.make_parallel(4)

train_json = []
for ii in train_idx:
    train_json.append('%s/cx1_%03d_synapses.json' % (train_dir,ii))

mm_train = []
for ii in range(len(train_data)):
    pred       = network.infer(train_data[ii][0])
    out        = fplobjdetect.voxel2obj(pred, 27, 5, None, 5)
    gt         = fplsynapses.load_from_json(train_json[ii])

    mm_train.append(fplobjdetect.obj_pr_curve(
        out, gt, 27, np.arange(0.6,0.96,0.02) ))
mm_train_agg = fplobjdetect.aggregate_pr(mm_train)

test_dir   = '%s/cx1_0_1' % base_dir
test_data  = []
for ii in test_idx:
    test_image = '%s/cx1_%03d_mn135_std48_image.h5' % (test_dir,ii)
    test_json  = '%s/cx1_%03d_synapses.json'        % (test_dir,ii)
    test_data.append( [test_image, test_json] )
mm_test_agg, mm_test = fplobjdetect.evaluate_substacks(
    network, test_data, np.arange(0.6,0.96,0.02),
    obj_min_dist=27, smoothing_sigma=5, volume_offset=None, buffer_sz=5)

plt.plot(mm_train_agg.rr, mm_train_agg.pp, 'b-')
plt.plot(mm_test_agg.rr,  mm_test_agg.pp,  'r-')
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(['train', 'test'], loc='lower left')
plt.show()
