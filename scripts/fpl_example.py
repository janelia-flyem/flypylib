from flypylib import fplobjdetect, fplmodels, fplnetwork, fplsynapses
import numpy as np
import matplotlib.pyplot as plt

# choose a net architecture
#   possible models: baseline_model, vgg_like, resnet_like, unet_like
model = fplmodels.baseline_model;

network = fplnetwork.FplNetwork(model)
is_mask = (model==fplmodels.unet_like)

base_dir   = '/groups/flyem/data/synapse_training/cx'
train_data = []
for ii in (3,5,7):
    train_data.append( (
        '%s/pb26_%02d_mn135_std48_image.h5'     % (base_dir,ii),
        '%s/pb26_%02d_ru7_ri15_mn135_std48_im_' % (base_dir,ii) ))
generator  = fplobjdetect.gen_batches(
    train_data, network.rf_size, 32, is_mask)

network.train(generator, 1000, 10)

# parallelize inference across 4 gpus
network.make_parallel(4)

train_json = []
for ii in (3,5,7):
    train_json.append('%s/pb26_%02d_synapses.json' % (base_dir,ii))

mm_train = []
for ii in range(3):
    pred       = network.infer(train_data[ii][0])
    out        = fplobjdetect.voxel2obj(pred, 27, 5, None, 5)
    gt         = fplsynapses.load_from_json(train_json[ii])

    mm_train.append(fplobjdetect.obj_pr_curve(
        out, gt, 27, np.arange(0.6,0.96,0.02) ))
mm_train_agg = fplobjdetect.aggregate_pr(mm_train)

mm_test = []
for ii in (4,9):
    test_data  = '%s/pb26_%02d_mn135_std48_image.h5' % (base_dir,ii)
    test_json  = '%s/pb26_%02d_synapses.json'        % (base_dir,ii)

    pred       = network.infer(test_data)
    out        = fplobjdetect.voxel2obj(pred, 27, 5, None, 5)

    gt         = fplsynapses.load_from_json(test_json)

    mm_test.append(fplobjdetect.obj_pr_curve(
        out, gt, 27, np.arange(0.6,0.96,0.02) ))
mm_test_agg = fplobjdetect.aggregate_pr(mm_test)

plt.plot(mm_train_agg.rr, mm_train_agg.pp, 'b-')
plt.plot(mm_test_agg.rr,  mm_test_agg.pp,  'r-')
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(['train', 'test'], loc='lower left')
plt.show()
