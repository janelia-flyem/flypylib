from flypylib import fplobjdetect, fplmodels, fplnetwork, fplsynapses
import numpy as np

network    = fplnetwork.FplNetwork(fplmodels.baseline_model, 18, 7, 4)

base_dir   = '/groups/flyem/data/synapse_training/cx'
train_data = (
    '%s/pb26_03_mn135_std48_image.h5' % base_dir,
    '%s/pb26_03_ru7_ri15_mn135_std48_im_' % base_dir )
generator  = fplobjdetect.gen_batches(
    train_data, network.rf_size, 32)

network.train(generator, 1000, 10)

train_json = '%s/pb26_03_synapses.json' % base_dir
pred       = network.infer(train_data[0])
out        = fplobjdetect.voxel2obj(pred, 27, 3, None, 5)

gt         = fplsynapses.load_from_json(train_json)

mm_train   = fplobjdetect.obj_pr_curve(
    out, gt, 27, np.arange(0.6,0.96,0.02) )

test_data  = '%s/pb26_04_mn135_std48_image.h5' % base_dir
test_json  = '%s/pb26_04_synapses.json' % base_dir

pred       = network.infer(test_data)
out        = fplobjdetect.voxel2obj(pred, 27, 3, None, 5)

gt         = fplsynapses.load_from_json(test_json)

mm_test    = fplobjdetect.obj_pr_curve(
    out, gt, 27, np.arange(0.6,0.96,0.02) )
