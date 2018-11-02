from flypylib import fplutils, multi_gpu
from keras.models import Model, load_model
from keras.layers import UpSampling3D
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback
import h5py
import numpy as np
import pickle

class multi_gpu_callback(Callback):

    def __init__(self, model, save_prefix):
         self.model_to_save = model
         self.save_prefix = save_prefix

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save(
            '%s_%03d.h5' % (self.save_prefix, epoch))

def get_custom_objects(compile_args):
    custom_objects = {}

    if not isinstance(compile_args['loss'], str):
        custom_objects[ compile_args['loss'].__name__ ] \
            = compile_args['loss']

    for mm in compile_args['metrics']:
        if not isinstance(mm, str):
            custom_objects[mm.__name__] = mm

    return custom_objects

def load_network(filepath):
    fn = open(filepath,'rb')
    network = pickle.load(fn)
    fn.close()

    custom_objects = get_custom_objects(network.compile_args)
    keras_filepath = filepath + '.keras.h5'
    network.train_single  = load_model(
        keras_filepath, custom_objects=custom_objects)
    network.train_network = network.train_single
    network._set_infer()

    return network

class FplNetwork:
    """deep learning/CNN class that wraps keras model

    supports training using keras fit_generator, and full stack
    inference

    """

    def __init__(self, model):
        self.model = model

        self.train_network, rf_info, infer_sz, compile_args = self.model()
        self.train_network.summary()
        self.train_single = self.train_network

        self.rf_size   = fplutils.to3d(rf_info[0])
        self.rf_offset = fplutils.to3d(rf_info[1])
        self.rf_stride = fplutils.to3d(rf_info[2])

        self.infer_network = None
        self.n_gpu         = 1

        self.infer_sz      = fplutils.to3d(infer_sz)
        # self.infer_sz      = tuple([
        #     round( (ii-2*oo)/ss ) * ss + 2*oo for
        #     ii,oo,ss in zip(self.infer_sz,
        #                     self.rf_offset, self.rf_stride)])

        if compile_args is None:
            compile_args = {'loss': 'binary_crossentropy',
                            'optimizer': 'adam',
                            'metrics': ['accuracy']}
        self.train_network.compile(**compile_args)
        self.compile_args = compile_args

    def save_network(self, filepath):
        keras_filepath = filepath + '.keras.h5'
        self.train_single.save(keras_filepath)

        self.train_single  = None
        self.train_network = None
        self.infer_network = None

        fn = open(filepath,'wb')
        pickle.dump(self, fn)
        fn.close()

        custom_objects     = get_custom_objects(self.compile_args)
        self.train_single  = load_model(keras_filepath,
                                        custom_objects=custom_objects)
        self.train_network = self.train_single
        self._set_infer()

    def _set_infer(self):
        if self.rf_stride != (1,1,1): # need to upsample
            initial_model,_,_,_ = self.model(self.infer_sz)
            upsample_pred   = UpSampling3D(self.rf_stride)(
                initial_model.output)
            self.infer_network = Model(initial_model.input,
                                       upsample_pred)
        else:
            self.infer_network,_,_,_ = self.model(self.infer_sz)

        self.infer_network.set_weights(
            self.train_single.get_weights())

    def train(self, generator, steps_per_epoch, epochs,
              log_file, save_filepath):

        csv_logger = CSVLogger(log_file)
        checkpoint = multi_gpu_callback(
            self.train_single, save_filepath)
        callbacks = [csv_logger, checkpoint]

        self.train_network.fit_generator(
            generator, steps_per_epoch, epochs, callbacks=callbacks)
        self._set_infer()

    def make_train_parallel(self, n_gpu, batch_size, input_shape):
        input_shape = list(fplutils.to3d(input_shape)) + [1,]
        self.train_network = multi_gpu.make_parallel(
            self.train_single, n_gpu, batch_size, input_shape)
        self.train_network.compile(**self.compile_args)

    def make_infer_parallel(self, n_gpu):
        self._set_infer()
        self.infer_network = multi_gpu.make_parallel(
            self.infer_network, n_gpu)
        self.n_gpu = n_gpu

    def infer(self, image):
        if(isinstance(image, str)):
            image = h5py.File(image,'r')
            image = image['/main'][:]

        assert self.infer_network is not None, \
            'network has not been trained'
        assert self.infer_network.input_shape[1:-1] == self.infer_sz, \
            'network input shape does not match expected infer_sz'

        image_sz  = np.asarray(image.shape).reshape(3,1)
        infer_sz  = np.asarray(self.infer_sz).reshape(3,1)
        offset_sz = np.asarray(self.rf_offset).reshape(3,1)
        out_sz    = infer_sz - 2*offset_sz

        locs = np.mgrid[
            offset_sz[0,0]:image_sz[0,0]-offset_sz[0,0]:out_sz[0,0],
            offset_sz[1,0]:image_sz[1,0]-offset_sz[1,0]:out_sz[1,0],
            offset_sz[2,0]:image_sz[2,0]-offset_sz[2,0]:out_sz[2,0]]
        locs = locs.reshape(3,-1)

        start_idx = locs - offset_sz
        end_idx   = np.minimum(locs + out_sz + offset_sz, image_sz)
        idx_sz    = end_idx - start_idx
        n_idx     = locs.shape[1]

        n_idx_batch = int(np.ceil(n_idx /
                                  float(self.n_gpu))*self.n_gpu)

        data_batch = np.zeros(
            (n_idx_batch,infer_sz[0,0],infer_sz[1,0],infer_sz[2,0],1))

        for ii in range(n_idx):
            data_batch[ii,:idx_sz[0,ii],:idx_sz[1,ii],:idx_sz[2,ii],
                 0] = image[
                     start_idx[0,ii]:end_idx[0,ii],
                     start_idx[1,ii]:end_idx[1,ii],
                     start_idx[2,ii]:end_idx[2,ii]]

        pred_batch = self.infer_network.predict(
            data_batch, batch_size=self.n_gpu)

        pred = np.zeros( image.shape, dtype='float32' )

        for ii in range(n_idx):
            pred[locs[0,ii]:end_idx[0,ii]-offset_sz[0,0],
                 locs[1,ii]:end_idx[1,ii]-offset_sz[1,0],
                 locs[2,ii]:end_idx[2,ii]-offset_sz[2,0]] = pred_batch[
                     ii,
                     :idx_sz[0,ii]-2*offset_sz[0,0],
                     :idx_sz[1,ii]-2*offset_sz[1,0],
                     :idx_sz[2,ii]-2*offset_sz[2,0],0]

        return pred

    def voxel_loss(self, image, lm_prefix,
                   l0_thresh=None, l1_thresh=None):
        pred = self.infer(image)
        ll   = h5py.File('%slabels.h5' % lm_prefix, 'r')['/main'][:]
        mm   = h5py.File('%smask.h5'   % lm_prefix, 'r')['/main'][:]

        context_rr  = tuple(round(cc/2) for cc in self.rf_size)
        mm[:context_rr[0],:,:] = 0
        mm[:,:context_rr[1],:] = 0
        mm[:,:,:context_rr[2]] = 0
        mm[-context_rr[0]:,:,:] = 0
        mm[:,-context_rr[1]:,:] = 0
        mm[:,:,-context_rr[2]:] = 0

        l0_loss = -1.*(mm==1)*(ll==0)*np.log(np.maximum(1-pred, 1e-8))
        conf_neg = (l0_loss < 0.005)*(mm==1)*(ll==0)
        l0_loss[conf_neg] = 0
        mm[     conf_neg] = 0
        if l0_thresh is not None:
            l0_mask = (ll==0)*(mm==1)
            l0_loss = np.maximum(l0_loss, l0_thresh[0]*l0_mask)
            l0_loss = np.minimum(l0_loss, l0_thresh[1]*l0_mask)

        l1_loss = -1.*(mm==1)*(ll==1)*np.log(np.maximum(  pred, 1e-8))
        if l1_thresh is not None:
            l1_mask = (ll==1)*(mm==1)
            l1_loss = np.maximum(l1_loss, l1_thresh[0]*l1_mask)
            l1_loss = np.minimum(l1_loss, l1_thresh[1]*l1_mask)

        return (l0_loss + l1_loss).astype('float32')
