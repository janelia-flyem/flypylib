from flypylib import fplutils
from keras.models import Model
from keras.layers import UpSampling3D
import keras.backend as K
import h5py
import numpy as np

def masked_binary_crossentropy(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 2), K.floatx())
    return K.mean(K.binary_crossentropy(y_pred * mask,
                                        y_true * mask))

def masked_accuracy(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 2), K.floatx())
    return K.mean(K.equal(y_true * mask,
                          K.round(y_pred * mask)))

class FplNetwork:
    """deep learning/CNN class that wraps keras model

    supports training using keras fit_generator, and full stack
    inference

    """

    def __init__(self, model, rf_size, rf_offset, rf_stride, model_type):
        self.model  = model
        self.rf_size   = fplutils.to3d(rf_size)
        self.rf_offset = fplutils.to3d(rf_offset)
        self.rf_stride = fplutils.to3d(rf_stride)
        self.model_type = model_type

        self.train_network = None
        self.infer_network = None

        self.infer_sz      = (100,100,100)
        self.infer_sz      = tuple([
            round( (ii-2*oo)/ss ) * ss + 2*oo for
            ii,oo,ss in zip(self.infer_sz,
                            self.rf_offset, self.rf_stride)])

    def train(self, generator, steps_per_epoch, epochs):
        if self.train_network is None:
            if (self.model_type == 'unet'):
                self.train_network = self.model(self.rf_size)
            else:
                self.train_network = self.model(None)
            self.train_network.summary()
            if (self.model_type == 'unet'):
                self.train_network.compile(
                    loss=masked_binary_crossentropy,
                    optimizer='adam',
                    metrics=[masked_accuracy]
                )
            else:
                self.train_network.compile(
                    loss      = 'binary_crossentropy',
                    optimizer = 'adam',
                    metrics   = ['accuracy'])
        self.train_network.fit_generator(
            generator, steps_per_epoch, epochs)

    def infer(self, image):
        if(isinstance(image, str)):
            image = h5py.File(image,'r')
            image = image['/main'][:]

        if(self.infer_network is None or \
           self.infer_network.input_shape[1:-1] != self.infer_sz):
            if self.model_type == 'resnet' or \
               self.model_type == 'None':
                initial_model = self.model(self.infer_sz)
                upsample_pred = UpSampling3D(self.rf_stride)(initial_model.output)
                self.infer_network = Model(initial_model.input, upsample_pred)
            elif self.model_type == 'unet':
                self.infer_network = self.model(self.infer_sz)
            else:
                assert False, 'should not get here after refactoring'

            self.infer_network.set_weights(
                self.train_network.get_weights())

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

        data_batch = np.zeros(
            (n_idx,infer_sz[0,0],infer_sz[1,0],infer_sz[2,0],1))

        for ii in range(n_idx):
            data_batch[ii,:idx_sz[0,ii],:idx_sz[1,ii],:idx_sz[2,ii],
                 0] = image[
                     start_idx[0,ii]:end_idx[0,ii],
                     start_idx[1,ii]:end_idx[1,ii],
                     start_idx[2,ii]:end_idx[2,ii]]

        pred_batch = self.infer_network.predict(
            data_batch, batch_size=1)

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
