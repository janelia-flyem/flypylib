"""defines keras models/network architectures to use for object
detection

"""

from flypylib import fplutils
from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv3D, MaxPooling3D
from keras.layers import BatchNormalization
import numpy as np

def baseline_model(in_sz = None):
    """returns simple baseline model
    """

    in_sz = fplutils.to3d(in_sz)
    in_sz = in_sz + (1,)

    model = Sequential()

    model.add(Conv3D(32, (3,3,3), use_bias=False,
                     input_shape=in_sz))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))

    model.add(Conv3D(32, (3,3,3), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))

    model.add(Conv3D(32, (3,3,3), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3D(64, (1,1,1), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv3D(1, (1,1,1), activation='sigmoid'))

    return model

def vgg_like(in_sz = None):
    """returns standard model based on VGG architecture"""

    in_sz = fplutils.to3d(in_sz)
    in_sz = in_sz + (1,)

    model = Sequential()

    model.add(Conv3D(48, (3,3,3), use_bias=False,
                     input_shape=in_sz))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(48, (1,1,1), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))

    model.add(Conv3D(48, (3,3,3), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv3D(48, (1,1,1), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2,2,2)))

    model.add(Conv3D(48, (3,3,3), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv3D(96, (1,1,1), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv3D(96, (1,1,1), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Conv3D(1, (1,1,1), activation='sigmoid'))

    return model
