"""defines keras models/network architectures to use for object
detection

"""

from flypylib import fplutils
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Activation, Conv3D, MaxPooling3D, Cropping3D, UpSampling3D
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import add, concatenate
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

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization()(input)
    return Activation("relu")(norm)

def resnet_like(in_sz=None):
    """ returns a model that uses residual components
    """
    in_sz = fplutils.to3d(in_sz)
    in_sz = in_sz + (1,)

    inputs = Input(shape=in_sz)

    conv1 = Conv3D(32, (3, 3, 3), use_bias=False)(inputs) # 16x16x16
    conv1 = _bn_relu(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1) # 8x8x8

    conv2 = Conv3D(32, (3, 3, 3), use_bias=False)(pool1) # 6x6x6
    conv2 = _bn_relu(conv2)
    conv2 = Conv3D(32, (1, 1, 1), use_bias=False)(conv2) # 6x6x6
    conv2 = BatchNormalization()(conv2)
    crop_pool1 = Cropping3D(cropping=((1, 1), (1, 1), (1, 1)))(pool1)
    conv2 = add([crop_pool1, conv2])
    conv2 = Activation("relu")(conv2)

    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2) # 3x3x3

    conv3 = Conv3D(64, (3, 3, 3), use_bias=False)(pool2) # 1x1x1
    conv3 = _bn_relu(conv3)
    conv3 = Conv3D(64, (1, 1, 1), use_bias=False)(conv3) # 1x1x1
    pool2_shortcut = Conv3D(64, (1, 1, 1), use_bias=False)(pool2)
    crop_pool2 = Cropping3D(cropping=((1, 1), (1, 1), (1, 1)))(pool2_shortcut)
    conv3 = BatchNormalization()(conv3)
    conv3 = add([crop_pool2, conv3])
    conv3 = Activation("relu")(conv3)

    predictions = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv3)

    model = Model(inputs=inputs, outputs=predictions)

    return model

def unet_like(in_sz=None):
    '''
    construct a u-net style network
    '''
    in_sz = fplutils.to3d(in_sz)
    in_sz = in_sz + (1,)

    inputs = Input(shape=in_sz) # 18x18x18

    # down-sample
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', use_bias=False)(inputs) # 16x16x16
    #conv1 = _bn_relu(conv1)
    conv1 = Conv3D(32, (1, 1, 1), activation='relu', use_bias=False)(conv1) # 16x16x16
    #conv1 = _bn_relu(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1) # 8x8x8

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', use_bias=False)(pool1) # 6x6x6
    #conv2 = _bn_relu(conv2)
    conv2 = Conv3D(64, (1, 1, 1), activation='relu', use_bias=False)(conv2) # 6x6x6
    #conv2 = _bn_relu(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2) # 3x3x3

    conv3 = Conv3D(128, (1, 1, 1), activation='relu', use_bias=False)(pool2) # 3x3x3
    #conv3 = _bn_relu(conv3)

    # up-sample
    up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2]) # 6x6x6
    conv4 = Conv3D(64, (3, 3, 3), activation='relu', use_bias=False)(up4) # 4x4x4
    #conv4 = _bn_relu(conv4)
    conv4 = Conv3D(64, (1, 1, 1), activation='relu', use_bias=False)(conv4) # 4x4x4
    #conv4 = _bn_relu(conv4)

    crop_conv1 = Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(conv1)
    up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), crop_conv1]) # 8x8x8
    conv5 = Conv3D(32, (3, 3, 3), activation='relu', use_bias=False)(up5) # 6x6x6
    #conv5 = _bn_relu(conv5)
    conv5 = Conv3D(32, (1, 1, 1), activation='relu', use_bias=False)(conv5) # 6x6x6
    #conv5 = _bn_relu(conv5)

    predictions = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False)(conv5) # 6x6x6

    model = Model(inputs=inputs, outputs=predictions)
    return model
