"""defines keras models/network architectures to use for object
detection

"""

from flypylib import fplutils
from keras.models import Model
from keras.layers import Dropout, Activation, Conv3D, MaxPooling3D, Cropping3D, UpSampling3D
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import add, concatenate
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
import tensorflow as tf
import math
import keras.backend as K
import numpy as np

def _to_tensor(x, dtype):
    x = ops.convert_to_tensor(x)
    if x.dtype != dtype:
        x = math_ops.cast(x, dtype)
    return x

def masked_weighted_binary_crossentropy(y_true, y_pred):
    # Epsilon fuzz factor used throughout the codebase.
    _EPSILON = 10e-8
    mask = K.cast(K.not_equal(y_true, 2), K.floatx())
    y_true = y_true * mask
    y_pred = y_pred * mask
    epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    y_pred = clip_ops.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_pred = math_ops.log(y_pred / (1 - y_pred))
    cost = nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=100)
    return K.mean(cost, axis=-1)

def masked_binary_crossentropy(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 2), K.floatx())
    return K.mean(K.binary_crossentropy(y_pred * mask,
                                        y_true * mask), axis=-1)

def masked_focal_loss(y_true, y_pred):
    gamma = 2
    alpha = 1#0.25
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    mask = K.cast(K.less(y_true, 2), K.floatx())
    return -K.sum(alpha * mask * K.pow(1. - pt, gamma) * K.log(pt+K.epsilon()), axis=-1)

def lb0l1err(y_true, y_pred):
    mask = K.cast(K.equal(y_true, 0), K.floatx())
    err = y_pred * mask
    return K.sum(err) / K.maximum(K.sum(mask), 1)

def lb1l1err(y_true, y_pred):
    mask = K.cast(K.equal(y_true, 1), K.floatx())
    err = (1-y_pred) * mask
    return K.sum(err) / K.maximum(K.sum(mask), 1)

def masked_accuracy(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, 2), K.floatx())
    return K.mean(K.equal(y_true * mask,
                          K.round(y_pred * mask)), axis=-1)

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization()(input)
    return Activation("relu")(norm)

def baseline_model(in_sz = None):
    """returns simple baseline model
    """

    in_sz = fplutils.to3d(in_sz)
    in_sz = in_sz + (1,)

    inputs = Input(shape=in_sz)

    conv1 = Conv3D(32, (3,3,3), use_bias=False)(inputs)
    conv1 = _bn_relu(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)

    conv2 = Conv3D(32, (3,3,3), use_bias=False)(pool1)
    conv2 = _bn_relu(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

    conv3 = Conv3D(32, (3,3,3), use_bias=False)(pool2)
    conv3 = _bn_relu(conv3)

    full1 = Conv3D(64, (1,1,1), use_bias=False)(conv3)
    full1 = _bn_relu(full1)
    full1 = Dropout(0.5)(full1)

    predictions = Conv3D(1, (1,1,1), activation='sigmoid')(full1)

    model = Model(inputs=inputs, outputs=predictions)
    return model, (18, 7, 4), 102, None

def vgg_like(in_sz = None):
    """returns standard model based on VGG architecture"""

    in_sz = fplutils.to3d(in_sz)
    in_sz = in_sz + (1,)

    inputs = Input(shape=in_sz)

    conv1 = Conv3D(48, (3,3,3), use_bias=False)(inputs)
    conv1 = _bn_relu(conv1)
    conv1 = Conv3D(48, (1,1,1), use_bias=False)(conv1)
    conv1 = _bn_relu(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)

    conv2 = Conv3D(48, (3,3,3), use_bias=False)(pool1)
    conv2 = _bn_relu(conv2)
    conv2 = Conv3D(48, (1,1,1), use_bias=False)(conv2)
    conv2 = _bn_relu(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

    conv3 = Conv3D(48, (3,3,3), use_bias=False)(pool2)
    conv3 = _bn_relu(conv3)

    full1 = Conv3D(96, (1,1,1), use_bias=False)(conv3)
    full1 = _bn_relu(full1)
    full1 = Dropout(0.5)(full1)

    full2 = Conv3D(96, (1,1,1), use_bias=False)(full1)
    full2 = _bn_relu(full2)
    full2 = Dropout(0.5)(full2)

    predictions = Conv3D(1, (1,1,1), activation='sigmoid')(full2)

    model = Model(inputs=inputs, outputs=predictions)
    return model, (18, 7, 4), 102, None

def vgg_like2(in_sz = None):
    """returns standard model based on VGG architecture"""

    in_sz = fplutils.to3d(in_sz)
    in_sz = in_sz + (1,)

    inputs = Input(shape=in_sz)

    conv1 = Conv3D(48, (3,3,3), use_bias=False)(inputs)
    conv1 = _bn_relu(conv1)
    conv1 = Conv3D(48, (3,3,3), use_bias=False)(conv1)
    conv1 = _bn_relu(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)

    conv2 = Conv3D(48, (3,3,3), use_bias=False)(pool1)
    conv2 = _bn_relu(conv2)
    conv2 = Conv3D(48, (3,3,3), use_bias=False)(conv2)
    conv2 = _bn_relu(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

    conv3 = Conv3D(48, (3,3,3), use_bias=False)(pool2)
    conv3 = _bn_relu(conv3)

    full1 = Conv3D(96, (1,1,1), use_bias=False)(conv3)
    full1 = _bn_relu(full1)
    full1 = Dropout(0.5)(full1)

    full2 = Conv3D(96, (1,1,1), use_bias=False)(full1)
    full2 = _bn_relu(full2)
    full2 = Dropout(0.5)(full2)

    predictions = Conv3D(1, (1,1,1), activation='sigmoid')(full2)

    model = Model(inputs=inputs, outputs=predictions)
    return model, (24, 10, 4), 100, None

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
    return model, (18, 7, 4), 102, None

def unet_like(in_sz=18):
    '''
    construct a u-net style network
    '''
    in_sz = fplutils.to3d(in_sz)
    in_sz = in_sz + (1,)

    inputs = Input(shape=in_sz) # 18x18x18

    # down-sample
    conv1 = Conv3D(32, (3, 3, 3), use_bias=False)(inputs) # 16x16x16
    conv1 = _bn_relu(conv1)
    conv1 = Conv3D(32, (1, 1, 1), use_bias=False)(conv1) # 16x16x16
    conv1 = _bn_relu(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1) # 8x8x8

    conv2 = Conv3D(64, (3, 3, 3), use_bias=False)(pool1) # 6x6x6
    conv2 = _bn_relu(conv2)
    conv2 = Conv3D(64, (1, 1, 1), use_bias=False)(conv2) # 6x6x6
    conv2 = _bn_relu(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2) # 3x3x3

    conv3 = Conv3D(128, (1, 1, 1), use_bias=False)(pool2) # 3x3x3
    conv3 = _bn_relu(conv3)

    # up-sample
    up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2]) # 6x6x6
    conv4 = Conv3D(64, (3, 3, 3), use_bias=False)(up4) # 4x4x4
    conv4 = _bn_relu(conv4)
    conv4 = Conv3D(64, (1, 1, 1), use_bias=False)(conv4) # 4x4x4
    conv4 = _bn_relu(conv4)

    crop_conv1 = Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(conv1)
    up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), crop_conv1]) # 8x8x8
    conv5 = Conv3D(32, (3, 3, 3), use_bias=False)(up5) # 6x6x6
    conv5 = _bn_relu(conv5)
    conv5 = Conv3D(32, (1, 1, 1), use_bias=False)(conv5) # 6x6x6
    conv5 = _bn_relu(conv5)

    predictions = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False)(conv5) # 6x6x6

    model = Model(inputs=inputs, outputs=predictions)
    compile_args = {'loss': masked_binary_crossentropy,
                    'optimizer': 'adam',
                    'metrics': [masked_accuracy,
                                lb0l1err, lb1l1err]}
    return model, (18, 6, 1), 102, compile_args

def unet_like2(in_sz=24):
    '''
    construct a u-net style network
    '''
    in_sz = fplutils.to3d(in_sz)
    in_sz = in_sz + (1,)

    inputs = Input(shape=in_sz) # 24x24x24

    # down-sample
    conv1 = Conv3D(32, (3, 3, 3), use_bias=False)(inputs) # 22x22x22
    conv1 = _bn_relu(conv1)
    conv1 = Conv3D(32, (3, 3, 3), use_bias=False)(conv1) # 20x20x20
    conv1 = _bn_relu(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1) # 10x10x10

    conv2 = Conv3D(64, (3, 3, 3), use_bias=False)(pool1) # 8x8x8
    conv2 = _bn_relu(conv2)
    conv2 = Conv3D(64, (3, 3, 3), use_bias=False)(conv2) # 6x6x6
    conv2 = _bn_relu(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2) # 3x3x3

    conv3 = Conv3D(128, (1, 1, 1), use_bias=False)(pool2) # 3x3x3
    conv3 = _bn_relu(conv3)

    # up-sample
    up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), conv2]) # 6x6x6
    conv4 = Conv3D(64, (3, 3, 3), use_bias=False)(up4) # 4x4x4
    conv4 = _bn_relu(conv4)
    conv4 = Conv3D(64, (1, 1, 1), use_bias=False)(conv4) # 4x4x4
    conv4 = _bn_relu(conv4)

    crop_conv1 = Cropping3D(cropping=((6, 6), (6, 6), (6, 6)))(conv1)
    up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), crop_conv1]) # 8x8x8
    conv5 = Conv3D(32, (3, 3, 3), use_bias=False)(up5) # 6x6x6
    conv5 = _bn_relu(conv5)
    conv5 = Conv3D(32, (1, 1, 1), use_bias=False)(conv5) # 6x6x6
    conv5 = _bn_relu(conv5)

    predictions = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False)(conv5) # 6x6x6

    model = Model(inputs=inputs, outputs=predictions)
    compile_args = {'loss': masked_focal_loss, #masked_binary_crossentropy,
                    'optimizer': 'adam',
                    'metrics': [masked_accuracy,
                                lb0l1err, lb1l1err]}
    return model, (24, 9, 1), 100, compile_args

def unet_like3(in_sz=32):
    '''
    construct a u-net style network
    '''
    in_sz = fplutils.to3d(in_sz)
    in_sz = in_sz + (1,)

    inputs = Input(shape=in_sz) # 32^2

    # down-sample
    conv1 = Conv3D(32, (3, 3, 3), use_bias=False)(inputs) # 30
    conv1 = _bn_relu(conv1)
    conv1 = Conv3D(32, (3, 3, 3), use_bias=False)(conv1) # 28
    conv1 = _bn_relu(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1) # 14

    conv2 = Conv3D(64, (3, 3, 3), use_bias=False)(pool1) # 12
    conv2 = _bn_relu(conv2)
    conv2 = Conv3D(64, (3, 3, 3), use_bias=False)(conv2) # 10
    conv2 = _bn_relu(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2) # 5

    conv3 = Conv3D(128, (3, 3, 3), use_bias=False)(pool2) # 3
    conv3 = _bn_relu(conv3)
    conv3 = Conv3D(128, (1, 1, 1), use_bias=False)(conv3) # 3
    conv3 = _bn_relu(conv3)

    # up-sample
    crop_conv2 = Cropping3D(cropping=((2, 2), (2, 2), (2, 2)))(conv2)
    up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), crop_conv2]) # 6
    conv4 = Conv3D(64, (3, 3, 3), use_bias=False)(up4) # 4
    conv4 = _bn_relu(conv4)
    conv4 = Conv3D(64, (1, 1, 1), use_bias=False)(conv4) # 4
    conv4 = _bn_relu(conv4)

    crop_conv1 = Cropping3D(cropping=((10, 10), (10, 10), (10, 10)))(conv1)
    up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), crop_conv1]) # 8x8x8
    conv5 = Conv3D(32, (3, 3, 3), use_bias=False)(up5) # 6
    conv5 = _bn_relu(conv5)
    conv5 = Conv3D(32, (1, 1, 1), use_bias=False)(conv5) # 6
    conv5 = _bn_relu(conv5)

    predictions = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False)(conv5) # 6

    model = Model(inputs=inputs, outputs=predictions)
    compile_args = {'loss': masked_focal_loss, #masked_binary_crossentropy,
                    'optimizer': 'adam',
                    'metrics': [masked_accuracy,
                                lb0l1err, lb1l1err]}
    return model, (32, 13, 1), 100, compile_args


def unet_like4(in_sz=40):
    '''
    construct a u-net style network
    '''
    in_sz = fplutils.to3d(in_sz)
    in_sz = in_sz + (1,)

    inputs = Input(shape=in_sz) # 40^2

    # down-sample
    conv1 = Conv3D(32, (3, 3, 3), use_bias=False)(inputs) # 38
    conv1 = _bn_relu(conv1)
    conv1 = Conv3D(32, (3, 3, 3), use_bias=False)(conv1) # 36
    conv1 = _bn_relu(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1) # 18

    conv2 = Conv3D(64, (3, 3, 3), use_bias=False)(pool1) # 16
    conv2 = _bn_relu(conv2)
    conv2 = Conv3D(64, (3, 3, 3), use_bias=False)(conv2) # 14
    conv2 = _bn_relu(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2) # 7

    conv3 = Conv3D(128, (3, 3, 3), use_bias=False)(pool2) # 5
    conv3 = _bn_relu(conv3)
    conv3 = Conv3D(128, (3, 3, 3), use_bias=False)(conv3) # 3
    conv3 = _bn_relu(conv3)

    # up-sample
    crop_conv2 = Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(conv2)
    up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), crop_conv2]) # 6
    conv4 = Conv3D(64, (3, 3, 3), use_bias=False)(up4) # 4
    conv4 = _bn_relu(conv4)
    conv4 = Conv3D(64, (1, 1, 1), use_bias=False)(conv4) # 4
    conv4 = _bn_relu(conv4)

    crop_conv1 = Cropping3D(cropping=((14, 14), (14, 14), (14, 14)))(conv1)
    up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), crop_conv1]) # 8x8x8
    conv5 = Conv3D(32, (3, 3, 3), use_bias=False)(up5) # 6
    conv5 = _bn_relu(conv5)
    conv5 = Conv3D(32, (1, 1, 1), use_bias=False)(conv5) # 6
    conv5 = _bn_relu(conv5)

    predictions = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False)(conv5) # 6

    model = Model(inputs=inputs, outputs=predictions)
    compile_args = {'loss': masked_focal_loss, #masked_binary_crossentropy,
                    'optimizer': 'adam',
                    'metrics': [masked_accuracy,
                                lb0l1err, lb1l1err]}
    return model, (40, 17, 1), 100, compile_args


def unet_like4b(in_sz=40):
    '''
    construct a u-net style network
    '''
    in_sz = fplutils.to3d(in_sz)
    in_sz = in_sz + (1,)

    inputs = Input(shape=in_sz) # 40^2

    # down-sample
    conv1 = Conv3D(32, (3, 3, 3), use_bias=False)(inputs) # 38
    conv1 = _bn_relu(conv1)
    conv1 = Conv3D(32, (3, 3, 3), use_bias=False)(conv1) # 36
    conv1 = _bn_relu(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1) # 18

    conv2 = Conv3D(64, (3, 3, 3), use_bias=False)(pool1) # 16
    conv2 = _bn_relu(conv2)
    conv2 = Conv3D(32, (1, 1, 1), use_bias=False)(conv2) # 14
    conv2 = _bn_relu(conv2)
    conv2 = Conv3D(64, (3, 3, 3), use_bias=False)(conv2) # 14
    conv2 = _bn_relu(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2) # 7

    conv3 = Conv3D(48, (1, 1, 1), use_bias=False)(pool2) # 7
    conv3 = _bn_relu(conv3)
    conv3 = Conv3D(128, (3, 3, 3), use_bias=False)(conv3) # 5
    conv3 = _bn_relu(conv3)
    conv3 = Conv3D(48, (1, 1, 1), use_bias=False)(conv3) # 5
    conv3 = _bn_relu(conv3)
    conv3 = Conv3D(128, (3, 3, 3), use_bias=False)(conv3) # 3
    conv3 = _bn_relu(conv3)
    conv3 = Conv3D(48, (1, 1, 1), use_bias=False)(conv3) # 5
    conv3 = _bn_relu(conv3)

    # up-sample
    crop_conv2 = Cropping3D(cropping=((4, 4), (4, 4), (4, 4)))(conv2)
    up4 = concatenate([UpSampling3D(size=(2, 2, 2))(conv3), crop_conv2]) # 6
    conv4 = Conv3D(64, (3, 3, 3), use_bias=False)(up4) # 4
    conv4 = _bn_relu(conv4)
    conv4 = Conv3D(64, (1, 1, 1), use_bias=False)(conv4) # 4
    conv4 = _bn_relu(conv4)

    crop_conv1 = Cropping3D(cropping=((14, 14), (14, 14), (14, 14)))(conv1)
    up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), crop_conv1]) # 8x8x8
    conv5 = Conv3D(32, (3, 3, 3), use_bias=False)(up5) # 6
    conv5 = _bn_relu(conv5)
    conv5 = Conv3D(32, (1, 1, 1), use_bias=False)(conv5) # 6
    conv5 = _bn_relu(conv5)

    predictions = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False)(conv5) # 6

    model = Model(inputs=inputs, outputs=predictions)
    compile_args = {'loss': masked_focal_loss, #masked_binary_crossentropy,
                    'optimizer': 'adam',
                    'metrics': [masked_accuracy,
                                lb0l1err, lb1l1err]}
    return model, (40, 17, 1), 100, compile_args


def unet_like_vol(in_sz=62):
    """construct a u-net style network
    """
    in_sz = fplutils.to3d(in_sz)
    in_sz = in_sz + (1,)

    inputs = Input(shape=in_sz) # 62x62x62

    # down-sample
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', use_bias=False)(inputs) # 60x60x60
    #conv1 = _bn_relu(conv1)
    conv1 = Conv3D(16, (1, 1, 1), activation='relu', use_bias=False)(conv1) # 60x60x60
    #conv1 = _bn_relu(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1) # 30x30x30

    conv2 = Conv3D(32, (3, 3, 3), activation='relu', use_bias=False)(pool1) # 28x28x28
    #conv2 = _bn_relu(conv2)
    conv2 = Conv3D(32, (1, 1, 1), activation='relu', use_bias=False)(conv2) # 28x28x28
    #conv2 = _bn_relu(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2) # 14x14x14

    conv3 = Conv3D(64, (1, 1, 1), activation='relu', use_bias=False)(pool2) # 14x14x14
    #conv3 = _bn_relu(conv3)

    conv2_sz = tuple((math.floor((ss-2)/2)-2) for ss in in_sz[0:3])
    conv3_sz = tuple(math.floor(ss/2)*2 for ss in conv2_sz) # size after up-sampling conv3
    #crop_conv2 = Cropping3D(cropping=(
    #    (0, conv2_sz[0]-conv3_sz[0]),
    #    (0, conv2_sz[1]-conv3_sz[1]),
    #    (0, conv2_sz[2]-conv3_sz[2])))(conv2)
    # up-sample
    up_conv3 = UpSampling3D(size=(2, 2, 2))(conv3)
    up4 = concatenate([up_conv3, conv2]) # 28x28x28
    conv4 = Conv3D(64, (3, 3, 3), activation='relu', use_bias=False)(up4) # 26x26x26
    #conv4 = _bn_relu(conv4)
    conv4 = Conv3D(64, (1, 1, 1), activation='relu', use_bias=False)(conv4) # 26x26x26
    #conv4 = _bn_relu(conv4)

    conv1_sz = tuple((ss-2) for ss in in_sz[0:3])
    conv4_sz = tuple((ss-2)*2 for ss in conv3_sz)

    crop_conv1 = Cropping3D(cropping=(
        (math.floor((conv1_sz[0]-conv4_sz[0])/2), math.ceil((conv1_sz[0]-conv4_sz[0])/2)),
        (math.floor((conv1_sz[1]-conv4_sz[1])/2), math.ceil((conv1_sz[1]-conv4_sz[1])/2)),
        (math.floor((conv1_sz[2]-conv4_sz[2])/2), math.ceil((conv1_sz[2]-conv4_sz[2])/2))))(conv1)
    up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), crop_conv1]) # 52x52x52
    conv5 = Conv3D(32, (3, 3, 3), activation='relu', use_bias=False)(up5) # 50x50x50
    #conv5 = _bn_relu(conv5)
    conv5 = Conv3D(32, (1, 1, 1), activation='relu', use_bias=False)(conv5) # 50x50x50
    #conv5 = _bn_relu(conv5)

    predictions = Conv3D(1, (1, 1, 1), activation='sigmoid', use_bias=False)(conv5) # 50x50x50
    model = Model(inputs=inputs, output=predictions)
    compile_args = {'loss': masked_weighted_binary_crossentropy,
                    'optimizer': 'adam',
                    'metrics': ['masked_accuracy']}
    return model, (62, 6, 1), 102, compile_args
