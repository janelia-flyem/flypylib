"""parallelize inference over multiple gpus

copied from keras-extras/utils:
https://github.com/kuza55/keras-extras

updated to conform to Keras 2 API

hard-coded tf.slice start and size in get_slice, assuming fixed
batch_size at run time of 1 per gpu; necessary to TF to infer shape
size, otherwise causes error when constructing outputs = model(inputs)

"""

from keras.layers import concatenate
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf

def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = data.get_shape().as_list()[1:]
        # assume at run time batch_size is 1 per gpu
        size  = [1,]   + shape
        start = [idx,] + len(shape)*[0,]
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(concatenate(outputs, axis=0))

        return Model(inputs=model.inputs, outputs=merged)
