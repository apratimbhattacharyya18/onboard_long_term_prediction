import h5py
import numpy
import json
import sys
import random
import operator
import tensorflow as tf


from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, merge, Lambda, RepeatVector, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras import backend as K
from keras.engine import Layer, InputSpec

class nDropout(Layer):
    def __init__(self, p, **kwargs):
        self.supports_masking = True
        self.p = p
        super(nDropout, self).__init__(**kwargs)

    def call(self, x, mask=None):

        inputs = x;

        input_shape = K.shape(x);

        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples,input_dim)
        initial_state = K.ones_like(initial_state)  # (samples, input_dim)

        mask = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        mask = K.sum(mask, axis=(0,1))  # (input_dim,)
        #mask = K.ones_like(K.expand_dims(mask, axis=0))  # (1,input_dim)
        mask = K.ones_like(mask)

        mask = K.dropout(mask, level=self.p)
       

        initial_state = tf.multiply(initial_state,mask);

        def step(inputs,states):
            mask = states[0];
            return tf.multiply(inputs,mask), [mask]

        ( _, outputs, _) = K.rnn(step, x, [initial_state]);

        return outputs

    def get_config(self):
        config = {'p': self.p}
        base_config = super(nDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
