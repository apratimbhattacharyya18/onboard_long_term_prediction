import h5py
import numpy as np
import json
import sys
import random
import operator
import tensorflow as tf


from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, merge, Lambda, RepeatVector, Dropout
from keras.layers.wrappers import TimeDistributed
from keras import regularizers
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras import backend as K
from keras.engine import Layer, InputSpec

from new_recurrent import dLSTM
from time_distributed_dropout import nDropout

count = 0;

def get_lossfunc(true, pred):
	x1_data = true[:,:,0]
	y1_data = true[:,:,1]
	x2_data = true[:,:,2]
	y2_data = true[:,:,3]
	z_mux1 = pred[:,:,0]
	z_muy1 = pred[:,:,1]
	z_mux2 = pred[:,:,2]
	z_muy2 = pred[:,:,3]
	z_sx = pred[:,:,4]
	z_sy = pred[:,:,5]

	result1 = tf.multiply(tf.square(tf.subtract(x1_data, z_mux1)), K.exp(-z_sx)) + tf.multiply(tf.square(tf.subtract(y1_data, z_muy1)), K.exp(-z_sy))

	result2 = tf.multiply(tf.square(tf.subtract(x2_data, z_mux2)), K.exp(-z_sx)) + tf.multiply(tf.square(tf.subtract(y2_data, z_muy2)), K.exp(-z_sy))

	reg = tf.add(tf.reduce_mean((z_sx)), tf.reduce_mean((z_sy)))

	return tf.add( tf.divide(tf.add(tf.reduce_mean(result1), tf.reduce_mean(result2)), tf.constant(4.0, dtype=tf.float32, shape=(1, 1))), reg)


def get_center( bbox ):
	return [ (float(bbox[0]) + float(bbox[2]))/(2*2048), (float(bbox[1]) + float(bbox[3]))/(2*1024) ];

def get_bbox_centers( bboxes ):
	bbox_centers = [];
	for bbox in bboxes:
		bbox_centers.append( get_center(bbox) );
	return bbox_centers

def slice_tracks( bboxes, length ):
	bboxes = [ bboxes[i:i+length] for i in xrange(len(bboxes) - length + 1) ]
	#timestamps = [ timestamps[i:i+length] for i in xrange(len(timestamps) - length + 1) ]
	return bboxes

def get_diff_array( arr ):
	arr = arr[:,0:4]
	arr1 = arr[1:];
	arr2 = arr[0];
	return np.subtract(arr1,arr2);

def training_example( arr, in_frames ):
	x = arr[0:in_frames-1];
	y = arr[in_frames-1:];
	x = np.reshape( x, (in_frames-1,4))
	return ( x, y)


def get_modeld(input_shape1,out_seq):
	input1 = Input(shape=input_shape1)

	l2_reg = regularizers.l2(0.0001);

	decoder_1 = TimeDistributed(Dense(64, activation='relu', kernel_regularizer = l2_reg))(input1)
	decoder_1 = nDropout(0.10)(decoder_1)
	decoder_1 = dLSTM(128, implementation = 1, kernel_regularizer = l2_reg, recurrent_regularizer = l2_reg, bias_regularizer = l2_reg)(decoder_1);
	decoder_1 = RepeatVector(out_seq)(decoder_1);
	decoder_1_ = dLSTM(128, kernel_regularizer = l2_reg, recurrent_regularizer = l2_reg, bias_regularizer = l2_reg, implementation = 1, return_sequences=True)(decoder_1);
	decoder_1_m = TimeDistributed(Dense(4))(decoder_1_)
	decoder_1_v = TimeDistributed(Dense(2, activation='relu'))(decoder_1_)
	decoder_1 = merge([decoder_1_m,decoder_1_v], mode='concat', concat_axis=2)
	print decoder_1._keras_shape

	model = Model(input= [input1], output=decoder_1)
	model.compile(optimizer = 'adam', loss = get_lossfunc)

	return model	

in_frames = 8;
out_frames = 15;
shuffle_range = 5120*4;
batch_size = 128;
min_seq_len = 12;

source_f = h5py.File('./tracks_train.h5','r');


for seq_len in xrange(min_seq_len,in_frames+out_frames+1):
	data_X = [];
	data_Y = [];

	model = get_modeld( (in_frames-1,4), seq_len - in_frames );
	if seq_len > min_seq_len:
		model.load_weights('ver_len_128_8.h5');

	for track_key in source_f:
		curr_track = json.loads(source_f[track_key][()])
		bboxes = curr_track['bboxes']
		if len(bboxes) >= in_frames + out_frames:
			first_frame = curr_track['firstFrame'];
			last_frame = curr_track['lastFrame'];
			#bboxes = get_bbox_centers( bboxes );
			bbox_slices = slice_tracks( bboxes, seq_len )
			for bbox_slice in bbox_slices:
				diff_array = get_diff_array( np.array(bbox_slice) )
				( x, y) = training_example( diff_array, in_frames );
				data_X.append(x);
				data_Y.append(y);
				
	data_X = np.array(data_X);
	data_Y = np.array(data_Y);
	print(data_X.shape)
	print(data_Y.shape)
	if seq_len >= 12 and seq_len < 19:
		model.fit([data_X],data_Y,batch_size=64,nb_epoch=35,verbose=1);
	else:
		model.fit([data_X],data_Y,batch_size=64,nb_epoch=25,verbose=1);
	model.save('ver_len_128_8.h5');
