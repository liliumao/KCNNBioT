from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import geneData
from keras.models import Graph
from keras.optimizers import SGD
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

import pdb

default_args = {
	'nb_filters_c1': np.array([300]),
	'nb_filters_p1': np.array([3]),
	'nb_conv_1': np.array([19]),
	'nb_filters_c2': np.array([200]),
	'nb_filters_p2': np.array([4]),
	'nb_conv_2': np.array([11]),
	'nb_filters_c3': np.array([200]),
	'nb_filters_p3': np.array([4]),
	'nb_conv_3': np.array([7]),

	'nb_hunit_1': np.array([100]),
	'dropout_1': np.array([0.3]),
	'norm_w_1': np.array([8]),
	'nb_hunit_2': np.array([100]),
	'dropout_2': np.array([0.3]),
	'norm_w_2': np.array([8])
}

def deepSEA(newparams):
	params = default_args
	params.update(newparams)

	batch_size = 128
	nb_epoch = 30

	# number of convolutional filters to use
	nb_filters = (int(params['nb_filters_c1'][0]), int(params['nb_filters_c2'][0]), int(params['nb_filters_c3'][0]))
	# size of pooling area for max pooling
	nb_pool = (int(params['nb_filters_p1'][0]), int(params['nb_filters_p2'][0]), int(params['nb_filters_p3'][0]))
	# convolution kernel size
	nb_conv = (int(params['nb_conv_1'][0]), int(params['nb_conv_2'][0]), int(params['nb_conv_3'][0]))

	nb_hunit = (int(params['nb_hunit_1'][0]), int(params['nb_hunit_2'][0]))
	dropout = (params['dropout_1'][0], params['dropout_2'][0])
	norm_w = (params['norm_w_1'][0], params['norm_w_2'][0])
	
	# the data, shuffled and split between train and test sets
	X_train, y_train, X_valid, y_valid, X_test, y_test = geneData.loadData()

	# X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	# X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	# X_train = X_train.astype('float32')
	# X_test = X_test.astype('float32')
	# X_train /= 255
	# X_test /= 255
	# print('X_train shape:', X_train.shape)
	# print(X_train.shape[0], 'train samples')
	# print(X_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	# Y_train = np_utils.to_categorical(y_train, nb_classes)
	# Y_test = np_utils.to_categorical(y_test, nb_classes)

	# nb_targets = int(y_train.shape[1])
	nb_classes = 2

	seq_len = int(X_train.shape[1])
	ini_depth = int(X_train.shape[2])
	num_seq = int(X_train.shape[0])
	num_out = int(y_train.shape[0])

	# pdb.set_trace()

	model = Graph()

	model.add_input(name='input', input_shape=(seq_len, ini_depth))

	model.add_node(Convolution1D(nb_filter=nb_filters[0], filter_length=nb_conv[0], border_mode='valid', activation='relu', input_dim=ini_depth, input_length=seq_len), name='conv1', input='input')
	model.add_node(MaxPooling1D(pool_length=nb_pool[0]), name='pool1', input='conv1')

	model.add_node(Convolution1D(nb_filters[1], nb_conv[1], activation='relu'), name='conv2', input='pool1')
	model.add_node(MaxPooling1D(pool_length=nb_pool[1]), name='pool2', input='conv2')

	model.add_node(Convolution1D(nb_filters[2], nb_conv[2], activation='relu'), name='conv3', input='pool2')
	model.add_node(MaxPooling1D(pool_length=nb_pool[2]), name='pool3', input='conv3')

	model.add_node(Flatten(), name='flat', input='pool3')

	model.add_node(Dense(nb_hunit[0], activation='sigmoid', W_constraint = maxnorm(8)), name='den1', input='flat')
	# model.add_node(Dropout(dropout[0]), name='dp1', input='den1')

	model.add_node(Dense(nb_hunit[1], activation='sigmoid', W_constraint = maxnorm(8)), name='den2', input='den1')
	# model.add_node(Dropout(dropout[1]), name='dp2', input='den2')

	train_dict = {'input':X_train}
	valid_dict = {'input':X_valid}
	test_dict = {'input':X_test}
	loss_dict = {}
	for i in range(num_out):
		name = "out_"+str(i)
		model.add_node(Dense(1, activation='sigmoid'), name=name, input='den2', create_output = True)
		loss_dict[name] = 'mse'
		train_dict[name] = y_train[i]
		valid_dict[name] = y_valid[i]
		test_dict[name] = y_test[i]
		# pdb.set_trace()

	# sgd = SGD(lr=lr, decay=0.0, momentum=momentum, nesterov=False)
	model.compile(loss=loss_dict, optimizer='adadelta')

	# earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='min')
	model.fit(train_dict, batch_size=batch_size, nb_epoch=nb_epoch,
		show_accuracy=True, verbose=1, validation_data=valid_dict, callbacks=[])
	score = model.evaluate(test_dict, show_accuracy=True, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	return (1-score[1])


def main(job_id, params):
	print ('Anything printed here will end up in the output directory for job #%d' % job_id)
	print (params)
	return deepSEA(params)


if __name__ == "__main__":
	main(0, default_args)