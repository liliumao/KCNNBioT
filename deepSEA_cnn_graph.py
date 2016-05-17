from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import geneData
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, activity_l2

import pdb

batch_size = 128
nb_epoch = 50

# number of convolutional filters to use
nb_filters = (300, 200, 200)
# size of pooling area for max pooling
nb_pool = (3,4,4)
# convolution kernel size
nb_conv = (19,11,7)

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

model.add_node(Convolution1D(nb_filter=nb_filters[0], filter_length=nb_conv[0], border_mode='valid', activation='relu', input_dim=ini_depth, input_length=seq_len, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)), name='conv1', input='input')
model.add_node(MaxPooling1D(pool_length=nb_pool[0]), name='pool1', input='conv1')

model.add_node(Convolution1D(nb_filters[1], nb_conv[1], activation='relu', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)), name='conv2', input='pool1')
model.add_node(MaxPooling1D(pool_length=nb_pool[1]), name='pool2', input='conv2')

model.add_node(Convolution1D(nb_filters[2], nb_conv[2], activation='relu', W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)), name='conv3', input='pool2')
model.add_node(MaxPooling1D(pool_length=nb_pool[2]), name='pool3', input='conv3')

model.add_node(Flatten(), name='flat', input='pool3')

model.add_node(Dense(100, activation='sigmoid'), name='den1', input='flat')
model.add_node(Dropout(0.3), name='dp1', input='den1')

model.add_node(Dense(100, activation='sigmoid'), name='den2', input='dp1')
model.add_node(Dropout(0.3), name='dp2', input='den2')

train_dict = {'input':X_train}
valid_dict = {'input':X_valid}
test_dict = {'input':X_test}
loss_dict = {}
for i in range(num_out):
	name = "out_"+str(i)
	model.add_node(Dense(1, activation='sigmoid'), name=name, input='dp2', create_output = True)
	loss_dict[name] = 'mse'
	train_dict[name] = y_train[i]
	valid_dict[name] = y_valid[i]
	test_dict[name] = y_test[i]
	# pdb.set_trace()

# sgd = SGD(lr=0.002, decay=0.0, momentum=0.98, nesterov=False)
model.compile(loss=loss_dict, optimizer='adadelta')

earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
model.fit(train_dict, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=valid_dict, callbacks=[])
score = model.evaluate(test_dict, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
