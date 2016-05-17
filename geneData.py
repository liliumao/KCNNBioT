import numpy as np
import pdb
import sys

def loadData(path="learn_cd4_graph.npz"):
	data = np.load(path)
	# pbd.set_trace()
	X_train = data['train_in']
	y_train = data['train_out']
	X_valid = data['valid_in']
	y_valid = data['valid_out']
	X_test = data['test_in']
	y_test = data['test_out']
	return X_train, y_train, X_valid, y_valid, X_test, y_test