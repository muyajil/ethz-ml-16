import os
import numpy as np
import nibabel as nib
import cPickle
import sPickle # -> https://github.com/pgbovine/streaming-pickle
import json

root_dir = os.path.dirname(os.path.realpath(__file__))

test_dir = "set_test"
train_dir = "set_train"

file_prefix_train = "train_"
file_prefix_test = "test_"

file_suffix = ".nii"

targets_file = "targets.csv"
targets = []

#with open(targets_file, "r") as file:
#	targets = file.readlines()

#Y_train = map(int, targets)

#X_train = []

X_length = 0

data_points = 278 # train
#data_points = 138 # test
#data_points = 3

out_file = open('cont_spickle_train_data.pickle', 'w')

for i in range(data_points):
	image = nib.load(os.path.join(root_dir, train_dir, file_prefix_train + str(i+1) + file_suffix))
	#image = nib.load(os.path.join(root_dir, test_dir, file_prefix_test + str(i+1) + file_suffix))
	(height, width, depth, values) = image.shape
	data = image.get_data()
	X = []
	for a in range(height):
		for b in range(width):
			c_vec = [num for sub in data[a][b] for num in sub]
			c = filter(lambda a: a != 0, c_vec)
			X.extend(c)
	if(i == 0):
		X_length = len(X)
	else:
		if(len(X) != X_length):
			print str(i + 1) + ": Length mismatch!" + " current: " + str(len(X)) + " vs. first: " + str(X_length)
	sPickle.s_dump_elt(X, out_file)
	#X_train.append(X)

	print "Finished file " + str(i+1) + "; " + "%.2f" % (((i+1)/float(data_points)) * 100) + "%"
	del image

#print "Finished"
#print X_train

#file = open('./train_data.txt', 'w')
#print >> file, X_train
#close(file)

#with open(r"pickle_test_data.pickle", "wb") as pout_file:
#with open(r"pickle_train_data.pickle", "w") as pout_file:
#	sPickle.s_dump(X_train, pout_file)
#	del X_train

#sPickle.s_dump(X_train, open('spickle_train_data.pickle', 'w'))

#with open("json_train_data.json", "wb") as jout_file:
#	json.dump(X_train, jout_file)
