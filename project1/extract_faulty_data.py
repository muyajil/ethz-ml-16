import os
#import numpy as np
import nibabel as nib
import cPickle
import sPickle
#import json

root_dir = os.path.dirname(os.path.realpath(__file__))

test_dir = "set_test"
train_dir = "set_train"

file_prefix_train = "train_"
file_prefix_test = "test_"

file_suffix = ".nii"

X_train = []

X_length = 0

image_15 = nib.load(os.path.join(root_dir, train_dir, file_prefix_train + str(15) + file_suffix)) #change 15 to test length mismatch
image_16 = nib.load(os.path.join(root_dir, train_dir, file_prefix_train + str(1) + file_suffix))

X = []

(height, width, depth, values) = image_16.shape
data_15 = image_15.get_data()
data_16 = image_16.get_data()
for a in range(height):
	for b in range(width):
		c_vec_15 = [num for sub in data_15[a][b] for num in sub]
		c_vec_16 = [num for sub in data_16[a][b] for num in sub]
		c_bool_16 = map(bool, c_vec_16)
		c_bitmask_16 = map(int, c_bool_16)
		c_vec = [m+n for (m,n) in zip(c_vec_15, c_bitmask_16)]
		c_filtered = filter(lambda k: k != 0, c_vec)
		c = map(lambda o: o-1, c_filtered)
		X.extend(c)
		#print "(" + str(a) + ", " + str(b) + ")"


print len(X)
#print "Finished"
#print X_train

#with open(r"pickle_train_data_faulty.pickle", "wb") as pout_file:
sPickle.s_dump(X, open("spickle_test_data_faulty.pickle", "w"))

#with open("json_train_data.json", "wb") as jout_file:
#	json.dump(X_train, jout_file)
