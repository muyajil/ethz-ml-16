import os
import numpy as np
import nibabel as nib

root_dir = os.path.dirname(os.path.realpath(__file__))

test_dir = "set_test"
train_dir = "set_train"

file_prefix_train = "train_"
file_prefix_test = "test_"

file_suffix = ".nii"

targets_file = "targets.csv"
targets = []

with open(targets_file, "r") as file:
	targets = file.readlines()

Y_train = map(int, targets)
X_train = []

X_length = 0

for i in range(278):
	image = nib.load(os.path.join(root_dir, train_dir, file_prefix_train + str(i+1) + file_suffix))	
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
			print str(i) + ": Length mismatch!" + " " + str(len(X)) + " vs. " + X_length
				
	X_train.append(X)
	print "Finished file " + str(i+1) + "; " + "%.2f" % (((i+1)/278.0) * 100) + "%"
	del image

#print "Finished"
#print X_train

file = open('./train_data.txt', 'w')
print >> file, X_train
