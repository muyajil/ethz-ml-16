'''
Run on Euler:

>>ssh netzID@euler.ethz.ch
>>module load python/2.7.6

use git to download code, scp to upload data

>>python -m pip install --user nibabel
>>bsub -n 4 -R "rusage[mem=2048, scratch=10000]" -N -W 04:00 python read_data.py
	-n: num of processor cores
	-R mem: RAM per core
	-R scratch: Disk space per core
	-N: notify when done with mail
	-W: max runtime

observe jobs:
>>bjobs // status of all jobs
>>bbjobs JOB_ID // stats of specific job
>>bkill JOB_ID // kill job

Output fill be written into lsf.*
'''

import os
import numpy as np
import nibabel as nib
import sPickle # -> https://github.com/pgbovine/streaming-pickle
import sys

bool_euler = False # deactivates interaction with user and just computes both test and train
debug = True  # just computes the first image of whatever set is selected

def query_yes_no(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


root_dir = os.path.dirname(os.path.realpath(__file__))

test_dir = "set_test"
train_dir = "set_train"

file_prefix_train = "train_"
file_prefix_test = "test_"

file_suffix = ".nii"
data_points_test = 138
data_points_train = 278

# ask which to run
kinds = []
total_datapoints = 0
if bool_euler:
	btest = True
	btrain = True
else:
	btest = query_yes_no("Run for test set?")
	btrain = query_yes_no("Run for train set?")

if btest:
	kinds.append("test")
	total_datapoints += data_points_test
if btrain:
	kinds.append("train")
	total_datapoints += data_points_train

print "Running " + str(kinds) + " in " + ("debug" if debug else "normal") + " mode"

current_number = 0
for kind in kinds:
	X_length = 0

	if kind == "train":
		data_points = data_points_train # train
	elif kind == "test":
		data_points = data_points_test # test
	else:
		print "error, not correct test/train for kind"
	if debug:
		data_points = 1
		total_datapoints = 2

	out_file = open("spickle_" + kind + "_data.pickle", 'w')

	for i in range(data_points):
		if kind == "train":
			image = nib.load(os.path.join(root_dir, train_dir, file_prefix_train + str(i+1) + file_suffix))
		elif kind == "test":
			image = nib.load(os.path.join(root_dir, test_dir, file_prefix_test + str(i+1) + file_suffix))

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
				print kind + " " + str(i + 1) + ": Length mismatch!" + " current: " + str(len(X)) + " vs. first: " + str(X_length)
		sPickle.s_dump_elt(X, out_file)

		current_number += 1
		print "Finished file " + str(i+1) + "; " + "%.2f" % ((current_number/float(total_datapoints)) * 100) + "%"
		del image

	out_file.close()

	print "done with everything, all good."
