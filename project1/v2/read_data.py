import os
import numpy as np
import nibabel as nib
import sPickle # -> https://github.com/pgbovine/streaming-pickle
import sys

bool_euler = False # deactivates interaction with user and just computes both test and train
debug = False  # just computes the first image of whatever set is selected
number_test_images = 1

modes = ["avg", "vector"]
mode = ""

root_dir = os.path.dirname(os.path.realpath(__file__ + "/../"))

test_dir = "set_test"
train_dir = "set_train"

file_prefix_train = "train_"
file_prefix_test = "test_"

file_suffix = ".nii"
data_points_test = 138
data_points_train = 278

# for meaningfull feedback
current_number = 0
total_datapoints = 0

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
def process_img(img):
	(height, width, depth, values) = img.shape
	data = img.get_data()
	X = []
	for a in range(height):
		for b in range(width):
			c_vec = [num for sub in data[a][b] for num in sub]
			c = filter(lambda a: a != 0, c_vec)
			if len(c) > 0:
				X.append(c)
	del data
	return X

def process_faulty_img(img):
	image_correct = nib.load(os.path.join(root_dir, train_dir, file_prefix_train + str(1) + file_suffix))
	(height, width, depth, values) = image_correct.shape
	data = img.get_data()
	data_correct = image_correct.get_data()
	X = []
	for a in range(height):
		for b in range(width):
			c_vec = [num for sub in data[a][b] for num in sub]
			c_vec_correct = [num for sub in data_correct[a][b] for num in sub]
			c_bool_correct = map(bool, c_vec_correct)
			c_bitmask_correct = map(int, c_bool_correct)
			c_vec = [m+n for (m,n) in zip(c_vec, c_bitmask_correct)]
			c_filtered = filter(lambda k: k != 0, c_vec)
			c = map(lambda o: o-1, c_filtered)
			if len(c) > 0:
				X.append(c)
	del [data, data_correct, image_correct]
	return X			

def extract_data(kind, current_number, total_datapoints):
	X_length = 0

	if kind == "train":
		image_num = data_points_train # train
	elif kind == "test":
		image_num = data_points_test # test
	else:
		print "error, not correct test/train for kind"
	if debug:
		image_num = number_test_images

	out_file = open("spickle_" + kind + "_" + mode + "_data_clean.pickle", 'w')

	for i in range(image_num):
		if kind == "train":
			image = nib.load(os.path.join(root_dir, train_dir, file_prefix_train + str(i+1) + file_suffix))
		elif kind == "test":
			image = nib.load(os.path.join(root_dir, test_dir, file_prefix_test + str(i+1) + file_suffix))
		
		X_matrix = process_img(image)
		

		if mode == "avg":
			X = [sum(elm) / float(len(elm)) for elm in X_matrix]
		elif mode == "vector":
			X = [item for sublist in X_matrix for item in sublist]
		else:
			print "unexpected error: unsupported mode. exiting.."
			exit()


		if(i == 0):
			X_length = len(X)
		else:
			if(len(X) != X_length):
				X_matrix = process_faulty_img(image)
				if mode == "avg":
					X = [sum(elm) / float(len(elm)) for elm in X_matrix]
				elif mode == "vector":
					X = [item for sublist in X_matrix for item in sublist]
				else:
					print "unexpecter error: unsupported mode. exiting.."

		sPickle.s_dump_elt(X, out_file)

		current_number += 1
		print "Finished file " + str(i+1) + "; " + "%.2f" % ((current_number/float(total_datapoints)) * 100) + "%"
		del image

	out_file.close()
	return current_number


# ask which to run
kinds = []
if bool_euler:
	btest = True
	btrain = True
	mode = modes[0]
else:
	while True:
		sys.stdout.write("Chose one of the following modes: " + str(modes) + "  ")
		choice = raw_input().lower()
		if choice in modes:
			mode = choice
			break
		elif choice == "e":
			exit()
		else:
			print "Not a valid mode, please choose a valid option or exit with 'e'"

	btest = query_yes_no("Extract data from test set?")
	btrain = query_yes_no("Extract data from train set?")

if btest:
	kinds.append("test")
	total_datapoints += data_points_test
if btrain:
	kinds.append("train")
	total_datapoints += data_points_train

print "Running " + str(kinds) + " in " + ("debug" if debug else "normal") + " mode"

for kind in kinds:
	if debug:
		total_datapoints = len(kinds) * number_test_images

	current_number += extract_data(kind, current_number, total_datapoints)

print "done with everything, all good."
