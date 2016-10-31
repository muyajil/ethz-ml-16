import os
import numpy as np
import nibabel as nib
import sPickle # -> https://github.com/pgbovine/streaming-pickle
import sys
from scipy import ndimage
#from skimage import filters as skifilter
#from skimage import exposure as skex

'''
Preprocessing with FAST () to correcting spatial intensity variations. Then run this script on the "restored input" output of FAST.

Fist the MRI picture were processed with the FAST command-line program. This
yields different outputfiles, we only use the "Restored input" image, which is
again a MRI picture but corrected for spatial intensity variations (bias fields).
Website for more information on FAST: http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FAST

After this step, the new MRI images must be saved in the folder
/set_[test, train]_segmented/[train, test]_i_restore.nii.gz for i in N

After that, this script can be run to extract the data: Our best submission was
with a simplified histogram of the voxel intesities over the whole picture. This
simpified histogram maps the values of the voxels to 100 buckets.

Since we cannot easily provide the FAST test data or a script computing these
new MRI files, we supply the data file AFTER the execution of this script with
our submission.
'''

bool_euler = True # deactivates interaction with user and just computes both test and train with a successful configuration

# debug parameter
debug = True  # just computes the first image of whatever set is selected
number_test_images = 1

modes = ["avg", "vector", "grad", "custom"]
mode = ""

root_dir = os.path.dirname(os.path.realpath(__file__))

test_dir = "set_test_segmented"
train_dir = "set_train_segmented"

file_prefix_train = "train_"
file_prefix_test = "test_"

file_suffix = "_restore.nii.gz"
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

def process_img(img, filtered):
    (height, width, depth) = img.shape
    data = img.get_data()
    X_3d = []
    for a in range(height):
        X_2d = []
        for b in range(width):
            X_1d = [num for num in data[a][b]]
            if filtered:
                X_1d = filter(lambda a: a != 0, X_1d)
            if len(X_1d) > 0:
                X_2d.append(X_1d)
        if len(X_2d) > 0:
            X_3d.append(X_2d)
    del data
    return X_3d

def process_faulty_img(img):
    image_correct = nib.load(os.path.join(root_dir, train_dir, file_prefix_train + str(1) + file_suffix))
    (height, width, depth) = image_correct.shape
    data = img.get_data()
    data_correct = image_correct.get_data()
    X_3d = []
    for a in range(height):
        X_2d = []
        for b in range(width):
            X_1d = [num for num in data[a][b]]
            c_vec_correct = [num for num in data_correct[a][b]]
            c_bool_correct = map(bool, c_vec_correct)
            c_bitmask_correct = map(int, c_bool_correct)
            X_1d = [m+n for (m,n) in zip(X_1d, c_bitmask_correct)]
            X_1d_filtered = filter(lambda k: k != 0, X_1d)
            X_1d_final = map(lambda o: o-1, X_1d_filtered)
            if len(X_1d_final) > 0:
                X_2d.append(X_1d_final)
        if len(X_2d) > 0:
            X_3d.append(X_2d)
    del [data, data_correct, image_correct]
    return X_3d

def extract_data(kind, current_number, total_datapoints, histogram=True):
    X_length = 0

    if kind == "train":
        image_num = data_points_train # train
    elif kind == "test":
        image_num = data_points_test # test
    else:
        print "error, not correct test/train for kind"
    if debug:
        image_num = number_test_images

    out_file = open(mode + "_" + kind + ".pickle", 'w')

    if mode == "custom":
        out_file4 = open('short_histogram100_' + kind + '.pickle', 'w')
    
    out_file_histo = open(mode + '_histogram_' + kind + '.pickle', 'w')

    for i in range(image_num):
        if kind == "train":
            image = nib.load(os.path.join(root_dir, train_dir, file_prefix_train + str(i+1) + file_suffix))
        elif kind == "test":
            image = nib.load(os.path.join(root_dir, test_dir, file_prefix_test + str(i+1) + file_suffix))

        if mode == "avg":
            X_3d = process_img(image, True)
            X = [(sum(vec) / float(len(vec))) for matrix in X_3d for vec in matrix]
            X_histogram = [0] * 1000
        elif mode == "vector":
            X_3d = process_img(image, True)
            X = [elm for matrix in X_3d for vec in matrix for elm in vec]
            X_histogram = [0] * 1000
        elif mode == "custom":
            X_3d = process_img(image, True)
            X = [elm for matrix in X_3d for vec in matrix for elm in vec]
            X_histogram = [0] * 5000
        else:
            print "unexpected error: unsupported mode. exiting.."
            exit()


        if(i == 0):
            X_length = len(X)
        else:
            if(len(X) != X_length):
                X_3d = process_faulty_img(image)
                if mode == "avg":
                    X = [(sum(vec) / float(len(vec))) for matrix in X_3d for vec in matrix]
                elif mode == "vector":
                    X = [elm for matrix in X_3d for vec in matrix for elm in vec]
                elif mode == "custom":
                    X = [elm for matrix in X_3d for vec in matrix for elm in vec]
                else:
                    print "unexpecter error: unsupported mode. exiting.."

        sPickle.s_dump_elt(X, out_file)

        # make histogram

        for elm in X:
            if elm > 5000:
                print elm
            else:
                X_histogram[int(elm)] += 1
        sPickle.s_dump_elt(X_histogram, out_file_histo)

        if mode == "custom":
            X_sh = [0] * 100
            index = 0
            for itm in X_histogram:
                X_sh[i / 60] += itm
                index += 1
            sPickle.s_dump_elt(X_sh, out_file4)

        current_number += 1
        print "Finished file " + str(i+1) + "; " + "%.2f" % ((current_number/float(total_datapoints)) * 100) + "%"
        del image

    out_file.close()
    out_file_histo.close()
    out_file4.close()

    return current_number


# ask which to run
print "Starting to extract data from MRI pictures.."
kinds = []
if bool_euler: # this option allows the script to be run without user interaction
    # these options will run the script for both the test and train set and extract
    # a histogram of the voxels.
    btest = True
    btrain = True
    mode = "custom"
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
