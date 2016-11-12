import os
import sys
import numpy as np
import nibabel as nib
import sPickle # -> https://github.com/pgbovine/streaming-pickle
import sklearn
from multiprocessing import Pool

# Execution flags
SUBMISSION_VERSION = False # True for final submission -> no output or customizability
DEBUG = True
debug_num = 3

# constants
data_points_train = 278
data_points_test = 138
res_folder = "Out/"

def load_img(kind, number):
    img = nib.load("set_" + kind + "/" + kind + "_" + str(number) + ".nii")
    (height, width, depth, values) = img.shape
    data = img.get_data()
    X_3d = []
    for a in range(height):
        X_2d = []
        for b in range(width):
            X_1d = [num for sub in data[a][b] for num in sub]
            X_2d.append(X_1d)
        X_3d.append(X_2d)
    del data
    return X_3d

def process_img(kind, index):
    X = []
    X_3d = load_img(kind, index)

    # process img and store in 'X'
    # TODO

    return X

def process_img_train(index):
    X_train = process_img("train", index)
    print "Finished reading file train_" + str(index) + "; " + "%.2f" % ((index/float(data_points_train)) * 100) + "%"
    return X_train

def process_img_test(index):
    X_test = process_img("test", index)
    print "Finished reading file test_" + str(index) + "; " + "%.2f" % ((index/float(data_points_test)) * 100) + "%"
    return X_test

def extract_data(kind):
    image_num = globals()["data_points_" + kind]
    if DEBUG:
        image_num = debug_num

    feature_matrix = []

    file_name = kind + "_test.spickle" # change test to descriptive name
    out_file = os.getcwd() + "/" + res_folder + file_name

    if os.path.isfile(out_file):
        print "\'" + file_name + "\' found, loading data..."
        for elm in sPickle.s_load(open(out_file)):
            feature_matrix.append(elm)
        print "done loading " + kind + " data"
    else:
        print "No file \'" + file_name + "\' found, starting to read data..."

        # parallel reading data
        feature_matrix = range(1, image_num + 1)
        p = Pool(5)
        p.map(globals()["process_img_" + str(kind)], feature_matrix)

        # write data to file
        out_file = open(out_file, 'w')
        sPickle.s_dump(feature_matrix, out_file)

    return feature_matrix

def read_targets():
    targets = []
    with open("targets.csv", 'r') as file:
        targets = file.read().split()
    targets = map(int, targets)

    if DEBUG:
        return targets[:debug_num]

    return targets

def generate_submission(Y_test, Name):
    filename = os.getcwd() + "/Submissions/submission_" + Name + ".csv"
    if os.path.isfile(filename):
        generate_submission(Y_test, Name + "1") # TODO change name to avoid colisions more elegant
        return
    with open(filename, "w") as file:
                file.write("Id,Prediction\n")
                for i in range(len(Y_test)):
                        file.write(str(i+1) + "," + str(int(Y_test[i])) + "\n")
                file.close()

def main():
    # First extract feature matrix from train set and load targets
    X_train = extract_data("train")
    Y_train = read_targets()

    # TODO Train models

    # Extract feature matrix from test set
    X_test = extract_data("test")

    # Make predictions for the test set and write it to a file
    Y_test = [] # TODO predict
    generate_submission(Y_test, "test")
if __name__ == "__main__":
    main()
