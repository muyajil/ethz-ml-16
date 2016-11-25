import os
import sys
import numpy as np
import nibabel as nib
import sPickle # -> https://github.com/pgbovine/streaming-pickle
import sklearn.grid_search as skgs
import sklearn.svm as sksvm
import multiprocessing
from functools import partial
import threading
from time import time

# Execution flags
SUBMISSION_VERSION = False # True for final submission -> no output or customizability
DEBUG = True
debug_num = 10

# params for aggregating
cube_number = 5
MAX = 4000
histogram_bins = 50
histogram_range = (1, 4001)

# constants
data_points_train = 278
data_points_test = 138
res_folder = "Out/"
computational_cores = 5

# constants for cutting cube into right shape
x_start = 20
x_end = 156
y_start = 20
y_end = 188
z_start = 20
z_end = 156

# Output file names
SUBMISSION_NAME = "not_defined"
PREPROCESSING_NAME = "cube-histo-" + str(cube_number)

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
    X_3d = np.array(load_img(kind, index))

    # XXX process img and store in 'X'
    X_3d = X_3d[x_start:x_end, y_start:y_end, z_start:z_end]

    XX, YY, ZZ = X_3d.shape

    X_step = XX/cube_number
    Y_step = YY/cube_number
    Z_step = ZZ/cube_number

    X = []

    for x in range(cube_number):
        x_a = x*X_step
        x_b = min((x+1)*X_step - 1, x_end)
        for y in range(cube_number):
            y_a = y*Y_step
            y_b = min((y+1)*Y_step - 1, y_end)
            for z in range(cube_number):
                z_a = z*Z_step
                z_b = min((z+1)*Z_step - 1, z_end)
                cube = X_3d[x_a:x_b, y_a:y_b, z_a:z_b]
                #print str(x_a) + "  " + str(x_b) + "," + str(y_a) + "  " + str(y_b) + "," + str(z_a) + "  " + str(z_b)

                temp = cube.flatten()
                X.extend(np.histogram(temp, bins=histogram_bins, range=histogram_range)[0])
    return X # 1D feature vector

def process_img_train(index):
    X_train = process_img("train", index)
    #print "Finished reading file train_" + str(index) + "; " + "%.2f" % ((progress_tracker/float(data_points_train)) * 100) + "%"
    return X_train

def process_img_test(index):
    X_test = process_img("test", index)
    #print "Finished reading file test_" + str(index) + "; " + "%.2f" % ((progress_tracker/float(data_points_test)) * 100) + "%"
    return X_test

def extract_data(kind):
    global computational_cores
    image_num = globals()["data_points_" + kind]

    file_name = kind + "_" + PREPROCESSING_NAME + ".spickle" # change test to descriptive name

    if DEBUG:
        file_name = "debug_" + kind + "_" + PREPROCESSING_NAME + ".spickle"
        image_num = debug_num
        computational_cores = min(debug_num, computational_cores)

    feature_matrix = []
    out_file = os.getcwd() + "/" + res_folder + file_name

    if os.path.isfile(out_file) and not DEBUG:
        print "\'" + file_name + "\' found, loading data..."
        for elm in sPickle.s_load(open(out_file)):
            feature_matrix.append(elm)
        print "done loading " + kind + " data"
    else:
        print "No file \'" + file_name + "\' found, starting to read data..."

        # parallel reading data
        pic_num = range(1, image_num + 1)
        lock = multiprocessing.Lock()
        pool = multiprocessing.Pool(computational_cores)
        feature_matrix = pool.map(globals()["process_img_" + str(kind)], pic_num)
        pool.close()
        pool.join()

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

def generate_submission(Y_test, Name, param=""):
    filename = os.getcwd() + "/Submissions/" + str(int(time())) + "_" + Name + "_" + param + ".csv"
    if os.path.isfile(filename):
        generate_submission(Y_test, Name + "1") # TODO change name to avoid colisions more elegant
        return
    with open(filename, "w") as file:
                file.write("Id,Prediction\n")
                for i in range(len(Y_test)):
                    file.write(str(i+1) + "," + str(Y_test[i][1]) + "\n")
                file.close()

def svcSIGMOIDGridSearch(X, y):
    global SUBMISSION_NAME
    SUBMISSION_NAME = "SVC_SIG"
    param_grid = [{'C': np.logspace(-3,20,2), 'gamma': np.logspace(-5,3,20), 'kernel': ['sigmoid']}]
    grid_search = skgs.GridSearchCV(sksvm.SVC(probability=True), param_grid, cv=5, verbose=5)
    grid_search.fit(X,y)
    print 'Best Score of Grid Search: ' + str(grid_search.best_score_)
    print 'Best Params of Grid Search: ' + str(grid_search.best_params_)
    return (grid_search.best_estimator_, str(grid_search.best_params_))

def svcPOLYGridSearch(X, y):
    global SUBMISSION_NAME
    SUBMISSION_NAME = "SVC_POLY"
    param_grid = [{'degree': np.linspace(1,5,5),'C': np.logspace(-3.20,10), 'gamma': np.logspace(-5,3,20), 'kernel': ['poly']}]
    grid_search = skgs.GridSearchCV(sksvm.SVC(probability=True), param_grid, cv=5, verbose=5)
    grid_search.fit(X,y)
    print 'Best Score of Grid Search: ' + str(grid_search.best_score_)
    print 'Best Params of Grid Search: ' + str(grid_search.best_params_)
    return (grid_search.best_estimator_, str(grid_search.best_params_))

def svcRBFGridsearch(X, y):
    global SUBMISSION_NAME
    SUBMISSION_NAME = "SVC_RBF"
    param_grid = [{'C': np.logspace(1,4,10), 'kernel': ['rbf'], 'gamma': np.logspace(-10,-4,10)}]
    grid_search = skgs.GridSearchCV(sksvm.SVC(probability=True, class_weight='balanced'), param_grid, cv=5, verbose=5)
    grid_search.fit(X,y)
    print 'Best Score of Grid Search: ' + str(grid_search.best_score_)
    print 'Best Params of Grid Search: ' + str(grid_search.best_params_)
    return (grid_search.best_estimator_, str(grid_search.best_params_))

def main():
    # First extract feature matrix from train set and load targets
    if(DEBUG):
        pass

    X_train = extract_data("train")
    Y_train = read_targets()
    print sum(Y_train)

    # Train models
    estimator, params = svcRBFGridsearch(X_train, Y_train)
    #estimator = svcPOLYGridSearch(X_train, Y_train)
    #estimator = svcSIGMOIDGridSearch(X_train, Y_train)

    # Extract feature matrix from test set
    X_test = extract_data("test")

    # Make predictions for the test set and write it to a file
    Y_test = estimator.predict_proba(X_test)
    generate_submission(Y_test, SUBMISSION_NAME, params)

if __name__ == "__main__":
    main()
