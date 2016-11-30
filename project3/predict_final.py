import os
import sys
import numpy as np
import nibabel as nib
import src.sPickle as sPickle # -> https://github.com/pgbovine/streaming-pickle
import sklearn.grid_search as skgs
import sklearn.svm as sksvm
import multiprocessing
import threading
from time import time

# Execution flags
SUBMISSION_VERSION = False # True for final submission -> no output or customizability
DEBUG = False
debug_num = 10

# params for aggregating
cube_number = 7
histogram_bins = 50
histogram_range = (1, 4001)

# constants
data_points_train = 278
data_points_test = 138
feature_vectors_folder = "Out"
submission_folder = "Submission"
computational_cores = 8

# constants for cutting cube into right shape
x_start = 20
x_end = 156
y_start = 20
y_end = 188
z_start = 20
z_end = 156

# Output file names
PREPROCESSING_NAME = "cube-histo-" + str(cube_number) + "-" + str(histogram_bins)
SUBMISSION_NAME = "not_defined"

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

    file_name = kind + "_" + PREPROCESSING_NAME + ".spickle"

    if DEBUG:
        file_name = "debug_" + kind + "_" + PREPROCESSING_NAME + ".spickle"
        image_num = debug_num
        computational_cores = min(debug_num, computational_cores)

    feature_matrix = []
    out_file = os.getcwd() + "/" + feature_vectors_folder + "/" + file_name

    if os.path.isfile(out_file) and not DEBUG:
        print "\'" + file_name + "\' found, loading data..."
        for elm in sPickle.s_load(open(out_file)):
            feature_matrix.append(elm)
        print "done loading " + kind + " data"
    else:
        print "No file \'" + file_name + "\' found, starting to read data..."

        # parallel reading data
        pic_num = range(1, image_num + 1)
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
    with open("data/targets.csv", 'r') as file:
        targets = map(int, file.read().split())

    if DEBUG:
        return targets[:debug_num]
    return targets

def generate_submission(Y_test, Name, info=""):
    # Y_test should hold "ID,Sample,Label,Predicted" in one line for every datapoint
    boolean = {0: "False", 1: "True"}
    if SUBMISSION_VERSION:
        filename = "final_sub.csv"
    else:
        filename = os.getcwd() + "/" + submission_folder + "/" + str(int(time())) + "_" + PREPROCESSING_NAME + "_" + Name + "_" + info + ".csv"

    if os.path.isfile(filename) and not SUBMISSION_VERSION:
        generate_submission(Y_test, Name + "1", info) # TODO change name to avoid colisions more elegant
        return
    with open(filename, "w") as file:
        file.write("ID,Sample,Label,Predicted\n")
        for i in range(len(Y_test)):
            file.write(str(3*i) + "," + str(i) + ",gender," + boolean[Y_test[i][1]] + "\n") #TODO check wheter correct index and correct logic
            file.write(str(3*i + 1) + "," + str(i) + ",age," + boolean[Y_test[i][2]] + "\n") #TODO check wheter correct index and correct logic
            file.write(str(3*i + 2) + "," + str(i) + ",health," + boolean[Y_test[i][3]] + "\n") #TODO check wheter correct index and correct logic
        file.close()
    print "Wrote submission file '" + filename + "'."

def generate_name():
    par = [str(k) + "=" + str(v) for k,v in zip(params.keys(), params.values())]

def make_folder(foldername):
    folder = os.getcwd() + "/" + foldername + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

def svcSIGMOIDGridSearch(X, y):
    global SUBMISSION_NAME
    SUBMISSION_NAME = "SVC_SIG"
    param_grid = [{'C': np.logspace(-3,20,2), 'gamma': np.logspace(-5,3,20), 'kernel': ['sigmoid']}]
    grid_search = skgs.GridSearchCV(sksvm.SVC(probability=True), param_grid, cv=5, verbose=5)
    grid_search.fit(X,y)
    print 'Best Score of Grid Search: ' + str(grid_search.best_score_)
    print 'Best Params of Grid Search: ' + str(grid_search.best_params_)
    return (grid_search.best_estimator_, grid_search.best_params_)

def svcPOLYGridSearch(X, y):
    global SUBMISSION_NAME
    SUBMISSION_NAME = "SVC_POLY"
    param_grid = [{'degree': np.linspace(1,5,5),'C': np.logspace(-3.20,10), 'gamma': np.logspace(-5,3,20), 'kernel': ['poly']}]
    grid_search = skgs.GridSearchCV(sksvm.SVC(probability=True), param_grid, cv=5, verbose=5)
    grid_search.fit(X,y)
    print 'Best Score of Grid Search: ' + str(grid_search.best_score_)
    print 'Best Params of Grid Search: ' + str(grid_search.best_params_)
    return (grid_search.best_estimator_, grid_search.best_params_)

def svcRBFGridsearch(X, y):
    global SUBMISSION_NAME
    SUBMISSION_NAME = "SVC_RBF"
    param_grid = [{'C': np.logspace(0,1,2), 'kernel': ['rbf'], 'gamma': np.logspace(-10,-8,2)}]
    grid_search = skgs.GridSearchCV(sksvm.SVC(probability=True, class_weight='balanced'), param_grid, cv=5, verbose=5)
    grid_search.fit(X,y)
    print 'Best Score of Grid Search: ' + str(grid_search.best_score_)
    print 'Best Params of Grid Search: ' + str(grid_search.best_params_)
    return (grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_)

def partial_svc(a):
    return svcRBFGridsearch(a[0], a[1])

def print_done():
    print "\n\nDone. Have a good night."
    print("""\
                                       ._ o o
                                       \_`-)|_
                                    ,""       \\
                                  ,"  ## |   o o.
                                ," ##   ,-\__    `.
                              ,"       /     `--._;)
                            ,"     ## /
                          ,"   ##    /
                    """)

def main():
    # make sure folders exist so that output can be written
    if not SUBMISSION_VERSION:
        make_folder(submission_folder)
    make_folder(feature_vectors_folder)

    # First extract feature matrix from train set and load targets
    X_train = extract_data("train")
    Y_train = read_targets()

    # Train models
    print "Starting to train..."
    if SUBMISSION_VERSION: # exact parameters for final submission
        estimator = sksvm.SVC(probability=True, class_weight='balanced', gamma=0.0000000001, C=100, kernel='rbf')
        estimator.fit(X_train, Y_train)
        params = {}
        score = 0
        SUBMISSION_NAME = "finale_submission"
    else:
        estimator, params, score = svcRBFGridsearch(X_train, Y_train)
        #estimator, params, score = svcPOLYGridSearch(X_train, Y_train)
        #estimator, params, score = svcSIGMOIDGridSearch(X_train, Y_train)

        # distributed learning
        '''
        X_train = np.array(X_train)
        # splits the n**3 cube vectors into n vectors (representing n**2 cubes)
        split_X_train = [X_train[:, cube_number*i:(cube_number**2)*histogram_bins*(i+1)] for i in range(cube_number)]

        pool = multiprocessing.Pool(computational_cores)
        # each svc receives a strip of the X matrix (feature x0-xj for every brain) and the full Y_train
        # return value is a list of tuples with estimator, params, score
        learner_stuff = pool.map(partial_svc, zip(split_X_train, [Y_train for i in range(cube_number)]))
        '''
    del X_train, Y_train

    # Extract feature matrix from test set
    X_test = extract_data("test")

    # Make predictions for the test set and write it to a file
    print "Making predictions."
    Y_test = estimator.predict_proba(X_test)

    '''
    # averages the weight vectors to one more reliable weight vector
    X_test = np.array(X_test)
    Y_test = np.zeros((data_points_test, 2))
    # again split the featur matrix into strips with a partial feature set for every picture
    split_X_test = [X_test[:, cube_number*i:(cube_number**2)*histogram_bins*(i+1)] for i in range(cube_number)]
    for i in range(cube_number):
        temp = learner_stuff[i][0].predict_proba(split_X_test[i])
        Y_test = np.add(Y_test, temp)
    Y_test = Y_test / cube_number

    params = {}
    score = 0
    SUBMISSION_NAME = "parallel-svc"
    '''

    info = "" #TODO print score and parameters info into info
    generate_submission(Y_test, SUBMISSION_NAME, info)

    print_done()

if __name__ == "__main__":
    main()
