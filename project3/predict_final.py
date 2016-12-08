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
SUBMISSION_VERSION = False # True for final submission -> single prediction file and overrites old!
computational_cores = 7 # number of workers to process paralizable workload

# Debug Flags
DEBUG = False
debug_num = 10

# Feature selection
cube_number = 7 # 3D cubes are cut into cube_number**3 smaller cubes before further processing
histogram_bins = 50 # number of bins to aggregate histogram
histogram_range = (1, 4001) # range from minimal to maximal significant data value

# constants
data_points_train = 278 # number of datapoints in train set
data_points_test = 138 # number of datapoints in test set
feature_vectors_folder = "Cache" # folder name for intermediate results
submission_folder = "Submissions" # folder name for submission files

# constants for cutting unsignificant boundary off of cube
x_start = 20
x_end = 156
y_start = 20
y_end = 188
z_start = 20
z_end = 156

# Output file names
PREPROCESSING_NAME = "cube-histo-" + str(cube_number) + "-" + str(histogram_bins)
SUBMISSION_NAME = "not_defined"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
    return X_train

def process_img_test(index):
    X_test = process_img("test", index)
    return X_test

def extract_data(kind):
    # calculate or load feature matrix for the test or train set
    if DEBUG:
        file_name = "debug_" + kind + "_" + PREPROCESSING_NAME + ".spickle"
        image_num = debug_num
    else:
        file_name = kind + "_" + PREPROCESSING_NAME + ".spickle"
        image_num = globals()["data_points_" + kind]

    feature_matrix = []
    out_file = os.getcwd() + "/" + feature_vectors_folder + "/" + file_name

    # only calculate in debug mode or if not allready done earlier
    if os.path.isfile(out_file) and not DEBUG:
        print bcolors.OKBLUE + "\'" + file_name + "\' found, loading data..." + bcolors.ENDC
        for elm in sPickle.s_load(open(out_file)):
            feature_matrix.append(elm)
    else:
        print bcolors.WARNING + "No file \'" + file_name + "\' found, starting to read data..." + bcolors.ENDC

        # parallel reading data
        pic_num = range(1, image_num + 1)
        pool = multiprocessing.Pool(computational_cores)
        feature_matrix = pool.map(globals()["process_img_" + str(kind)], pic_num)
        pool.close()
        pool.join()

        # write data to file
        sPickle.s_dump(feature_matrix, open(out_file, 'w'))

    print bcolors.OKGREEN + "Loading " + kind + " data finished." + bcolors.ENDC
    return feature_matrix

def read_targets():
    # returns the list of targets, each target is a list with 3 entries, eg., [[0,1,1],[1,0,1],...]
    # targets represent [1,1,1] for female, yung, healthy, [0,0,0] for male, old, sick
    targets = []
    with open("data/targets.csv", 'r') as file:
        targets = [map(int, x.split(',')) for x in file.read().split()]

    if DEBUG:
        return targets[:debug_num]
    return targets

def generate_submission(Y_test, Name, info=""):
    # Y_test should hold "ID,Sample,Label,Predicted" in one line for every datapoint
    #
    boolean = {0: "False", 1: "True"}
    if SUBMISSION_VERSION:
        filename = "final_sub.csv"
    else:
        filename = os.getcwd() + "/" + submission_folder + "/" + str(int(time())) + "_" + PREPROCESSING_NAME + "_" + Name + "_" + info + ".csv"

    if os.path.isfile(filename) and not SUBMISSION_VERSION: # only overrite a previous final_submission, not a normal one
        generate_submission(Y_test, Name + "1", info) # TODO change name to avoid colisions more elegant
        return
    with open(filename, "w") as file:
        file.write("ID,Sample,Label,Predicted\n")
        for i in range(len(Y_test)):
            file.write(str(3*i) + "," + str(i) + ",gender," + boolean[round(Y_test[i][0])] + "\n") #TODO check wheter correct index and correct logic
            file.write(str(3*i + 1) + "," + str(i) + ",age," + boolean[round(Y_test[i][1])] + "\n") #TODO check wheter correct index and correct logic
            file.write(str(3*i + 2) + "," + str(i) + ",health," + boolean[round(Y_test[i][2])] + "\n") #TODO check wheter correct index and correct logic
        file.close()
    print bcolors.OKBLUE + "Wrote submission file '" + filename[len(os.getcwd()) + 1:] + "'." + bcolors.ENDC

def generate_name(params_list, score_list):
    # expects a list of the used parameters and scores in the order [gender, age, health]
    # TODO: change to work with other formats/ more classifiers
    temp = zip(params_list[0].keys(), params_list[0].values(), params_list[1].values(), params_list[2].values())
    par = [str(k) + "=" + str(v0) + "_" + str(v1) + "_" + str(v2) for k,v0,v1,v2 in temp]
    return par

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
    print bcolors.UNDERLINE + bcolors.OKBLUE + 'Best Score of Grid Search: ' + str(grid_search.best_score_) + bcolors.ENDC
    print bcolors.UNDERLINE + bcolors.OKBLUE + 'Best Params of Grid Search: ' + str(grid_search.best_params_) + bcolors.ENDC
    return (grid_search.best_estimator_, grid_search.best_params_)

def svcPOLYGridSearch(X, y):
    global SUBMISSION_NAME
    SUBMISSION_NAME = "SVC_POLY"
    param_grid = [{'degree': np.linspace(1,5,5),'C': np.logspace(-3.20,10), 'gamma': np.logspace(-5,3,20), 'kernel': ['poly']}]
    grid_search = skgs.GridSearchCV(sksvm.SVC(probability=True), param_grid, cv=5, verbose=5)
    grid_search.fit(X,y)
    print bcolors.UNDERLINE + bcolors.OKBLUE + 'Best Score of Grid Search: ' + str(grid_search.best_score_) + bcolors.ENDC
    print bcolors.UNDERLINE + bcolors.OKBLUE + 'Best Params of Grid Search: ' + str(grid_search.best_params_) + bcolors.ENDC
    return (grid_search.best_estimator_, grid_search.best_params_)

def svcRBFGridsearch(X, y, param_grid, prob=True, cl_weight='balanced'):
    global SUBMISSION_NAME
    SUBMISSION_NAME = "SVC_RBF"
    grid_search = skgs.GridSearchCV(sksvm.SVC(probability=prob, class_weight=cl_weight), param_grid, cv=5, verbose=5)
    grid_search.fit(X,y)
    print bcolors.UNDERLINE + bcolors.OKBLUE + 'Best Score of Grid Search: ' + str(grid_search.best_score_) + bcolors.ENDC
    print bcolors.UNDERLINE + bcolors.OKBLUE + 'Best Params of Grid Search: ' + str(grid_search.best_params_) + bcolors.ENDC
    return (grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_)

def partial_svc(a):
    return svcRBFGridsearch(a[0], a[1])

def print_done():
    print bcolors.OKGREEN + bcolors.BOLD + "\n\nDone. Have a good night." + bcolors.ENDC
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
    global SUBMISSION_NAME
    # make sure folders exist so that output can be written
    if not SUBMISSION_VERSION:
        make_folder(submission_folder)
    make_folder(feature_vectors_folder)

    # First extract feature matrix from train set and load targets
    print bcolors.HEADER + "Reading traing data.." + bcolors.ENDC
    X_train = extract_data("train")
    Y_train = read_targets()

    Y_gender = [y[0] for y in Y_train]
    Y_age = [y[1] for y in Y_train]
    Y_sick = [y[2] for y in Y_train]

    # Train models
    print bcolors.HEADER + "Starting to train..." + bcolors.ENDC
    if SUBMISSION_VERSION: # exact parameters for final submission
        # TODO: old version, need to train 3 classifications!
        estimator = sksvm.SVC(probability=True, class_weight='balanced', gamma=0.0000000001, C=100, kernel='rbf')
        estimator.fit(X_train, Y_train)
        info = ""
        SUBMISSION_NAME = "finale_submission"
    else:
        param_grid = [{'C': np.logspace(0,10,10), 'kernel': ['rbf'], 'gamma': np.logspace(-10,-6,10)}]
        estimator_gender, params_gender, score_gender = svcRBFGridsearch(X_train, Y_gender, param_grid)
        param_grid = [{'C': np.logspace(0,10,10), 'kernel': ['rbf'], 'gamma': np.logspace(-10,-6,10)}]
        estimator_age, params_age, score_age = svcRBFGridsearch(X_train, Y_age, param_grid)
        param_grid = [{'C': np.logspace(0,10,10), 'kernel': ['rbf'], 'gamma': np.logspace(-10,-6,10)}]
        estimator_sick, params_sick, score_sick = svcRBFGridsearch(X_train, Y_sick, param_grid)

        info = generate_name([params_gender, params_age, params_sick], [score_gender, score_age, score_sick])

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
    print bcolors.HEADER + "Reading test data.." + bcolors.ENDC
    X_test = extract_data("test")

    # Make predictions for the test set and write it to a file
    print bcolors.HEADER + "Making predictions.." + bcolors.ENDC
    Y_test_gender = estimator_gender.predict_proba(X_test)
    Y_test_age = estimator_age.predict_proba(X_test)
    Y_test_sick = estimator_sick.predict_proba(X_test)
    Y_test_gender = np.array(Y_test_gender)[:, 1]
    Y_test_age = np.array(Y_test_age)[:, 1]
    Y_test_sick = np.array(Y_test_sick)[:, 1]
    Y_test = zip(Y_test_gender, Y_test_gender, Y_test_sick)

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

    generate_submission(Y_test, SUBMISSION_NAME, info)

    print_done()

if __name__ == "__main__":
    main()
