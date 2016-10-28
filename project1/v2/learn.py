"""
To implement a new model:
1. create function do_modelname
2. add modelname to models
"""

import sPickle
import sys

from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn import grid_search
import numpy as np

NUM_DATAPOINTS = -1
MODEL_NAME = ""
GRID_SEARCH = False
FILE_NAME = ""
HISTOGRAM = False

MAX_VALUE = 4418

def generate_histogram(vector):
    global MAX_VALUE
    histogram = [0]*(MAX_VALUE+1)
    for feature in vector:
        histogram[int(feature)] +=1
    return histogram

def read_data(filename):
    global MAX_VALUE
    global HISTOGRAM
    print "Loading " + filename + "..."

    matrix = []
    i = 0

    for elm in sPickle.s_load(open("data/" + filename)):
        max_elem = max(elm)
        if(max_elem > MAX_VALUE && MAX_VALUE == 0):
            MAX_VALUE = max_elem
        
        if HISTOGRAM:
            matrix.append(generate_histogram(elm))
        else:
            matrix.append(elm)
        
        i += 1
        print "Finished file " + str(i)
        
        if i == NUM_DATAPOINTS:
            break

    if HISTOGRAM:
        with open("data/" + filename + "_histo.csv", 'w') as file:
            for elm in matrix:
                file.write(",".join([str(x) for x in elm]))
            file.close()

    print "Finished loading " + filename

    return matrix

def read_targets():
    targets = []
    with open("data/targets.csv", 'r') as file:
        targets = file.read().split()
    targets = map(int, targets)

    if NUM_DATAPOINTS > 0:
        return targets[:NUM_DATAPOINTS]

    return targets

def generate_submission(Y_test):
    filename = "submission_" + MODEL_NAME + ".csv"
    with open(filename, "w") as file:
                file.write("Id,Prediction\n")
                for i in range(len(Y_test)):
                        file.write(str(i+1) + "," + str(int(Y_test[i])) + "\n")
                file.close()

def train_and_predict(model):
    global MAX_VALUE
    Y_train = read_targets()
    
    X_train = read_data(FILE_NAME + "_train.pickle")

    print "Start training..."
    
    model.fit(X_train, Y_train)
    
    if GRID_SEARCH:
        print 'Best score of Grid Search: ' + str(model.best_score_)
        print 'Best params of Grid Search: ' + str(model.best_params_)
    
    print "Finished training"
    del X_train
    
    X_test = read_data(FILE_NAME + "_test.pickle")

    print "Making predictions"

    Y_test = model.predict(X_test)
    
    generate_submission(Y_test)

# BEGIN MODELS

def do_lasso():
    global MODEL_NAME
    print "Chosen Method: LASSO"
    MODEL_NAME = "LASSO"

    if GRID_SEARCH:
        param_grid = [{'alpha':np.linspace(10, 1000, 100)}]
        model = grid_search.GridSearchCV(Lasso(max_iter=20000), param_grid, cv=5, verbose=5)
    else:
        model = Lasso(max_iter=20000, alpha=1)

    train_and_predict(model)

def do_svm_rbf():
    global MODEL_NAME
    print "Chosen Method: SVM with RBF kernel"
    MODEL_NAME = "SVM_RBF"

    if GRID_SEARCH:
        param_grid = [{'C':[1.0, 10.0],  'epsilon':[0.01, 0.1, 1], 'kernel': ['rbf']}]
        model = grid_search.GridSearchCV(SVR(), param_grid, cv=5)
    else:
        model = SVR()

    train_and_predict(model)

def do_svm_poly():
    global MODEL_NAME
    print "Chosen Method: SVM with Poly kernel"
    MODEL_NAME = "SVM_POLY"

    if GRID_SEARCH:
        param_grid = [{'C':[1.0, 10.0, 0.1], 'kernel': ['poly'], 'degree':[1,2,3,4,5,6]}]
        model = grid_search.GridSearchCV(SVR(), param_grid, cv=5)
    else:
        model = SVR()

    train_and_predict(model)

def do_ridge_rbf():
    global MODEL_NAME
    print "Chosen Method: RIDGE with RBF kernel"
    MODEL_NAME = "RIDGE_RBF"
    if GRID_SEARCH:
        param_grid = [{'alpha':[1.0, 10.0],  'gamma':[0.01, 0.1, 1], 'kernel': ['rbf']}]
        model = grid_search.GridSearchCV(KernelRidge(), param_grid, cv=5)
    else:
        model = KernelRidge(alpha=10, kernel='rbf', gamma='0.1')

    train_and_predict(model)

def do_ridge_poly():
    global MODEL_NAME
    print "Chosen Method: RIDGE with Poly kernel"
    MODEL_NAME = "RIDGE_POLY"
    if GRID_SEARCH:
        param_grid = [{'alpha':[1.0, 10.0, 0.1], 'kernel': ['poly'], 'degree':[1,2,3,4,5,6]}]
        model = grid_search.GridSearchCV(KernelRidge(), param_grid, cv=5)
    else:
        model = KernelRidge(alpha=10, kernel='poly', degree='5')

    train_and_predict(model)

# END MODELS

def print_usage():
    print ""
    print "Usage: learn.py {" + ','.join(models) + "} {grid_search, no_grid_search} input_file_name {histogram, no_histogram} [num_datapoints]"
    print "{...}: Choose one of the possibilities"
    print "[...]: Either use this param or not"
    print ""

if __name__ == "__main__":
    models = ['lasso', 'svm_rbf', 'svm_poly', 'ridge_rbf', 'ridge_poly']
    if len(sys.argv) <= 1:
        print "You need to provide a model type"
        print_usage()
        exit()
    else:
        print "Your chosen parameters:"
        if len(sys.argv) > 2:
            GRID_SEARCH = sys.argv[2] == 'grid_search'
            print "Grid Search\t" + str(GRID_SEARCH)

        if len(sys.argv) > 3:
            FILE_NAME = sys.argv[3]

        if len(sys.argv) > 4:
            HISTOGRAM = sys.argv[4] == 'histogram'
            print "Histogram\t" + str(HISTOGRAM)

        if len(sys.argv) > 5:
            NUM_DATAPOINTS = int(sys.argv[5])
            print "Datapoints\t" + str(NUM_DATAPOINTS)

        if sys.argv[1] in models:
            globals()["do_"+sys.argv[1]]()
            exit()
        else:
            print "Unsupported model type"
            print_usage()
            exit()
            
