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

def read_train():
    print "Loading training data from file..."

    matrix = []
    i = 0
    for elm in sPickle.s_load(open("spickle_train_avg_data_clean.pickle")):
        matrix.append(elm)
        i += 1
        if i == NUM_DATAPOINTS:
            break

    print "Finished loading training data"
    return matrix

def read_test():
    print "Loading test data from file..."

    matrix = []
    i = 0
    for elm in sPickle.s_load(open("spickle_test_avg_data_clean.pickle")):
        matrix.append(elm)
        i += 1
        if i == NUM_DATAPOINTS:
            break

    print "Finished loading test data"
    return matrix

def read_targets():
    targets = []
    with open("../targets.csv", 'r') as file:
        targets = file.read().split()
    targets = map(int, targets)

    if NUM_DATAPOINTS > 0:
        return targets[:NUM_DATAPOINTS]

    return targets

def generate_submission(Y_test):
    filename = "../submission_" + MODEL_NAME + ".csv"
    with open(filename, "w") as file:
                file.write("Id,Prediction\n")
                for i in range(len(Y_test)):
                        file.write(str(i+1) + "," + str(int(Y_test[i])) + "\n")
                file.close()

def train_and_predict(model):

    Y_train = read_targets()
    
    X_train = read_train()

    print "Start training..."
    
    model.fit(X_train, Y_train)
    
    if GRID_SEARCH:
        print 'Best score of Grid Search: ' + str(model.best_score_)
        print 'Best params of Grid Search: ' + str(model.best_params_)
    
    print "Finished training"
    del X_train

    X_test = read_test()

    print "Making predictions"

    Y_test = model.predict(X_test)
    
    generate_submission(Y_test)

# BEGIN MODELS

def do_lasso():
    print "Chosen Method: LASSO"
    MODEL_NAME = "LASSO"

    if GRID_SEARCH:
        param_grid = [{'alpha':np.linspace(100, 150, 10)}]
        model = grid_search.GridSearchCV(Lasso(max_iter=20000), param_grid, cv=5, verbose=10)
    else:
        model = Lasso(max_iter=20000, alpha=129)

    train_and_predict(model)

def do_svm_rbf():
    print "Chosen Method: SVM with RBF kernel"
    MODEL_NAME = "SVM_RBF"

    if GRID_SEARCH:
        param_grid = [{'C':[1.0, 10.0],  'epsilon':[0.01, 0.1, 1], 'kernel': ['rbf']}]
        model = grid_search.GridSearchCV(SVR(), param_grid, cv=5)
    else:
        model = SVR()

    train_and_predict(model)

def do_svm_poly():
    print "Chosen Method: SVM with Poly kernel"
    MODEL_NAME = "SVM_POLY"

    if GRID_SEARCH:
        param_grid = [{'C':[1.0, 10.0, 0.1], 'kernel': ['poly'], 'degree':[1,2,3,4,5,6]}]
        model = grid_search.GridSearchCV(SVR(), param_grid, cv=5)
    else:
        model = SVR()

    train_and_predict(model)

def do_ridge_rbf():
    print "Chosen Method: RIDGE with RBF kernel"
    MODEL_NAME = "RIDGE_RBF"
    if GRID_SEARCH:
        param_grid = [{'alpha':[1.0, 10.0],  'gamma':[0.01, 0.1, 1], 'kernel': ['rbf']}]
        model = grid_search.GridSearchCV(KernelRidge(), param_grid, cv=5)
    else:
        model = KernelRidge(alpha=10, kernel='rbf', gamma='0.1')

    train_and_predict(model)

def do_ridge_poly():
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
    print "Usage: learn.py {" + ','.join(models) + "} {grid_search, no_grid_search} [num_datapoints]"
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
        if len(sys.argv) > 2:
            GRID_SEARCH = sys.argv[2] == 'grid_search'

        if len(sys.argv) > 3:
            NUM_DATAPOINTS = int(sys.argv[3])

        if sys.argv[1] == models[0]:
            MODEL_NAME = models[0]
            do_lasso()
            exit()

        elif sys.argv[1] == models[1]:
            MODEL_NAME = models[1]
            do_svm_rbf()
            exit()

        elif sys.argv[1] == models[2]:
            MODEL_NAME = models[2]
            do_svm_poly()
            exit()

        elif sys.argv[1] == models[3]:
            MODEL_NAME = models[3]
            do_ridge_rbf()
            exit()

        elif sys.argv[1] == models[4]:
            MODEL_NAME = models[4]
            do_ridge_poly()
            exit()

        else:
            print "Unsupported model type"
            print_usage()
            exit()
