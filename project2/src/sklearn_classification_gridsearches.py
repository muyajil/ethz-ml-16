import numpy as np
import sklearn.svm as sksvm
import sklearn.grid_search as skgs
import sklearn.linear_model as sklm
import sklearn.neighbors as skn
from sklearn import datasets

# The following functions each implement a gird search over the hyper-parameter
# space of various supervised classification algorithms
# Use them to find the best model for the problem.
def radNearestNeighborsGridSeach(X,y):
    param_grid = [{'radius': np.linspace(0.1,2,20), 'weights': ['uniform', 'distance'], 'algorithm':['auto', 'kd_tree'], 'outlier_label':[-1]}]
    grid_search = skgs.GridSearchCV(skn.RadiusNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X,y)
    print 'Best Score of Grid Search: ' + str(grid_search.best_score_)
    print 'Best Params of Grid Search: ' + str(grid_search.best_params_)

def kNearestNeighborsGridSearch(X, y):
    param_grid = [{'n_neighbors': np.linspace(1,10,10), 'weights': ['uniform', 'distance'], 'algorithm': ['auto']}]
    grid_search = skgs.GridSearchCV(skn.KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X,y)
    print 'Best Score of Grid Search: ' + str(grid_search.best_score_)
    print 'Best Params of Grid Search: ' + str(grid_search.best_params_)

def sgdClassifierGridSearch(X, y):
    param_grid = [{'loss': ['hinge', 'modified_huber', 'squared_hinge', 'perceptron', 'log', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], 'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': ['optimal', '0.0001'], 'epsilon': np.logspace(-3,10,10)}]
    grid_search = skgs.GridSearchCV(sklm.SGDClassifier(), param_grid, cv=5)
    grid_search.fit(X,y)
    print 'Best Score of Grid Search: ' + str(grid_search.best_score_)
    print 'Best Params of Grid Search: ' + str(grid_search.best_params_)

def logisticRegressionGridSearch(X, y):
    param_grid = [{'penalty':['l1', 'l2'], 'dual': [False], 'C': np.logspace(-3.20,10), 'solver':['sag']}]
    grid_search = skgs.GridSearchCV(sklm.LogisticRegression(), param_grid, cv=5)
    grid_search.fit(X,y)
    print 'Best Score of Grid Search: ' + str(grid_search.best_score_)
    print 'Best Params of Grid Search: ' + str(grid_search.best_params_)

def svcSIGMOIDGridSearch(X, y, Test):
    param_grid = [{'C': np.logspace(-3,20,10), 'gamma': np.logspace(-5,3,20), 'kernel': ['sigmoid']}]
    grid_search = skgs.GridSearchCV(sksvm.SVC(), param_grid, cv=5)
    grid_search.fit(X,y)
    print 'Best Score of Grid Search: ' + str(grid_search.best_score_)
    print 'Best Params of Grid Search: ' + str(grid_search.best_params_)

def svcPOLYGridSearch(X, y):
    param_grid = [{'degree': np.linspace(1,5,5),'C': np.logspace(-3.20,10), 'gamma': np.logspace(-5,3,20), 'kernel': ['poly']}]
    grid_search = skgs.GridSearchCV(sksvm.SVC(), param_grid, cv=5)
    grid_search.fit(X,y)
    print 'Best Score of Grid Search: ' + str(grid_search.best_score_)
    print 'Best Params of Grid Search: ' + str(grid_search.best_params_)

def svcRBFGridsearch(X, y):
    param_grid = [{'C': np.logspace(-1,20,10), 'gamma': np.logspace(-5,3,20), 'probability':[True, False], 'kernel': ['rbf']}]
    grid_search = skgs.GridSearchCV(sksvm.SVC(), param_grid, cv=5)
    grid_search.fit(X,y)
    print 'Best Score of Grid Search: ' + str(grid_search.best_score_)
    print 'Best Params of Grid Search: ' + str(grid_search.best_params_)

if __name__ == '__main__':

    datas = datasets.load_iris()
    # Split data into training/testing sets
    data_X = datas.data
    data_X_train = data_X[:-20]
    data_X_test = data_X[-20:]
    data_y_train = datas.target[:-20]
    data_y_test = datas.target[-20:]

    #svcRBFGridsearch(data_X_train, data_y_train)
    svcPOLYGridSearch(data_X_train, data_y_train)
