import sPickle
import sys

from sklearn.linear_model import Lasso
from sklearn import svm
from sklearn import grid_search
import numpy as np

def read_train():
	matrix = []
	i = 0
	for elm in sPickle.s_load(open("spickle_train_data_clean.pickle")):
		matrix.append(elm)
		i+=1
		if(i == test_number):
			break
	return matrix

def read_test():
	matrix = []
	i = 0
	for elm in sPickle.s_load(open("spickle_test_data_clean.pickle")):
		matrix.append(elm)
		i+=1
		if(i == test_number):
			break
	return matrix

def read_targets():
	targets = []
	with open("targets.csv", 'r') as file:
		targets = file.read().split()
	targets = map(int, targets)
	return targets

def generate_submission(predictions):
	with open("prediction_lasso.csv", "w") as file:
                file.write("Id,Prediction\n")
                for i in range(len(predictions)):
                        file.write(str(i+1) + "," + str(int(predictions[i])) + "\n")
                file.close()

def train_and_predict(model, grid_search):
	print "Loading training data from file..."

	clean_train = read_train()
	
	print "Finished loading training data"

	print "Start training..."
	
	model.fit(clean_train, target)
	
	if(grid_search):
		print 'Best score of Grid Search: ' + str(model.best_score_)
		print 'Best params of Grid Search: ' + str(model.best_params_)
	
	print "Finished training"
	del clean_train

	print "Loading test data from file..."

	clean_test = read_test()

	print "Finished loading test data"
	print "Making predictions"

	predictions = model.predict(clean_test)
	
	generate_submission(predictions)

def do_lasso(grid_search):
	print "Chosen Method: LASSO"
	
	if grid_search:
		param_grid = [{'alpha':np.logspace(-3, 20, 10)}]
		model = grid_search.GridSearchCV(Lasso(max_iter=2000), param_grid, cv=5)
	else:
		model = Lasso(max_iter=2000)

	train_and_predict(model, grid_search)

def do_svm(grid_search):
	print "Chosen Method: SVM"

	if grid_search:
		param_grid = [{'C':np.logspace(-3, 20, 10), 'epsilon':np.logspace(-5,3,20), 'kernel': ['rbf']}]
        model = grid_search.GridSearchCV(svm.SVR(), param_grid, cv=5)
    else:
    	model = svm.SVR()

    train_and_predict(model, grid_search)

def do_ridge(grid_search):
	print "Chosen Method: SVM"
	print "TODO"
	#if grid_search:
	#	param_grid = 
    #    model = 
    #else:
    #	model = 

    #train_and_predict(model, grid_search)

if __name__ == "__main__":
	if len(sys.argv == 0):
		print "You need to provide a model type"
		print "Usage: learn.py \{lasso, svm, ridge\} [grid_search]"
		exit()
	else:
		if sys.argv[0] == 'lasso':
			do_lasso(len(sys.argv) > 1 and sys.argv[1] == 'grid_search')
			exit()
		elif sys.argv[0] == 'svm':
			do_svm(len(sys.argv) > 1 and sys.argv[1] == 'grid_search')
			exit()
		elif sys.argv[0] == 'ridge':
			do_ridge(len(sys.argv) > 1 and sys.argv[1] == 'grid_search')
			exit()
		else:
			print "Unsupported model type"
			exit()
