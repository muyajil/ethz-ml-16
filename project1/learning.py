import sPickle
import sys

from numpy import genfromtxt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn import grid_search
import numpy as np

test_number = -1
length = 1667264

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
		#if len(elm) != length:
		#	print "!!! \nTest missmatch: " + str(i) + "; " + str(len(elm))
	return matrix

def read_targets():
	#targets = genfromtxt('targets.csv', delimiter='\n')
	targets = []
	with open("targets.csv", 'r') as file:
		targets = file.read().split()
	targets = map(int, targets)
	return targets

#decide for learning algorithm
methodeid = 0
methodes = {
	"lasso": 1,
	"ridge": 2
}
while True:
	sys.stdout.write("Available learning methodes: \n" + str(methodes.keys()))
	print ""
	choice = raw_input().lower()
	if(choice in methodes):
		methodeid = methodes[choice]
		break
	else:
		sys.stdout.write("Not a valid choice, pls enter one of the following options:\n" + str(methodes))

target = read_targets()

# learning part
if methodeid == 1: # LASSO
	print "Chosen Methode: LASSO\nStarting by reading in data..."

	#read in data
	clean_train = read_train()
	print "load clean train done"
	param_grid = [{'alpha':np.logspace(-3, 20, 10)}]
	model = Lasso(max_iter=2000)
	print "started training"
	gs = grid_search.GridSearchCV(model, param_grid, cv=5)
	gs.fit(clean_train, target)
	print 'Best score of Grid Search: ' + str(gs.best_score_)
	print 'Best params of Grid Search: ' + str(gs.best_params_)
	#scores = cross_val_score(model, clean_train, target, cv=5)
	#print "Cross validation scores"
	#print scores
	print "done training"
	#del clean_train
	print "reading test data"
	clean_test = read_test()
	print "finished reading test data"
	print "making predictions"
	predictions = gs.predict(clean_test)
	with open("prediction_lasso.csv", "w") as file:
		file.write("Id,Prediction\n")
		for i in range(len(predictions)):
			file.write(str(i) + "," + str(int(predictions[i])) + "\n")
		file.close()


elif methodeid == 2: # RIDGE
	print "TODO"
	exit()
else:
	print "invalid methode id, do nothing and exit"
