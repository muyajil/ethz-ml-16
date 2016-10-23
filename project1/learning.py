import sPickle
import sys

from numpy import genfromtxt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

test_number = 50
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
	#targets = genfromtxt('targets.csv', delimiter='\n')
	targets = []
	with open("targets.csv", 'r') as file:
		targets = file.read().split()
	targets = map(int, targets)
	return targets[:test_number]

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
print target

# learning part
if methodeid == 1: # LASSO
	print "Chosen Methode: LASSO\nStarting by reading in data..."

	#read in data
	clean_train = read_train()
	print "load clean train done"
	model = Lasso(alpha=0.9)
	print "started training"
	model.fit(clean_train, target)
	#scores = cross_val_score(model, clean_train, target, cv=5)
	#print "Cross validation scores"
	#print scores
	print "done training"
	del clean_train
	print "reading test data"
	clean_test = read_test()
	print "finished reading test data"
	print "making predictions"
	predictions = model.predict(clean_test)
	with open("prediction_lasso.csv", "w") as file:
		file.write("Id,Prediction\n")
		for i in range(len(predictions)):
			file.write(str(i) + "," + str(int(predictions[i])) + "\n")
		file.close()


elif methodeid == 2: # RIDGE
	print "TODO"
else:
	print "invalid methode id, do nothing and exit"
