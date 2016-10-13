import sPickle
import sys

from numpy import genfromtxt

from sklearn.linear_model import Lasso

def read_train(matrix):
	for elm in sPickle.s_load(open("spickle_train_data_clean.pickle")):
		matrix.append(elm)

def read_targets(targets):
	targets = genfromtxt('targets.csv', delimiter='\n')

#decide for learning algorithm
methodeid = 0
methodes = {
	"lasso": 1,
	"ridge":2
}
while True:
	sys.stdout.write("Available learning methodes: \n" + str(methodes.keys()))
	choice = raw_input().lower()
	if(choice in methodes):
		methodeid = methodes[choice]
		break
	else:
		sys.stdout.write("Not a valid choice, pls entwer one of the following options:\n" + str(methodes))

target = []
read_targets(target)
print target

# learning part
if methodeid == 1: # LASSO
	print "Chosen Methode: LASSO\nStarting by reading in data..."

	#read in data
	clean_train = []
	read_train(clean_train)
	print "load clean train done"

elif methodeid == 2: # RIDGE
	print "TODO"
else:
	print "invalid methode id, do nothing and exit"