import cPickle

with open(r"pickle_train_data.pickle", "rb") as pickle_in:
	e = cPickle.load(pickle_in)
	print e[0][0]
