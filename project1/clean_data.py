import cPickle

matrix_train = []
faulty_line_train = []

matrix_test = []
faulty_line_test = []

faulty_line_train_num = 14
faulty_line_test_num = None

with open(r"pickle_train_data.pickle", "rb") as pickle_in:
	matrix_train = cPickle.load(pickle_in)
with open(r"pickle_train_data_faulty.pickle", "rb") as pickle_in:
    faulty_line_train = cPickle.load(pickle_in)

#exchange faulty lines
matrix_train[faulty_line_train_num] = faulty_line_train

with open(r"pickle_train_data_clean.pickle", "wb") as pout_file:
	cPickle.dump(matrix_train, pout_file)

del matrix_train


with open(r"pickle_test_data.pickle", "rb") as pickle_in:
	matrix_test = cPickle.load(pickle_in)
with open(r"pickle_test_data_faulty.pickle", "rb") as pickle_in:
    faulty_line_test = cPickle.load(pickle_in)

#exchange faulty lines
matrix_test[faulty_line_test_num] = faulty_line_test

with open(r"pickle_test_data_clean.pickle", "wb") as pout_file:
	cPickle.dump(matrix_test, pout_file)

#with open(r"pickle_test_data_clean.pickle", "wb") as pout_file:
#	cPickle.dump(matrix_test, pout_file)
