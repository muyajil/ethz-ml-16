import cPickle
import sPickle

matrix_train = []
faulty_line_train = []

matrix_test = []
faulty_line_test = []

faulty_line_train_num = 14
faulty_line_test_num = 19

# Train data
for elm in sPickle.s_load(open("spickle_train_data.pickle")):
	matrix_train.append(elm)

for elm in sPickle.s_load(open("spickle_train_data_faulty.pickle")):
	faulty_line_train.append(elm)

#exchange faulty lines
matrix_train[faulty_line_train_num] = faulty_line_train

sPickle.s_dump(matrix_train, open("spickle_train_data_clean.pickle", "w"))

del matrix_train

# Test data
for elm in sPickle.s_load(open("spickle_test_data.pickle")):
	matrix_test.append(elm)

for elm in sPickle.s_load(open("spickle_test_data_faulty.pickle")):
	faulty_line_test.append(elm)

#exchange faulty lines
matrix_test[faulty_line_test_num] = faulty_line_test

sPickle.s_dump(matrix_train, open("spickle_test_data_clean.pickle", "w"))