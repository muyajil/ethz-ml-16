import cPickle
import sPickle

faulty_line_train = []
faulty_line_test = []

faulty_line_train_num = 15 #index from actual picture (starts with 1)
faulty_line_test_num = 20

# Train data
for elm in sPickle.s_load(open("spickle_train_data_faulty.pickle")):
	faulty_line_train.append(elm)

index = 0
out_file = open("spickle_train_data_clean.pickle", "w")
for elm in sPickle.s_load(open("spickle_train_data.pickle")):
	index += 1
	if(index == faulty_line_train_num):
		#exchange faulty lines
		sPickle.s_dump_elt(faulty_line_train, out_file)
	else:
		sPickle.s_dump_elt(elm, out_file)

del faulty_line_train
out_file.close()

# Test data
for elm in sPickle.s_load(open("spickle_test_data_faulty.pickle")):
	faulty_line_test.append(elm)

index = 0
out_file = open("spickle_test_data_clean.pickle", "w")
for elm in sPickle.s_load(open("spickle_test_data.pickle")):
	if(index == faulty_line_test_num):
		#exchange faulty lines
		sPickle.s_dump_elt(faulty_line_test, out_file)
	else:
		sPickle.s_dump_elt(elm, out_file)

del faulty_line_test
out_file.close()