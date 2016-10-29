import cPickle
import sPickle

faulty_line_train = []
faulty_line_test = []

faulty_line_train_num = 15 #index from actual picture (starts with 1)
faulty_line_test_num = 20

current_datapoint = 0
total_datapoints = 416

# Train data
for elm in sPickle.s_load(open("spickle_train_data_faulty.pickle")):
	faulty_line_train.append(elm)
print "done loading faulty line from train, now rewriting train data.."

index = 0
out_file = open("spickle_train_data_clean.pickle", "w")
for elm in sPickle.s_load(open("spickle_train_data.pickle")):
	index += 1
	if(index == faulty_line_train_num):
		#exchange faulty lines
		sPickle.s_dump_elt(faulty_line_train, out_file)
	else:
		sPickle.s_dump_elt(elm, out_file)
	current_datapoint += 1
	print "%.2f" % ((current_datapoint/float(total_datapoints)) * 100) + "%% done"

del faulty_line_train
out_file.close()

# Test data
for elm in sPickle.s_load(open("spickle_test_data_faulty.pickle")):
	faulty_line_test.append(elm)
print "done loading faulty line from test, now reqriting train data..."

index = 0
out_file = open("spickle_test_data_clean.pickle", "w")
for elm in sPickle.s_load(open("spickle_test_data.pickle")):
	index += 1
	if(index == faulty_line_test_num):
		#exchange faulty lines
		sPickle.s_dump_elt(faulty_line_test, out_file)
	else:
		sPickle.s_dump_elt(elm, out_file)
	current_datapoint += 1
	print "%.2f" % ((current_datapoint/float(total_datapoints)) * 100) + "%% done"

del faulty_line_test
out_file.close()

print "done with everything, all good."