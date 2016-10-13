import sPickle

matrix = []
matrix_test = []

total_datapoints = 416
current_datapoint = 0

for elm in sPickle.s_load(open('spickle_train_data_clean.pickle')):
    matrix.append(elm)
    current_datapoint += 1
	print "%.2f" % ((current_datapoint/float(total_datapoints)) * 100) + "%% done"

for elm in sPickle.s_load(open("spickle_test_data_clean.pickle")):
    matrix_test.append(elm)
    current_datapoint += 1
    print "%.2f" % ((current_datapoint/float(total_datapoints)) * 100) + "%% done"

input = input("Just to keep matrix in mem...")
