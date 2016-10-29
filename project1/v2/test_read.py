import sPickle

matrix = []
matrix_test = []

total_datapoints = 20
current_datapoint = 0

for elm in sPickle.s_load(open('spickle_test_data.pickle')):
	matrix.append(elm)
	current_datapoint += 1
	print "%.2f" % ((current_datapoint/float(total_datapoints)) * 100) + "%% done"

i = 0
for elm in sPickle.s_load(open("spickle_test_data_clean_v1.pickle")):
    #matrix_test.append(elm)
    current_datapoint += 1
    if matrix[i] != elm:
    	print "!!!"
    i += 1
    print "%.2f" % ((current_datapoint/float(total_datapoints)) * 100) + "%% done"
input = input("Just to keep matrix in mem...")
