import sPickle

raw_train = []
clean_train = []

for elm in sPickle.s_load(open("spickle_train_data.pickle")):
	raw_train.append(elm)
print "load train done"

for elm in sPickle.s_load(open("spickle_train_data_clean.pickle")):
	clean_train.append(elm)
print "load clean train done"

for i in range(16):
	print len(raw_train[i])
	if(raw_train[i] != clean_train[i]):
		print str(i) + " is corrupted"

print "all good! :)"