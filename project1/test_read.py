import sPickle

for elm in sPickle.s_load(open('spickle_train_data.pickle')):
    print len(elm)
    print elm[0]
