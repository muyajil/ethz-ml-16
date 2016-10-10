import sPickle

for elm in sPickle.s_load(open('cont_spickle_train_data.pickle')):
    print len(elm)
    print elm[0]
