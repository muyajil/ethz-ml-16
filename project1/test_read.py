import sPickle

matrix = []

for elm in sPickle.s_load(open('spickle_train_data_clean.pickle')):
    matrix.append(elm)

input = input("Just to keep matrix in mem...")
