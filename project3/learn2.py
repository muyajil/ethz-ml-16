import os
import sys
import numpy as np
import tensorflow as tf
import nibabel as nib
import random
from time import time
#import sPickle # -> https://github.com/pgbovine/streaming-pickle

from multiprocessing import Pool
import threading

cube_factor = 1
batch_size = cube_factor**3

# constants
data_points_total = 278
data_points_test = 138
data_points_validate = batch_size
data_points_train = data_points_total - data_points_validate
computational_cores = 5

filter1_width = 16
filter1_height = 16
filter1_depth = 16
conv1_out = 4

filter2_width = 8
filter2_height = 8
filter2_depth = 8
conv2_out = 8

mri_depth = 176
mri_height = 208
mri_width = 176



ffn_1 = 8

def cubify(examples, cube_factor):
    (num_examples, max_x, max_y, max_z, content) = np.shape(examples)
    #print(np.shape(examples))

    x_inter = max_x//cube_factor
    y_inter = max_y//cube_factor
    z_inter = max_z//cube_factor

    cubes = np.empty((num_examples*cube_factor**3, x_inter, y_inter, z_inter, content))

    idx = 0

    for example in examples:
        idx = 0
        for x in range(cube_factor):
            for y in range(cube_factor):
                for z in range(cube_factor):
                    cube = example[x*x_inter:(x+1)*x_inter, y*y_inter:(y+1)*y_inter, z*z_inter:(z+1)*z_inter]
                    cubes[idx] = cube
                    idx+=1
    print("cubify done")
    return cubes

def multiply_targets(targets, cube_factor):
    #print(np.shape(targets))
    (num_examples, dim) = np.shape(targets)
    y_len = len(targets)*cube_factor**3
    y = np.empty((y_len, 3))
    for i in range(num_examples):
        for j in range(cube_factor**3):
            y[i*j] = targets[i]
    return y

def best_prediction(predictions):
    best = [-1.0,-1.0,-1.0]
    for prediction in predictions:
        best[0] = max(best[0], prediction[0])
        best[1] = max(best[1], prediction[1])
        best[2] = max(best[2], prediction[2])
    return best

def compute_predictions(predictions, cube_factor):
    num_examples = int(len(predictions)/cube_factor**3)

    pred_new = np.empty((num_examples, 3))

    for i in range(num_examples):
        preds = predictions[i*cube_factor**3:(i+1)*cube_factor**3]
        pred_new[i] = best_prediction(preds)
    return pred_new


def load_img(kind, index):
    img = nib.load("set_" + kind + "/" + kind + "_" + str(index) + ".nii")
    (height, width, depth, values) = img.shape
    data = img.get_data()
    X_3d = data[:, :, :, 0]
    print("done with img " + str(index))
    return X_3d

def load_img_train(index):
    return load_img("train", index)

def load_img_test(index):
    return load_img("test", index)

def load_X():
    p = Pool(computational_cores)
    X_train = p.map(load_img_train, range(1, data_points_total + 1))
    X_test = p.map(load_img_test, range(1, data_points_test + 1))
    return X_train, X_test

def load_y():
    with open("targets.csv", 'r') as file:
        lines = file.read().split()
        targets = np.empty((len(lines), 3))
        for i in range(len(lines)):
            (x, y, z) = lines[i].split(',')
            targets[i] = [int(x), int(y), int(z)]
    return targets

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def generate_submission(Y_test, Name="submission", info=""):
    # Y_test should hold "ID,Sample,Label,Predicted" in one line for every datapoint
    boolean = {0: "False", 1: "True"}
    filename = os.getcwd() + "/" + str(int(time())) + "_" + Name + "_" + info + ".csv"

    if os.path.isfile(filename): # only overrite a previous final_submission, not a normal one
        generate_submission(Y_test, Name + "1", info) # TODO change name to avoid colisions more elegant
        return
    with open(filename, "w") as file:
        file.write("ID,Sample,Label,Predicted\n")
        for i in range(len(Y_test)):
            #TODO check wheter correct index and correct logic
            file.write(str(3*i) + "," + str(i) + ",gender," + boolean[round(Y_test[i][0])] + "\n")
            file.write(str(3*i + 1) + "," + str(i) + ",age," + boolean[round(Y_test[i][1])] + "\n")
            file.write(str(3*i + 2) + "," + str(i) + ",health," + boolean[round(Y_test[i][2])] + "\n")
        file.close()
    print("Wrote submission file")

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_2x2x2(x):
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME')

def max_pool_4x4x4(x):
    return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1],
                          strides=[1, 4, 4, 4, 1], padding='SAME')

def get_next_batch(batch_size, X_train, y_train):
    idxs = np.random.choice(len(y_train)-1, batch_size, replace=False)
    X_batch = X_train[idxs]
    y_batch = y_train[idxs]
    #print(np.shape(X_train))
    #print(np.shape(y_train))

    return X_batch, y_batch

def main():

    # load data
    X_train, X_test = load_X()
    X_train = np.expand_dims(X_train, axis=4)
    X_test = np.expand_dims(X_test, axis=4)
    y_train = load_y()

    all_idxs = range(data_points_total)
    validate_idxs = np.random.choice(data_points_total, data_points_validate, replace=False)
    train_idxs = np.delete(all_idxs, validate_idxs)

    X_validate = X_train[validate_idxs]
    X_train = X_train[train_idxs]

    y_validate = y_train[validate_idxs]
    y_train = y_train[train_idxs]

    X_train = cubify(X_train, cube_factor)
    X_test = cubify(X_test, cube_factor)
    X_validate = cubify(X_validate, cube_factor)
    y_validate = multiply_targets(y_validate, cube_factor)
    y_train = multiply_targets(y_train, cube_factor)

    sess = tf.Session()
    with sess.as_default():
        x = tf.placeholder(tf.float32, shape=(batch_size, int(mri_depth/cube_factor), int(mri_height/cube_factor), int(mri_width/cube_factor), 1))
        y_ = tf.placeholder(tf.float32, shape=(batch_size, 3))

        # shape = [filter_depth, filter_height, filter_width, in_channels, out_channels]
        W_conv1 = weight_variable([
            filter1_depth,
            filter1_height,
            filter1_width,
            1,
            conv1_out
        ])

        b_conv1 = bias_variable([conv1_out])

        h_conv1 = tf.nn.relu(conv3d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_4x4x4(h_conv1)

        W_conv2 = weight_variable([
            filter2_depth,
            filter2_height,
            filter2_width,
            conv1_out,
            conv2_out
        ])

        b_conv2 = bias_variable([conv2_out])

        h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_4x4x4(h_conv2)

        # mri size = 176, 208, 176
        # 2 times 4x4x4 pooling leads to reduced size of 11, 13, 11
        convsize = int(((mri_depth/cube_factor) / 16) * ((mri_height/cube_factor) / 16) * ((mri_width/cube_factor) / 16) * conv2_out)
        W_fc1 = weight_variable([convsize, ffn_1])
        b_fc1 = bias_variable([ffn_1])

        h_pool2_flat = tf.reshape(h_pool2, [-1, convsize])

        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([ffn_1, 3])
        b_fc2 = bias_variable([3])

        y_pred = tf.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        #print(y_pred)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y_)

        cross_entropy = tf.reduce_mean(loss)

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_prediction = tf.equal(tf.round(y_pred), y_)

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(10000):
            batch_X, batch_y = get_next_batch(batch_size, X_train, y_train)

            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch_X, y_: batch_y, keep_prob: 1.0})
                #validate_accuracy  = accuracy.eval(feed_dict={x:X_validate, y_:y_validate, keep_prob: 1.0})
                #print("step %d, training accuracy %g, validation_accuracy %g"%(i, train_accuracy, validate_accuracy))
                print("step %d, training accuracy %g" % (i, train_accuracy))
            train_step.run(feed_dict={x: batch_X, y_: batch_y, keep_prob: 0.5})

        predictions = []

        for i in range(data_points_test):
            test_set = X_test[i*batch_size:(i+1)*batch_size]
            #test_set = [X_test[i]]
            set_pred = y_pred.eval(feed_dict={x:test_set, keep_prob:1.0})
            predictions.extend(set_pred)

        final_predictions = compute_predictions(predictions, cube_factor)

        generate_submission(final_predictions, Name="conv_net_test_1")

if __name__ == "__main__":
    main()
