import numpy as np
import tensorflow as tf
import cv2
import pudb
import datetime
import time
import sys
import os
import fnmatch
import random

sys.path.append('../3_svm_model/')
sys.path.append('../2_compare_detectors/')

from train import filenames_for_participants
from divide import divide_in

hists = np.load("../3_svm_model/cached/train_cache_0.npy")
hists_test = np.load("../3_svm_model/cached/test_cache_0.npy")
labels = np.load("../3_svm_model/cached/train_cache_0_labels.npy")
labels_test = np.load("../3_svm_model/test_cache_0_labels.npy")

labels = labels[:, np.newaxis]
participants = []
for root, dirnames, filenames in os.walk("../../images/train_set"):
    for filename in fnmatch.filter(filenames, '*.png'):
        part = filename.split(" - ")[0]
        if part not in participants:
            participants.append(part)
participants.sort()

random.seed(0)
participants_sliced = divide_in(participants, 5)
folds = []
for p in participants_sliced:
    filenames_pos, filenames_neg = \
        filenames_for_participants(p,
            os.walk("../../images/train_set"), cream=True)
    filenames_pos_mining, filenames_neg_mining = \
        filenames_for_participants(p,
            os.walk("../../images/classified_mining"), cream=True)
    filenames_pos = filenames_pos + filenames_pos_mining
    filenames_neg = filenames_neg + filenames_neg_mining
    filenames_pos.sort()
    filenames_neg.sort()

    images_pos = []
    images_neg = []

    for i, filename in enumerate(filenames_pos):
        img = cv2.imread(filename)
        img = cv2.resize(img, (64, 64))
        images_pos.append(img)
    for i, filename in enumerate(filenames_neg):
        img = cv2.imread(filename)
        img = cv2.resize(img, (64, 64))
        images_neg.append(img)

    folds.append([images_pos, images_neg])

train_set_pos = []
train_set_neg = []

test_set_pos = []
test_set_neg = []
for j, f_ in enumerate(folds):
    if j == 0:
        test_set_pos = folds[j][0]
        test_set_neg = folds[j][1]
    else:
        train_set_pos += folds[j][0]
        train_set_neg += folds[j][1]

labels_train = np.zeros(len(train_set_pos)+len(train_set_neg))
labels_train[len(train_set_neg)-1:] = 1
hists_train = np.concatenate((train_set_neg, train_set_pos))/255.0-0.5

labels_test = np.zeros(len(test_set_pos)+len(test_set_neg))
labels_test[len(test_set_neg)-1:] = 1
hists_test = np.concatenate((test_set_neg, test_set_pos))/255.0-0.5

# one-hot the label array ([0, 1, 0] -> [[1, 0], [0, 1], [1, 0]]
labels_test = (labels_test[:,None] != np.arange(2)).astype(float)

# depth in convolutional layers
K = 12 # conv layer, depth = K, patchsize = 5, stride = 1
L = 16 # conv layer, depth = L, patchsize = 4, stride = 2 img=32
M = 24 # conv layer, depth = M, patchsize = 4, stride = 2 so img=16
N = 100 # fully connected layer 

W1 = tf.Variable(tf.truncated_normal([5, 5, 3, K], stddev=.1))
B1 = tf.Variable(tf.zeros([K]))
W2 = tf.Variable(tf.truncated_normal([4, 4, K, L], stddev=.1))
B2 = tf.Variable(tf.zeros([L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=.1))
B3 = tf.Variable(tf.zeros([M]))
W4 = tf.Variable(tf.truncated_normal([8*8*M, N], stddev=.1))
B4 = tf.Variable(tf.zeros([N]))
W5 = tf.Variable(tf.truncated_normal([N, 2], stddev=.1))
B5 = tf.Variable(tf.zeros([1]))

keep_prob = tf.placeholder(tf.float32)

# Input placeholder
X = tf.placeholder(tf.float32, [None, 64, 64, 3])

# dropout probability
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)
# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    # adding the iteration prevents from averaging across non-existing iterations
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

Y1 = tf.nn.relu(
    tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME") + B1)
poolY1 = tf.layers.max_pooling2d(Y1, 2, 2)
# dropY1 = tf.nn.dropout(poolY1, keep_prob)
Y2 = tf.nn.relu(
    tf.nn.conv2d(poolY1, W2, strides=[1, 1, 1, 1], padding="SAME") + B2)
poolY2 = tf.layers.max_pooling2d(Y2, 2, 2)
# dropY2 = tf.nn.dropout(poolY2, keep_prob, compatible_convolutional_noise_shape(poolY1))
Y3 = tf.nn.relu(
    tf.nn.conv2d(poolY2, W3, strides=[1, 1, 1, 1], padding="SAME") + B3)
poolY3 = tf.layers.max_pooling2d(Y3, 2, 2)
# dropY3 = tf.nn.dropout(poolY3, keep_prob, compatible_convolutional_noise_shape(poolY1))
YY = tf.reshape(poolY3, shape=[-1, 8*8*M])
Y4 = tf.nn.relu(
    tf.matmul(YY, W4) + B4)
Y4_drop = tf.nn.dropout(Y4, keep_prob)

Y = tf.nn.softmax(
    tf.matmul(Y4_drop, W5) + B5)

# keep_prob = tf.placeholder(tf.float32)
# Y1_drop = tf.nn.dropout(Y1, keep_prob

# placeholder value for the ground truth label of the network
Y_ = tf.placeholder(tf.float32, [None, 2])

batch_size = 64 # of 64

# loss function normalized for batchsize
cross_entropy = \
    tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y)) \
    * batch_size

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
num_pos = tf.reduce_sum(tf.cast(tf.argmax(Y, 1), tf.float32))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# training step
global_step = tf.Variable(0, trainable=False)
start_learning_rate = 0.001
learning_rate = \
    tf.train.exponential_decay(
        start_learning_rate, global_step, 10000, 0.90, staircase=True)

train_step = tf.train.AdamOptimizer(start_learning_rate) \
    .minimize(cross_entropy, global_step=global_step)

# make a session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

########################
# logging to tensorboard
with tf.name_scope("layer1"):
    tf.summary.histogram("weights", W1)
    tf.summary.histogram("activations", Y1)
    tf.summary.histogram("bias", B1)
with tf.name_scope("layer2"):
    tf.summary.histogram("weights", W2)
    tf.summary.histogram("activations", Y2)
    tf.summary.histogram("bias", Y2)
with tf.name_scope("layer3"):
    tf.summary.histogram("weights", W3)
    tf.summary.histogram("activations", Y3)
    tf.summary.histogram("bias", Y3)
with tf.name_scope("layer4"):
    tf.summary.histogram("weights", W4)
    tf.summary.histogram("activations", Y4)
    tf.summary.histogram("bias", Y4)
with tf.name_scope("layer5"):
    tf.summary.histogram("weights", W5)
    tf.summary.histogram("predictions", Y)
    tf.summary.histogram("bias", B5)

tf.summary.scalar('learning_rate', learning_rate)
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('num_pos', num_pos)

merged = tf.summary.merge_all()

# validation accuracy
acc_summary = tf.summary.scalar('test_accuracy', accuracy)
valid_summary_op = tf.summary.merge([acc_summary])
########################

# saving tensorboard logs
now = datetime.datetime.time(datetime.datetime.now())
date = now.strftime("%d %H:%M:%S")
train_writer = tf.summary.FileWriter('./logs/conv ' + date, sess.graph)

# make a random test set of the data with 50% pos and 50% neg
p = np.random.permutation(len(hists_test))
labels_test_shuffled = labels_test[p]
hists_test_shuffled = hists_test[p]
neg_test_idx = np.where(labels_test_shuffled[:,1]==0)[0]
pos_test_idx = np.where(labels_test_shuffled[:,1]==1)[0]

# limit size of positive set (pos > neg)
if pos_test_idx.shape[0] > neg_test_idx.shape[0]:
    pos_test_idx = pos_test_idx[0:neg_test_idx.shape[0]]
else:
    neg_test_idx = neg_test_idx[0:pos_test_idx.shape[0]]

hists_test = \
    hists_test_shuffled[np.concatenate((pos_test_idx, neg_test_idx))]
labels_test = \
    labels_test_shuffled[np.concatenate((pos_test_idx, neg_test_idx))]

for j in range(2000):
    #shuffle data in epochs
    accuracies = []
    entropy = []

    # make a random set of the data with 50% pos and 50% neg
    p = np.random.permutation(len(hists_train))
    hists_epoch = hists_train[p]
    labels_epoch = labels_train[p]
    neg_idx = np.where(labels_epoch[:,]==0)[0]
    pos_idx = np.where(labels_epoch[:,]==1)[0]

    # limit size of pos (pos > neg)
    if neg_idx.shape[0] > pos_idx.shape[0]:
        neg_idx = neg_idx[0:pos_idx.shape[0]]
    else:
        pos_idx = pos_idx[0:neg_idx.shape[0]]

    hists_epoch = hists_epoch[np.concatenate((pos_idx, neg_idx))]
    labels_epoch = labels_epoch[np.concatenate((pos_idx, neg_idx))]

    # shuffle epoch data
    p = np.random.permutation(len(hists_epoch))
    hists_epoch = hists_epoch[p]
    labels_epoch = labels_epoch[p]
    number_of_runs = int(np.floor(len(hists_epoch) / batch_size))

    # test if train and test set are equaly divided
    # print str(np.sum(labels_epoch)) + " of " + str(labels_epoch.shape[0])
    # print str(np.sum(np.argmax(labels_test, 1))) + " of " + str(labels_test.shape[0])
    for i in range(number_of_runs):
        batch_begin = int(i*batch_size)
        batch_end = int(np.min((i*batch_size+batch_size, len(hists)-1)))

        batch_X = hists_epoch[batch_begin:batch_end]
        batch_Y = labels_epoch[batch_begin:batch_end]

        # make one-hot array of labels
        batch_Y = np.squeeze(batch_Y)
        batch_Y = (batch_Y[:,None] != np.arange(2)).astype(float)
        train_data = {X: batch_X, Y_: batch_Y, keep_prob: 0.1}

        if i % 10 == 0:
            summary, weights, pred, batch_accuracy, batch_entropy = \
                sess.run(
                    [merged, W5, Y, accuracy, cross_entropy],
                    feed_dict=train_data)

            # if j > 1:
            #     print np.sum(weights-weights_old)
            # weights_old = weights

            train_writer.add_summary(summary, i+j*number_of_runs)
            accuracies.append(batch_accuracy)
            entropy.append(batch_entropy)

        sess.run(train_step, feed_dict=train_data)

    # Take the mean of you measure
    mean_accuracy = np.mean(accuracies)
    mean_entropy = np.mean(entropy)

    print("Mean accuracy of epoch (" + str(j + 1) + "): "
        + str(mean_accuracy))
    print("Mean entropy of epoch  (" + str(j + 1) + "): "
        + str(mean_entropy))

    test_num_pos, test_pred, test_accuracy, test_summary = \
        sess.run(
            [num_pos, Y, accuracy, valid_summary_op],
            feed_dict={X: hists_test, Y_: labels_test, keep_prob: 1.0})

    print("Test accuracy of epoch (" + str(j + 1) + "): "
        + str(test_accuracy) + ", pos: " + str(test_num_pos)
        + " / " + str(labels_test.shape[0]))

    print("")

    train_writer.add_summary(test_summary, i+j*number_of_runs)

#Create a saver object which will save all the variables
saver = tf.train.Saver()
save_path = saver.save(sess, "./models/conv.ckpt")
print("Model saved in file: %s" % save_path)

# first model:
# model_1_layer = 1 layer, 100 neurons, no dropout
# logs = "train 01 11:52:44"
# learning rate 0.01, every 100000 0.99
# batch_size 64
# 2000 epochs: 0.8 on train, 0.77 on test
# rescaling * 4

# first model
## depth in convolutional layers
# K = 12 # conv layer, depth = K, patchsize = 5, stride = 1
# pooling 2x2
# L = 16 # conv layer, depth = L, patchsize = 4, stride = 2 img=32
# pooling 2x2
# M = 24 # conv layer, depth = M, patchsize = 4, stride = 2 so img=16
# pooling 2x2
# N = 100 # fully connected layer 
# learning rate = 0.001
# batch_size 64
# dropout 0.1 on layer between connected and softmax
# Test set: 87%
