import numpy as np
import os
# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import cv2
import pudb
import datetime
import time
import sys
import fnmatch
import random
from imgaug import augmenters as iaa
import imgaug as ia
import matplotlib.pyplot as plt
from scipy import misc

sys.path.append('../3_svm_model/')
sys.path.append('../2_compare_detectors/')

from train import filenames_for_participants
from divide import divide_in

write_cache = False
random.seed(0)

if write_cache:
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

    hists_train = np.concatenate((train_set_neg, train_set_pos))
    print "Size of the original train data: " + str(hists_train.shape)
    # images_aug = seq1.augment_images(hists_train)
    # images1_aug = seq2.augment_images(np.concatenate((hists_train, images_aug)))
    # hists_train = np.concatenate((hists_train, images_aug, images1_aug))
    # labels_train = np.concatenate((labels_train, labels_train, labels_train, labels_train))

    print "Size of the augmented train data: " + str(hists_train.shape)
    print "Size of the augmented train labels: " + str(labels_train.shape)

    # hists_train = hists_train/255.0-0.5
    # hists_train = hists_train - np.mean(hists_train, axis=0)
    # hists_train = hists_train / 255.0

    labels_test = np.zeros(len(test_set_pos)+len(test_set_neg))
    labels_test[len(test_set_neg)-1:] = 1
    # hists_test = np.concatenate((test_set_neg, test_set_pos))/255.0-0.5
    hists_test = np.concatenate((test_set_neg, test_set_pos))
    # hists_test = hists_test - np.mean(hists_test, axis=0)
    hists_test = hists_test / 255.0 - 0.5

    # one-hot the label array ([0, 1, 0] -> [[1, 0], [0, 1], [1, 0]]
    labels_test = (labels_test[:,None] != np.arange(2)).astype(float)

    np.save("hists_test", hists_test)
    np.save("labels_test", labels_test)
    np.save("hists_train", hists_train)
    np.save("labels_train", labels_train)

else:
    hists_test = np.load("hists_test.npy")
    labels_test = np.load("labels_test.npy")
    hists_train = np.load("hists_train.npy")
    labels_train = np.load("labels_train.npy")

# depth in convolutional layers
K = 24 # conv layer, depth = K, patchsize = 5, stride = 1
L = 48 # conv layer, depth = L, patchsize = 4, stride = 2 so img=32
M = 96 # conv layer, depth = M, patchsize = 4, stride = 2 so img=16
N = 192 # conv layer, depth = M, patchsize = 4, stride = 2 so img=16
O = 200 # fully connected layer 
P = 200 # fully connected layer 

W1 = tf.Variable(tf.truncated_normal([3, 3, 3, K], stddev=2./(3*3*3)))
B1 = tf.Variable(tf.zeros([K]))
W2 = tf.Variable(tf.truncated_normal([3, 3, K, L], stddev=2./(3*3*3)))
B2 = tf.Variable(tf.zeros([L]))
W3 = tf.Variable(tf.truncated_normal([3, 3, L, M], stddev=2./(3*3*3)))
B3 = tf.Variable(tf.zeros([M]))
W4 = tf.Variable(tf.truncated_normal([3, 3, M, N], stddev=2./(3*3*3)))
B4 = tf.Variable(tf.zeros([N]))
W5 = tf.Variable(tf.truncated_normal([8 * 8 * N, O], stddev=2./(8*8*N)))
B5 = tf.Variable(tf.zeros([O]))
W6 = tf.Variable(tf.truncated_normal([O, P], stddev=2./O))
B6 = tf.Variable(tf.zeros([P]))
W7 = tf.Variable(tf.truncated_normal([P, 2], stddev=2./P))
B7 = tf.Variable(tf.zeros([2]))

batch_size = 32 # of 64
keep_rate_dropout = 0.3

keep_prob = tf.placeholder(tf.float32)

# Input placeholder
X = tf.placeholder(tf.float32, [None, 64, 64, 3])

# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    # adding the iteration prevents from averaging across non-existing iterations
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

Y1l = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
Y1 = tf.nn.relu(Y1bn)
# Y1 = tf.layers.max_pooling2d(Y1r, 2, 2)

Y2l = tf.nn.conv2d(Y1, W2, strides=[1, 2, 2, 1], padding='SAME')
Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
Y2 = tf.nn.relu(Y2bn)
#Y2 = tf.layers.max_pooling2d(Y2r, 2, 2)

Y3l = tf.nn.conv2d(Y2, W3, strides=[1, 2, 2, 1], padding='SAME')
Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3, convolutional=True)
Y3 = tf.nn.relu(Y3bn)
# Y3 = tf.layers.max_pooling2d(Y3r, 2, 2)

Y4l = tf.nn.conv2d(Y3, W4, strides=[1, 2, 2, 1], padding='SAME')
Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4, convolutional=True)
Y4 = tf.nn.relu(Y4bn)
# Y4 = tf.layers.max_pooling2d(Y3r, 2, 2)

YY = tf.reshape(Y4, shape=[-1, 8*8*N])

Y5l = tf.matmul(YY, W5)
Y5bn, update_ema5 = batchnorm(Y5l, tst, iter, B5)
Y5r = tf.nn.relu(Y5bn)
Y5 = tf.nn.dropout(Y5r, keep_prob)

Y6l = tf.matmul(Y5, W6)
Y6bn, update_ema6 = batchnorm(Y6l, tst, iter, B6)
Y6r = tf.nn.relu(Y6bn)
Y6 = tf.nn.dropout(Y6r, keep_prob)

Y = tf.nn.softmax(tf.matmul(Y6, W7) + B7)

update_ema = tf.group(update_ema1, update_ema2, \
    update_ema3, update_ema4, update_ema5, update_ema6)

# placeholder value for the ground truth label of the network
Y_ = tf.placeholder(tf.float32, [None, 2])

# loss function normalized for batchsize
cross_entropy = \
    tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y)) \
#    + 0.0000125 * tf.nn.l2_loss(W1) \
#    + 0.0000125 * tf.nn.l2_loss(W2) \
#    + 0.0000125 * tf.nn.l2_loss(W3) \
#    + 0.0000125 * tf.nn.l2_loss(W4) \
#    + 0.0000125 * tf.nn.l2_loss(W5) \
#    + 0.0000125 * tf.nn.l2_loss(W6) \
#    + 0.0000125 * tf.nn.l2_loss(W7) \
# approximately 10(lambda)/80000(n of images)

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
num_pos = tf.reduce_sum(tf.cast(tf.argmax(Y, 1), tf.float32))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# training step
global_step = tf.Variable(0, trainable=False)
start_learning_rate = 0.0005
learning_rate = tf.maximum(0.00005, \
    tf.train.exponential_decay(
        start_learning_rate, global_step, 1000, 0.95, staircase=True))

train_step = tf.train.AdamOptimizer(learning_rate) \
    .minimize(cross_entropy, global_step=global_step, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

# make a session
init = tf.global_variables_initializer()
sess = tf.Session()
run_metadata = tf.RunMetadata()
sess.run(init, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)

# Print to stdout an analysis of the memory usage and the timing information
# from running the graph broken down by operations.
tf.contrib.tfprof.model_analyzer.print_model_analysis(
    tf.get_default_graph(),
    run_meta=run_metadata,
    tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)

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
    tf.summary.histogram("predictions", Y5)
    tf.summary.histogram("bias", B5)
with tf.name_scope("layer6"):
    tf.summary.histogram("weights", W6)
    tf.summary.histogram("predictions", Y6)
    tf.summary.histogram("bias", B6)
with tf.name_scope("layer7"):
    tf.summary.histogram("weights", W7)
    tf.summary.histogram("predictions", Y)
    tf.summary.histogram("bias", B7)

tf.summary.scalar('learning_rate', learning_rate)
tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('num_pos', num_pos)
# tf.summary.scalar('moving average1', update_ema1)
# tf.summary.scalar('moving average2', update_ema2)
# tf.summary.scalar('moving average3', update_ema3)
# tf.summary.scalar('moving average4', update_ema4)
merged = tf.summary.merge_all()

# validation accuracy
acc_summary = tf.summary.scalar('test_accuracy', accuracy)
valid_summary_op = tf.summary.merge([acc_summary])
########################

# saving tensorboard logs
now = datetime.datetime.time(datetime.datetime.now())
date = now.strftime("%a %H:%M:%S")
train_writer = tf.summary.FileWriter('./logs2/conv ' + date, sess.graph)

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

#finalize graph to catch memory leaks
sess.graph.finalize()
with tf.Graph().as_default():
    tf.set_random_seed(42)

###################################3
###################################
# make a sample to debug fast
# p = np.random.permutation(len(hists_train))
# hists_train = hists_train[p][0:3000]
# labels_train = labels_train[p][0:3000]

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

imgaug_seq = iaa.Sequential([
    iaa.Sharpen(alpha=(0, 0.3), lightness=(0.75, 1.0)),
    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    sometimes(iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    ))
])

# Use this to show augmented images
# fig = plt.figure(figsize=(50, 50))  # width, height in inches
# pos_idx = np.where(labels_train[:,]==1)[0]
# tstimages = hists_train[pos_idx[0:64]]
# for i in range(64):
#     tstimages[i] = cv2.cvtColor(tstimages[i], cv2.COLOR_BGR2RGB)
# tstimages = imgaug_seq.augment_images(tstimages)
# misc.imshow(ia.draw_grid(tstimages, cols=8))

for j in range(10000):
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
    number_of_runs = np.ceil(len(hists_epoch) / batch_size)
    # test if train and test set are equaly divided
    # print str(np.sum(labels_epoch)) + " of " + str(labels_epoch.shape[0])
    # print str(np.sum(np.argmax(labels_test, 1))) + " of " + str(labels_test.shape[0])
    hists_epoch = imgaug_seq.augment_images(hists_epoch)
    hists_epoch = hists_epoch / 255. - 0.5 # normalize after augmenting

    for i in range(number_of_runs.astype(int)):
        batch_begin = int(i*batch_size)
        batch_end = int(np.min((i*batch_size+batch_size, len(hists_epoch)-1)))

        batch_X = hists_epoch[batch_begin:batch_end]
        batch_Y = labels_epoch[batch_begin:batch_end]

        # make one-hot array of labels
        batch_Y = np.squeeze(batch_Y)
        batch_Y = (batch_Y[:,None] != np.arange(2)).astype(float)
        train_data = {X: batch_X, Y_: batch_Y, keep_prob: keep_rate_dropout, \
            tst: False, iter: i+j*number_of_runs}

        if i > -1:
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
        sess.run(update_ema, {X: batch_X, Y_: batch_Y, keep_prob: 1.0, \
             tst: False, iter: i+j*number_of_runs})

    # Take the mean of you measure
    mean_accuracy = np.mean(accuracies)
    mean_entropy = np.mean(entropy)

    print("Mean accuracy of epoch (" + str(j + 1) + "): "
        + str(mean_accuracy))
    print("Mean entropy of epoch  (" + str(j + 1) + "): "
        + str(mean_entropy))

    # run the test in batches, the network is so large that there is not enough
    # memory for all the test data at once
    number_of_runs = np.ceil(len(hists_test) / batch_size)
    test_accuracies = []
    for i in range(number_of_runs.astype(int)):
        batch_begin = int(i*batch_size)
        batch_end = int(np.min((i*batch_size+batch_size, len(hists_test)-1)))

        batch_X = hists_test[batch_begin:batch_end]
        batch_Y = labels_test[batch_begin:batch_end]

        test_num_pos, test_pred, test_accuracy, test_summary = \
            sess.run(
                [num_pos, Y, accuracy, valid_summary_op],
                feed_dict={X: batch_X, Y_: batch_Y, keep_prob: 1.0, \
                    tst: True, iter: i+j*number_of_runs})

        test_accuracies.append(test_accuracy)

    mean_test_accuracy = np.mean(test_accuracies)
    summary = tf.Summary(value=[
        tf.Summary.Value(tag='test_accuracy', simple_value=mean_test_accuracy)])
    train_writer.add_summary(summary, i+j*number_of_runs)

    # print("Test accuracy of epoch (" + str(j + 1) + "): "
    #     + str(mean_test_accuracy) + ", pos: " + str(test_num_pos)
    #    + " / " + str(labels_test.shape[0]))
    print("Test accuracy of epoch (" + str(j + 1) + "): " + str(mean_test_accuracy))
    print("")


# Create a saver object which will save all the variables
saver = tf.train.Saver()
save_path = saver.save(sess, "./models/conv3.ckpt")
print("Model saved in file: %s" % save_path)

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
# no batch norm
# dropout 0.1 on layer between connected and softmax
# Test set: 88% in 200 epochs
# conv 01 16:51:39

# second model, ran for 2000 epochs, no improvement
## depth in convolutional layers
# K = 12 # conv layer, depth = K, patchsize = 5, stride = 1
# pooling 2x2
# L = 16 # conv layer, depth = L, patchsize = 4, stride = 2 img=32
# pooling 2x2
# M = 24 # conv layer, depth = M, patchsize = 4, stride = 2 so img=16
# pooling 2x2
# N = 100 # fully connected layer 
# learning rate = 0.0005
# batch_size 128
# no batch norm
# dropout 0.05 on layer between connected and softmax
# Test set: 87-88%
# seems to overfit less
# conv 01 18:29:09

# third model
# depth in convolutional layers
# K = 6 # conv layer, depth = K, patchsize = 5, stride = 1
# L = 12 # conv layer, depth = L, patchsize = 4, stride = 2 so img=32
# M = 18 # conv layer, depth = M, patchsize = 4, stride = 2 so img=16
# N = 200 # fully connected layer 
# dropout 0.1 on layer between connected and softmax
# learning rate = 0.003, min 0.00001, decay 0.9 per 1000 global steps
# batch size 128
# test accuracy stabilizes around 86.6%, but reaches 87.3% in the beginning
# train accuracy is around 96%
# 200 epochs
# conv 01 19:15:33

# fourth model
# stddev weight 0.03
# depth in convolutional layers
# K = 6 # conv layer, depth = K, patchsize = 9, stride = 1
# L = 12 # conv layer, depth = L, patchsize = 4, stride = 2 so img=32
# M = 18 # conv layer, depth = M, patchsize = 4, stride = 2 so img=16
# N = 200 # fully connected layer (think it was 100)
# dropout 0.1 on layer between connected and softmax
# learning rate = 0.003, min 0.00005, decay 0.9 per 1000 global steps
# batch size 32
# test accuracy stabilizes around 87.5%, really nice stable curve
# train accuracy is around 91%
# 200 epochs
# conv Mon 10:23:17

# IDEA: use network above and only flip images!

# Notes:
# Seems like batch norm makes the curve less stable (test curve).
# Seems like larger batch size decreases overfitting
# High dropout is needed for our data (maybe too big network or noisy data)
# after data augementatino is dropout not needed that much
# max pooling removal on first layer improves accuracy.
