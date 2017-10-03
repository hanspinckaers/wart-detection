import numpy as np
import tensorflow as tf
import cv2
import pudb

hists_train = np.load("../3_svm_model/cached/train_cache_0.npy")
hists_test = np.load("../3_svm_model/cached/test_cache_0.npy")
labels_train = np.load("../3_svm_model/cached/train_cache_0_labels.npy")
# labels_test = np.load("../3_svm_model/test_cache_0_labels.npy")

# hists = np.concatenate((hists_train, hists_test), axis=0)
# labels = np.concatenate((labels_train, labels_test), axis=0)

hists = hists_train
labels = labels_train[:, np.newaxis]
print(len(hists))

X = tf.placeholder(tf.float32, [None, 500])
W = tf.Variable(tf.zeros([500, 1]))
b = tf.Variable(tf.zeros([1]))

init = tf.initialize_all_variables()

Y = tf.nn.sigmoid(tf.matmul(X, W) + b)
Y_ = tf.placeholder(tf.float32, [None, 1])

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y)) * 100

# % of correct answers found in batch
is_correct = tf.equal(tf.round(Y), Y_)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

# make a session
sess = tf.Session()
sess.run(init)

batch_size = 50

for i in range(100):
    #shuffle data in epochs
    accuracies = []
    p = np.random.permutation(len(hists))
    hists = hists[p]
    labels = labels[p]

    for i in range(int(np.floor(len(hists) / batch_size))):
        batch_X = hists[i*batch_size: int(np.min((i*batch_size+batch_size, len(hists) - 1)))]
        batch_Y = labels[i*batch_size: int(np.min((i*batch_size+batch_size, len(labels) - 1)))]
        train_data = {X: batch_X, Y_: batch_Y}
        # train
        sess.run(train_step, feed_dict=train_data)

        cor,a,c = sess.run([is_correct, accuracy, cross_entropy], feed_dict=train_data)
        accuracies.append(a)

    # Take the mean of you measure
    mean_accuracy = np.mean(accuracies)
    print mean_accuracy

