import numpy as np
import tensorflow as tf
import cv2
import pudb

hists = np.load("../3_svm_model/cached/train_cache_0.npy")
hists_test = np.load("../3_svm_model/cached/test_cache_0.npy")
labels = np.load("../3_svm_model/cached/train_cache_0_labels.npy")
labels_test = np.load("../3_svm_model/test_cache_0_labels.npy")

labels = labels[:, np.newaxis]
labels_test = labels_test[:, np.newaxis]

X = tf.placeholder(tf.float32, [None, 500])
W = tf.Variable(tf.zeros([500, 1]))
b = tf.Variable(tf.zeros([1]))

init = tf.initialize_all_variables()

# with one output neuron using a sigmoid is the same as using softmax
Y = tf.nn.sigmoid(tf.matmul(X, W) + b)
Y_ = tf.placeholder(tf.float32, [None, 1])

# loss function
# cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y)) 
# according to documentation
# tf.nn.softmax_cross_entropy_with_logits is more stable 
cross_entropy = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_, logits=Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.round(Y), Y_)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

# make a session
sess = tf.Session()
sess.run(init)

batch_size = 50

for j in range(10):
    #shuffle data in epochs
    accuracies = []
    p = np.random.permutation(len(hists))
    hists = hists[p]
    labels = labels[p]

    for i in range(int(np.floor(len(hists) / batch_size))):
        batch_begin = int(i * batch_size)
        batch_end = int(np.min((i * batch_size + batch_size, len(hists) - 1)))
        batch_X = hists[batch_begin:batch_end]
        batch_Y = labels[batch_begin:batch_end]
        train_data = {X: batch_X, Y_: batch_Y}

        sess.run(train_step, feed_dict=train_data)

        batch_correct, batch_accuracy, batch_entropy = sess.run(
            [is_correct, accuracy, cross_entropy],
            feed_dict=train_data)
        accuracies.append(batch_accuracy)

    # Take the mean of you measure
    mean_accuracy = np.mean(accuracies)
    print("Mean accuracy of epoch (" + str(j + 1) + "/10): " 
        + str(mean_accuracy))

accuracy_test = accuracy.eval(
    session=sess,
    feed_dict={X: hists_test, Y_: labels_test})

print "Accuracy on test: " + str(accuracy_test)
