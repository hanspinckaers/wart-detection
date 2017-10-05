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

# fully connected layers
K = 100
L = 50
M = 30
N = 20

W1 = tf.Variable(tf.truncated_normal([500, K], stddev=100))
B1 = tf.Variable(tf.zeros([K]))

W2 = tf.Variable(tf.truncated_normal([K, L], stddev=50))
B2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.truncated_normal([L, M], stddev=5))
B3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.truncated_normal([N, 2], stddev=0.1))
B5 = tf.Variable(tf.zeros([2]))

X = tf.placeholder(tf.float32, [None, 500])

with tf.name_scope("layer1"):
    Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
    tf.summary.histogram("weights", W1)
    tf.summary.histogram("activations", Y1)

with tf.name_scope("layer2"):
    Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
    tf.summary.histogram("weights", W2)
    tf.summary.histogram("activations", Y2)

with tf.name_scope("layer3"):
    Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
    tf.summary.histogram("weights", W3)
    tf.summary.histogram("activations", Y3)

with tf.name_scope("layer4"):
    Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
    tf.summary.histogram("weights", W4)
    tf.summary.histogram("activations", Y4)

with tf.name_scope("layer5"):
    Y5 = tf.nn.softmax(tf.matmul(Y4, W5) + B5)
    tf.summary.histogram("weights", W5)
    tf.summary.histogram("predictions", Y5)
    tf.summary.histogram("bias", B5)

Y_ = tf.placeholder(tf.float32, [None, 2])

# loss function
# cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))
# according to documentation
# tf.nn.softmax_cross_entropy_with_logits is more stable
cross_entropy = tf.reduce_sum(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_, logits=Y5))
# tf.summary.scalar('cross_entropy', cross_entropy)

# classes_weights = tf.constant([1., 4.])
# cross_entropy = tf.reduce_mean(
#    tf.nn.weighted_cross_entropy_with_logits(
#        logits=Y5, targets=Y_, pos_weight=classes_weights))
tf.summary.scalar('cross_entropy', cross_entropy)

# % of correct answers found in batch
pred = Y5
is_correct = tf.equal(tf.argmax(Y5, 1), tf.argmax(Y_, 1))
num_pos = tf.reduce_sum(tf.cast(tf.argmax(Y5, 1), tf.float32))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('num_pos', num_pos)

# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train_step = optimizer.minimize(cross_entropy)

# training step, learning rate = 0.003
# global_step = tf.Variable(0, trainable=False)
# starter_learning_rate = 0.001
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                            100000, 0.96, staircase=True)

train_step = tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy)

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

# make a session
sess = tf.InteractiveSession()
# sess.run(init)

batch_size = 200

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./logs/train',
                                      sess.graph)

tf.global_variables_initializer().run()

for j in range(999999):
    #shuffle data in epochs
    accuracies = []
    entropy = []
    p = np.random.permutation(len(hists))
    hists_epoch = hists[p]
    labels_epoch = labels[p]
    neg_idx = np.where(labels_epoch[:,]==0)[0]
    pos_idx = np.where(labels_epoch[:,]==1)[0]
    pos_idx = pos_idx[0:neg_idx.shape[0]]
    hists_epoch = hists_epoch[np.concatenate((pos_idx, neg_idx))]
    labels_epoch = labels_epoch[np.concatenate((pos_idx, neg_idx))]

    p = np.random.permutation(len(hists_epoch))
    hists_epoch = hists_epoch[p]
    labels_epoch = labels_epoch[p]

    for i in range(int(np.floor(len(hists_epoch) / batch_size))):
        batch_begin = int(i * batch_size)
        batch_end = int(np.min((i * batch_size + batch_size, len(hists) - 1)))
        batch_X = hists_epoch[batch_begin:batch_end]
        batch_Y = labels_epoch[batch_begin:batch_end]
        batch_Y = np.squeeze(batch_Y)
        batch_Y = (batch_Y[:,None] != np.arange(2)).astype(float)
        train_data = {X: batch_X, Y_: batch_Y}

        summary, batch_num_pos, batch_predictions, batch_correct, batch_accuracy, batch_entropy = sess.run(
            [merged, num_pos, pred, is_correct, accuracy, cross_entropy],
            feed_dict=train_data)
        train_writer.add_summary(summary, i)
        accuracies.append(batch_accuracy)
        entropy.append(batch_entropy)
        sess.run(train_step, feed_dict=train_data)

    # Take the mean of you measure
    mean_accuracy = np.mean(accuracies)
    mean_entropy = np.mean(entropy)

    print("Mean accuracy of epoch (" + str(j + 1) + "): "
        + str(mean_accuracy))
    print("Mean accuracy of entropy (" + str(j + 1) + "): "
        + str(mean_entropy))

accuracy_test = accuracy.eval(
    session=sess,
    feed_dict={X: hists_test, Y_: labels_test})

print "Accuracy on test: " + str(accuracy_test)

