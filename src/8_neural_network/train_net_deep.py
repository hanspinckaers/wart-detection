import numpy as np
import tensorflow as tf
import cv2
import pudb
import datetime
import time
hists = np.load("../3_svm_model/cached/train_cache_0.npy")
hists_test = np.load("../3_svm_model/cached/test_cache_0.npy")
labels = np.load("../3_svm_model/cached/train_cache_0_labels.npy")
labels_test = np.load("../3_svm_model/test_cache_0_labels.npy")

labels = labels[:, np.newaxis]

# the data consist of normalized histograms so
# we can scale it by multiplications
hists *= 4
hists_test *= 4

# one-hot the label array ([0, 1, 0] -> [[1, 0], [0, 1], [1, 0]]
# labels_test = np.squeeze(labels_test)
labels_test = (labels_test[:,None] != np.arange(2)).astype(float)

# number of neurons in the fully connected layers
K = 100
L = 100
M = 20
N = 2

# placeholder variables for the layers
W1 = tf.Variable(tf.truncated_normal([500, K], stddev=0.1))
B1 = tf.Variable(tf.zeros([K]))

W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
B2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.truncated_normal([L, 2], stddev=0.1))
B5 = tf.Variable(tf.zeros([2]))

X = tf.placeholder(tf.float32, [None, 500])

with tf.name_scope("layer1"):
    Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
    tf.summary.histogram("weights", W1)
    tf.summary.histogram("activations", Y1)
    tf.summary.histogram("bias", B1)

    keep_prob = tf.placeholder(tf.float32)
    Y1_drop = tf.nn.dropout(Y1, keep_prob)

with tf.name_scope("layer2"):
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
    tf.summary.histogram("weights", W2)
    tf.summary.histogram("activations", Y2)
    tf.summary.histogram("bias", Y2)

    Y2_drop = tf.nn.dropout(Y2, keep_prob)

# with tf.name_scope("layer3"):
#    Y3 = tf.nn.relu(tf.matmul(Y2_drop, W3) + B3)
#    tf.summary.histogram("weights", W3)
#    tf.summary.histogram("activations", Y3)
#    tf.summary.histogram("bias", Y3)

#    Y3_drop = tf.nn.dropout(Y3, keep_prob)
# 
# with tf.name_scope("layer4"):
#     Y4 = tf.nn.sigmoid(tf.matmul(Y3_drop, W4) + B4)
#     tf.summary.histogram("weights", W4)
#     tf.summary.histogram("activations", Y4)
#     tf.summary.histogram("bias", Y4)

with tf.name_scope("layer5"):
    Y = tf.nn.softmax(tf.matmul(Y2_drop, W5) + B5)
    # Y5 = tf.nn.softmax(tf.matmul(Y3_drop, W5) + B5)
    # Y5 = tf.nn.softmax(tf.matmul(Y3_drop, W5) + B5)
    # Y5 = tf.nn.softmax(tf.matmul(Y3_drop, W5) + B5)
    tf.summary.histogram("weights", W5)
    tf.summary.histogram("predictions", Y)
    tf.summary.histogram("bias", B5)

# placeholder value for the ground truth label of the network
Y_ = tf.placeholder(tf.float32, [None, 2])

# loss function
cross_entropy = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_, logits=Y))
tf.summary.scalar('cross_entropy', cross_entropy)

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
num_pos = tf.reduce_sum(tf.cast(tf.argmax(Y, 1), tf.float32))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('num_pos', num_pos)

# training step, learning rate = 0.003
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = .01
learning_rate = tf.train.exponential_decay(starter_learning_rate,
    global_step, 100000, 0.99, staircase=True)
tf.summary.scalar('learning_rate', learning_rate)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    cross_entropy, global_step=global_step)

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

# make a session
sess = tf.InteractiveSession()

merged = tf.summary.merge_all()

# validation accuracy
acc_summary = tf.summary.scalar('test_accuracy', accuracy)
valid_summary_op = tf.summary.merge([acc_summary])

now = datetime.datetime.time(datetime.datetime.now())
date = now.strftime("%d %H:%M:%S")
train_writer = tf.summary.FileWriter('./logs/train ' + date, sess.graph)

tf.global_variables_initializer().run()

# make a random test set of the data with 50% pos and 50% neg
p = np.random.permutation(len(hists_test))
labels_test_shuffled = labels_test[p]
neg_test_idx = np.where(labels_test_shuffled[:,1]==0)[0]
pos_test_idx = np.where(labels_test_shuffled[:,1]==1)[0]

# limit size of positive set (pos > neg)
pos_test_idx = pos_test_idx[0:neg_test_idx.shape[0]]
hists_test = hists_test[np.concatenate((pos_test_idx, neg_test_idx))]
labels_test = labels_test[np.concatenate((pos_test_idx, neg_test_idx))]

batch_size = 64 # of 64

for j in range(60000):
    #shuffle data in epochs
    accuracies = []
    entropy = []

    # make a random set of the data with 50% pos and 50% neg
    p = np.random.permutation(len(hists))
    hists_epoch = hists[p]
    labels_epoch = labels[p]
    neg_idx = np.where(labels_epoch[:,]==0)[0]
    pos_idx = np.where(labels_epoch[:,]==1)[0]

    # limit size of pos (pos > neg)
    pos_idx = pos_idx[0:neg_idx.shape[0]]
    hists_epoch = hists_epoch[np.concatenate((pos_idx, neg_idx))]
    labels_epoch = labels_epoch[np.concatenate((pos_idx, neg_idx))]

    # shuffle epoch data
    p = np.random.permutation(len(hists_epoch))
    hists_epoch = hists_epoch[p]
    labels_epoch = labels_epoch[p]

    number_of_runs = int(np.floor(len(hists_epoch) / batch_size))
    for i in range(number_of_runs):
        batch_begin = int(i*batch_size)
        batch_end = int(np.min((i*batch_size+batch_size, len(hists)-1)))

        batch_X = hists_epoch[batch_begin:batch_end]
        batch_Y = labels_epoch[batch_begin:batch_end]

        # make one-hot array of labels
        batch_Y = np.squeeze(batch_Y)
        batch_Y = (batch_Y[:,None] != np.arange(2)).astype(float)

        train_data = {X: batch_X, Y_: batch_Y, keep_prob: 0.5}

        summary, batch_accuracy, batch_entropy = sess.run(
            [merged, accuracy, cross_entropy],
            feed_dict=train_data)

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

    test_num_pos, test_accuracy, test_summary = sess.run(
        [num_pos, accuracy, valid_summary_op],
        feed_dict={X: hists_test, Y_: labels_test, keep_prob : 1.0})

    print("Test accuracy of epoch (" + str(j + 1) + "): "
        + str(test_accuracy) + ", pos: " + str(test_num_pos)
        + " / " + str(labels_test.shape[0]))

    print("")

    train_writer.add_summary(test_summary, i+j*number_of_runs)

#Create a saver object which will save all the variables
saver = tf.train.Saver()
save_path = saver.save(sess, "./models/model_5_layer.ckpt")
print("Model saved in file: %s" % save_path)

# first model:
# model_1_layer = 1 layer, 100 neurons, no dropout
# logs = "train 01 11:52:44"
# learning rate 0.01, every 100000 0.99
# batch_size 64
# 2000 epochs: 0.8 on train, 0.77 on test
# rescaling * 4

# second model:
# model_2_layer = 2 layer, 100 neurons, no dropout
# logs = train 01 12:51:48
# learning rate 0.01, every 100000 0.99
# batch_size 64
# 2000 epochs: 0.8 on train, 0.77 on test
# rescaling * 4

# third model:
# model_3_layer = 2 layer, 100 neurons, 0.75 dropout 
# logs = train 01 14:37:43
# learning rate 0.01, every 100000 0.99
# batch_size 64
# 2000 epochs: 0.85 on train, 0.78 on test
# rescaling * 4
# only dropout second layer

# fourth model:
# model_4_layer = 2 layer, 100 neurons, 0.5 dropout 
# logs = train 01 16:45:31 
# learning rate 0.01, every 100000 0.99
# batch_size 64
# 2000 epochs: 0.85 on train, 0.78 on test
# rescaling * 4
# only dropout second layer

# fifth model:
# model_5_layer = 2 hidden layer, 100 neurons, 0.5 dropout 
# logs = train 01 16:45:31 
# learning rate 0.01, every 100000 0.99
# batch_size 64
# 60000 epochs: 0.93 on train, 0.77 on te
# rescaling * 4
# dropout first and second layer

# possible model:
# fast training model = 2 layer, 100 neurons, 0.5 dropout on all layers
# logs = train 01 15:45:19
# learning rate 0.03, every 100000 0.99
# batch_size 64
# 2000 epochs: 0.85 on train, 0.79 on test
# rescaling * 25

