import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 1000

donor_name = 'conv_net_3x3-61'
net_name = 'distil_linear'

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


#donor network

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

y_donor = tf.stop_gradient( tf.nn.softmax(model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)) )

# Acceptor network

def fc_layer(inp, size, name):
    W_layer = tf.Variable(tf.truncated_normal([inp.get_shape()[1].value, size],stddev=0.1))
    b_layer = tf.Variable(tf.constant(0, tf.float32, shape=[size]))
    res = tf.matmul(inp, W_layer) + b_layer
    tf.histogram_summary(name + "_weights", W_layer)
    return res, W_layer, b_layer

acceptor, A_w, A_b = fc_layer(tf.reshape(X, [-1,784]) , 10, "L")
y_acc = tf.nn.softmax( acceptor )
#L1 = tf.nn.sigmoid( layer(x, 1000, "L1") )
#y = tf.nn.softmax( layer(L1, 10, "L2") )

# distillation
cross_ent = tf.reduce_mean( - tf.reduce_sum(y_donor * tf.log(y_acc), reduction_indices=1))

distil_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_ent)

# Summaries
acc_prec = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_acc, 1), tf.argmax(Y, 1)), tf.float32))
donor_prec = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_donor, 1), tf.argmax(Y, 1)), tf.float32))

distil_writer = tf.train.SummaryWriter("logs/" + net_name + "/train", flush_secs=5)
distil_test_writer = tf.train.SummaryWriter("logs/" + net_name + "/test", flush_secs=5)

tf.scalar_summary('accuracy', acc_prec)
summaries = tf.merge_all_summaries()

# MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img



# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    # Donor loading
    donor_saver = tf.train.Saver([w,w2,w3,w4,w_o])
    donor_saver.restore(sess, 'checkpoints/' + donor_name)

    acc_saver = tf.train.Saver([A_w, A_b])

    k = 1
    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            val_prec, val_donor_prec, log_summaries, val_cross_ent, _ = sess.run([acc_prec, donor_prec,  summaries, cross_ent, distil_step],
                                                                 feed_dict={X: trX[start:end], Y: trY[start:end],
                                                                 p_keep_conv: 1, p_keep_hidden: 1 })

            test_val_prec, test_log_summaries = sess.run([acc_prec, summaries], feed_dict={X: teX[:test_size],
                                                Y: teY[:test_size],
                                                p_keep_conv: 1.0,
                                                p_keep_hidden: 1.0})

            distil_writer.add_summary(log_summaries, k)
            distil_test_writer.add_summary(test_log_summaries, k)
            print(i, k, 'distillation cross_ent:', val_cross_ent, 'donor_prec: ', val_donor_prec, '; train_prec', val_prec, '; test_prec', test_val_prec)

            k = k + 1

        acc_saver.save(sess, "checkpoints/" + net_name, global_step = i)
