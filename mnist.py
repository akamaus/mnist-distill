import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

net_name = "std-0.1"
floatX = tf.float32

x = tf.placeholder(floatX, [None, 784])
y_true = tf.placeholder(floatX, [None, 10])

def layer(inp, size, name):
    W_layer = tf.Variable(tf.truncated_normal([inp.get_shape()[1].value, size],stddev=0.1))
    b_layer = tf.Variable(tf.constant(0, floatX, shape=[size]))
    res = tf.matmul(inp, W_layer) + b_layer
    tf.histogram_summary(name + "_weights", W_layer)
    return res

L1 = tf.nn.sigmoid( layer(x, 1000, "L1") )
y = tf.nn.softmax( layer(L1, 10, "L2") )

cross_ent = tf.reduce_mean( - tf.reduce_sum(y_true * tf.log(y), reduction_indices=1))


step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_ent)
accuracy = tf.reduce_mean(tf.cast( tf.equal(tf.argmax(y,1), tf.argmax(y_true,1)), floatX))
tf.scalar_summary('accuracy', accuracy)

sess = tf.Session()
init = tf.initialize_all_variables()

merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter("logs/" + net_name + "/train", sess.graph)
test_writer = tf.train.SummaryWriter("logs/" + net_name + "/test")

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

sess.run(init)

sav = tf.train.Saver()
#sav.restore(sess, "checkpoints-4900")

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    tlog, _ = sess.run([merged, step], feed_dict= {x: batch_xs, y_true: batch_ys})
    train_writer.add_summary(tlog, i)
    if i % 10 == 0:
        train_acc = sess.run(accuracy, feed_dict={x : batch_xs, y_true : batch_ys})
        log2, test_acc = sess.run([merged,accuracy], feed_dict={x : mnist.test.images, y_true : mnist.test.labels})
        test_writer.add_summary(log2, i)
        print(i, train_acc, test_acc)
    if i % 1000 == 0:
        sav.save(sess, "checkpoints/l2_full", global_step = i)


