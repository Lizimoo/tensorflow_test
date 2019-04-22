import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None, n_layer=1):
    layer_name = 'layer%s' % n_layer
    # with tf.name_scope('layer'):
        # with tf.name_scope('Weights'):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            # tf.summary.histogram(layer_name+'/weights', Weights)
        # with tf.name_scope('bias'):
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='bias')  # bias初始值不为0
        # with tf.name_scope('Wx_plus_b'):
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), bias, name='res')
    if activation_function is None:
            outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
            # tf.summary.histogram(layer_name+'/outputs', outputs)
    return outputs


# def compute_accuracy(v_xs, v_ys):
#     global y
#     y_pre = sess.run(y, feed_dict={xs: v_xs})
#     correct_prediction = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
#     return result

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# y = tf.nn.softmax(tf.matmul(xs, W2) + b2)
dense = tf.layers.dense(inputs=xs, units=784, activation=tf.nn.relu)
pre_y = tf.nn.softmax(tf.matmul(dense, W) + b)
# l1 = add_layer(xs, 784, 2000, activation_function=tf.nn.softmax)
# prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

cross_entropy = -tf.reduce_sum(ys*tf.log(pre_y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # lr不能太高

correct_prediction = tf.equal(tf.argmax(pre_y, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20000):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i % 200 == 0:
            print(sess.run(accuracy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels}))
