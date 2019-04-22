import tensorflow as tf
import numpy as py

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    re = sess.run(output, feed_dict={input1: [7.], input2: [3.]})
    print(re)
