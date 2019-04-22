import tensorflow as tf
import numpy as np
x = np.array([[1,2,3,4], [1,2,3,4]])
y = np.array([[2,4,6,8], [2,4,6,8]])

res = tf.reduce_sum(tf.square(y-x), reduction_indices=[0])

sess = tf.Session()
print(sess.run(res))
