import tensorflow as tf
import numpy as np
# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 +0.3

# create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))
y = Weights*x_data + bias

loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)  # lr = 0.5
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()  # 初始化变量
# create end

sess = tf.Session()  # 建立会话
sess.run(init)  # 激活init的步骤 sess.run 非常重要

for step in range(201):
    sess.run(train)  # 训练
    if step % 20 == 0:  # 每20步一输出
        print(step, sess.run(Weights), sess.run(bias))

