import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# y = AF(Wx)
# AF: relu, sigmoid, tanh
# 激励函数必须可微分
# 层数少，随便激活。层数多，可能梯度爆炸或梯度消失
# CNN: relu; RNN: tanh, relu


def add_layer(inputs, in_size, out_size, activation_function=None, n_layer=1):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/weights', Weights)
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='bias')  # bias初始值不为0
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), bias, name='res')
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name+'/outputs', outputs)
        return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')  # None表示样例数量不限制 这里注意dtype的表达
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu, n_layer=1)
prediction = add_layer(l1, 10, 1, activation_function=None, n_layer=2)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # lr=0.1

init = tf.initialize_all_variables()

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(r"./logs/", sess.graph)
    sess.run(init)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(x_data, y_data)
    # plt.ion()
    # plt.show()
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
    #         try:
    #             ax.lines.remove(lines[0])
    #         except Exception:
    #             pass
    #         prediction_value = sess.run(prediction, feed_dict={xs: x_data})
    #         lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
    #
    #         plt.pause(0.1)
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result, i)

# 以上为输入层，隐藏层，输出层

