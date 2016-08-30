# -*- coding:utf-8 -*-
# Created by yanlei on 16-8-30 at 下午3:31.

"""
A Convolutional Network implementation example using Tensorflow library.
This example is using MNIST database of handwritten digits
"""
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784
n_classes = 10
dropout = 0.75

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)


# Create some wrappers for simplicity
def conv2d(x_, w_, b, strides=1):
    # Conv2d wrapper, with bias and relu activation
    x_ = tf.nn.conv2d(x_, w_, strides=[1, strides, strides, 1], padding='SAME')
    x_ = tf.nn.bias_add(x_, b)
    return tf.nn.relu(x_)


def max_pool2d(x_, k=2):
    return tf.nn.max_pool(x_, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x_, weights_, biases_, dropout_):
    # Reshape input picture
    x_ = tf.reshape(x_, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x_, weights_['wc1'], biases_['bc1'])
    conv1 = max_pool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights_['wc2'], biases_['bc2'])
    conv2 = max_pool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights_['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights_['wd1']), biases_['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, dropout_)

    out = tf.add(tf.matmul(fc1, weights_['out']), biases_['out'])
    return out


def plot_weight(w_):
    size = w_.shape
    size_filter = size[:2]
    size_big = int(math.ceil(np.sqrt(size[3])))
    size_small = int(math.ceil(np.sqrt(size[2])))
    image_shape = (size_filter[0] * size_big*size_small, size_filter[1] * size_big * size_small)
    image_ = np.ndarray(image_shape)
    for row in range(size_big):
        for col in range(size_big):
            for row_ in range(size_small):
                for col_ in range(size_small):
                    filter_ = w_[:, :, row_*col_, row*col]
                    image_[row_ * size_filter[0]:(row_ + 1) * size_filter[0],
                    col_ * size_filter[1]:(col_ + 1) * size_filter[1]] = filter_
    return image_

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # 1024 inputs, 10 outputs(class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = conv_net(x, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        if step % display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            print(
                "Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy="
                + "{:.5f}".format(acc))
        step += 1
        w = sess.run(weights['wc1'])
        w2 = sess.run(weights['wc2'])
        image = plot_weight(w)
        image2 = plot_weight(w2)
        plt.subplot(121)
        plt.imshow(image)
        plt.title("weight 1 step: {}".format(str(step)))
        plt.subplot(122)
        plt.imshow(image2)
        plt.title("weight 2 step: {}".format(str(step)))
        plt.pause(0.05)
    print("Optimization Finished")

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256],
                                                             keep_prob: 1.}))
