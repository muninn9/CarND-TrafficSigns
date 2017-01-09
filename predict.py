import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import os
import csv

#######################################################################################################
# GRAPH

EPOCHS = 30
BATCH_SIZE = 128
do = 0.5

def LeNet(x, dropout):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    print(x)

    # Layer 1: Convolutional. Input = 32x32x3.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 10), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(10))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 10, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)


    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits, fc2_W, fc3_W, fc2_b, fc3_b


#######################################################################################################
# PLACEHOLDERS

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)

#######################################################################################################
# PREDICTION

logits, fc2_W, fc3_W, fc2_b, fc3_b = LeNet(x, keep_prob)
image_dir = os.listdir("my_signs/")
saver = tf.train.Saver()

def read_image(image):
    return mpimg.imread("my_signs/" + image)


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('lenet.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))


    for image in image_dir:
        read = read_image(image)
        read = [read]
        predictions = sess.run(logits, feed_dict={x: read, keep_prob: 1.})
        # prediction = np.array(predictions[0]).argmax()
        top_3_predictions = sess.run(tf.nn.top_k(tf.constant(predictions[0]), k=3))
        with open('signnames.csv', 'r') as f:
            reader = csv.reader(f)
            sign_list = list(reader)

        for prediction in top_3_predictions.indices:
            print(sign_list[prediction + 1])

        print('-----------------------------------')

