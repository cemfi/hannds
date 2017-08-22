import logging

import tensorflow as tf
from data import Dataset

import numpy as np

#logging.basicConfig(level=logging.DEBUG)

# Import data
data = Dataset('./data')

PAST_SAMPLES = 1

# Create the model
x = tf.placeholder(tf.float32, [None, 88 * PAST_SAMPLES], name='input')
W = tf.Variable(tf.zeros([88 * PAST_SAMPLES, 88]))
b = tf.Variable(tf.zeros([88]))
y = tf.tanh(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 88])
loss = tf.reduce_mean(tf.squared_difference(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#error rate calculation
masked_prediction = tf.multiply(x, y)
categories = tf.sign(masked_prediction)
errors = tf.cast(tf.not_equal(categories, y_), tf.float32)
num_errors = tf.reduce_sum(errors)
num_notes = tf.reduce_sum(tf.abs(y_))
error_rate = num_errors / num_notes

tf.summary.scalar('error rate', error_rate)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


writer = tf.summary.FileWriter("logs")

merged = tf.summary.merge_all()


# Train
TRAINING_STEPS = 100

for i in range(TRAINING_STEPS):
    batch_xs, batch_ys = data.next_batch(1000, past_samples=PAST_SAMPLES)
    summ, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})

    writer.add_summary(summ, global_step=i)


