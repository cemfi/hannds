import logging

import tensorflow as tf
from data import Dataset

logging.basicConfig(level=logging.DEBUG)

# Import data
data = Dataset('./data')

PAST_SAMPLES = 50

# Create the model
x = tf.placeholder(tf.float32, [None, 88 * PAST_SAMPLES], name='input')
W = tf.Variable(tf.zeros([88 * PAST_SAMPLES, 88 * 2]))
b = tf.Variable(tf.zeros([88 * 2]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 88 * 2])
loss = tf.reduce_mean(tf.squared_difference(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for i in range(1000000):
    batch_xs, batch_ys = data.next_batch(100, past_samples=PAST_SAMPLES)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # if i % 100 == 0:
    print(sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))
