import logging

import tensorflow as tf
from data import Dataset

import numpy as np

logging.basicConfig(level=logging.DEBUG)

# Import data
data = Dataset('./data')

PAST_SAMPLES = 1

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
TRAINING_STEPS = 100000 #1000000

for i in range(TRAINING_STEPS):
    batch_xs, batch_ys = data.next_batch(1, past_samples=PAST_SAMPLES)

    # Problem: batch_ys are symmetric. Is there an error in data/__init__.py?
    symmetric = True
    isZero = True
    for j in range(88):
        symmetric = symmetric and batch_ys[0, j] == batch_ys[0, j + 88]
        isZero = isZero and batch_ys[0, j] == 0 and batch_ys[0, j + 88] == 0
    if not isZero:
        print("batch_ys is symmetric:", symmetric)
        print(batch_ys)


    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if i % 1000 == 0:
        print(sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))
        # Mask for relevant output
        mask = tf.tile(x, [1, 2]) # Does not work for PAST_SAMPLES other than 1
        masked_prediction = y * mask
        left_hand, right_hand = tf.split(masked_prediction, 2, 1)
        left_class = tf.cast(tf.greater(left_hand, right_hand), tf.float32)
        right_class = tf.cast(tf.greater(right_hand, left_hand), tf.float32)
        result = tf.concat([left_class, right_class], 1)
        errors = tf.cast(tf.not_equal(y_, result), tf.float32)
        num_errors = tf.reduce_sum(errors)
        notes = tf.reduce_sum(y_)
        error_rate = tf.divide(num_errors, notes) # Not yet tested!
        #print("error_rate: ", sess.run(error_rate, feed_dict={x: batch_xs, y_: batch_ys}))
        #print("y_: ", sess.run(y_, feed_dict={x: batch_xs, y_: batch_ys}))
        #print("result: ", sess.run(result, feed_dict={x: batch_xs, y_: batch_ys}))
        #print("left_hand: ", sess.run(left_hand, feed_dict={x: batch_xs, y_: batch_ys}))
        #print("right_hand: ", sess.run(right_hand, feed_dict={x: batch_xs, y_: batch_ys}))
        #print("masked_prediction: ", sess.run(masked_prediction, feed_dict={x: batch_xs, y_: batch_ys}))


#test_xs, test_ys = data.next_batch(100, past_samples=PAST_SAMPLES)

