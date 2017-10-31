import os
import sys

import matplotlib

import matplotlib.pyplot as plt
import tensorflow as tf
import datetime

from import_data import Dataset, convert

import numpy as np

if sys.platform == 'darwin':
    matplotlib.use('Agg')

# logging.basicConfig(level=logging.DEBUG)

LOG_PATH = os.path.join('logs', datetime.datetime.now().strftime('%H-%M'))


def get_figure():
    fig = plt.figure(num=0, figsize=(25, 2.7), dpi=72)
    fig.clf()
    return fig


def fig2rgb_array(fig, expand=True):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)


def figure_to_summary(fig):
    image = fig2rgb_array(fig)
    return image_summary.eval(feed_dict={image_placeholder: image})


# Import data
path = os.path.join('.', 'data')
convert(path, overwrite=False)
data = Dataset(path)

PAST_SAMPLES = 1

# Create the model
x = tf.placeholder(tf.float32, [None, 88 * PAST_SAMPLES], name='input')
W_1 = tf.Variable(tf.zeros([88 * PAST_SAMPLES, 88]))
W_2 = tf.Variable(tf.zeros([88, 1000]))
W_3 = tf.Variable(tf.zeros([1000, 88]))
b_1 = tf.Variable(tf.zeros([88]))
b_2 = tf.Variable(tf.zeros([1000]))
b_3 = tf.Variable(tf.zeros([88]))
h_1 = tf.sigmoid(tf.matmul(x, W_1) + b_1)
h_2 = tf.sigmoid(tf.matmul(h_1, W_2) + b_2)

y = tf.sigmoid(tf.matmul(h_2, W_3) + b_3)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 88], name='labels')  # -1 ... +1
truth = (y_ + 1.0) / 2.0
loss = tf.reduce_mean(tf.squared_difference(y, truth))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Calculate error rate
rescaled_y = (y * 2.0) - 1.0
masked_prediction = tf.multiply(rescaled_y, x)
categories = tf.sign(masked_prediction)
errors = tf.cast(tf.not_equal(categories, y_), tf.float32)
num_errors = tf.reduce_sum(errors)
num_notes = tf.reduce_sum(tf.abs(y_))
error_rate = num_errors / num_notes

# Summary
error_rate_summary = tf.summary.scalar('error_rate', error_rate)
image_placeholder = tf.placeholder(tf.uint8, fig2rgb_array(get_figure()).shape)
image_summary = tf.summary.image('output', image_placeholder)

writer = tf.summary.FileWriter(LOG_PATH)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # Train
    TRAINING_STEPS = 10000

    for i in range(TRAINING_STEPS):
        batch_xs, batch_ys = data.next_batch(400)
        batch_xs = batch_xs.reshape([400 * PAST_SAMPLES, 88])
        _, error_rate_sum, result, any = sess.run([
            train_step,
            error_rate_summary,
            masked_prediction,
            y
        ], feed_dict={x: batch_xs, y_: batch_ys})
        if i % 100 == 0:
            print('step = ', i)

        # Write output as image to summary
        fig = get_figure()
        plt.imshow(result.T, cmap='bwr', origin='lower', vmin=-1, vmax=1)
        plt.tight_layout()
        # writer.add_summary(figure_to_summary(fig), i)

        # Write other values to summary
        writer.add_summary(error_rate_sum, i)
