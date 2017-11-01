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

TRAINING_STEPS = 10000
BATCH_SIZE = 100
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
PAST_WINDOWS = 100
path = os.path.join('.', 'data')
convert(path, overwrite=False)
data = Dataset(path, n_windows_past=PAST_WINDOWS)


# Create the model
with tf.name_scope('preprocessing'):
    x_in = tf.placeholder(tf.float32, [None, 88 * (PAST_WINDOWS + 1)], name='input')
    x_no_nans = tf.where(tf.is_nan(x_in), tf.zeros_like(x_in), x_in, name='replace_nans_x')
    x_last_win = x_no_nans[:, -88:]

    # -1: left hand
    #  1: right hand
    #  0: not played or simultaneously plyaed by both hands

    y_labels = tf.placeholder(tf.float32, [None, 88], name='labels')  # -1 ... +1
    y_non_nans = tf.where(tf.is_nan(y_labels), tf.zeros_like(y_labels), y_labels, name='replace_nans_y')
    y_01 = (y_non_nans + 1.0) / 2.0 # needed for cross-entropy calculation

with tf.name_scope('nn'):
    b_last = tf.Variable(tf.concat([-1.0 * tf.ones([44]), +1.0 * tf.ones([44])], axis=0))
    W_last = tf.Variable(tf.truncated_normal([(PAST_WINDOWS + 1) * 88, 88], stddev=0.1))
    pre_last = tf.matmul(x_no_nans, W_last) + b_last

    # for squared error loss
    # y_output = tf.tanh(h_last)

    # for cross-entropy loss
    y_output = (tf.sigmoid(pre_last) * 2.0) - 1.0

    tf.summary.histogram("b_last", b_last)
    tf.summary.histogram("W_last", W_last)
    tf.summary.histogram("pre_last", pre_last)
    tf.summary.histogram("y_output", y_output)



with tf.name_scope('optimizer'):
    masked_predictions = tf.multiply(y_output, tf.abs(x_last_win)) # works only for PAST_WINDOWS == 0
    # squared error loss
    # loss = tf.losses.mean_pairwise_squared_error(y_labels_non_nans, masked_predictions)

    # DIY
    # loss = tf.reduce_mean(tf.squared_difference(y_labels_non_nans, masked_prediction))

    # cross-entropy = maximum-likelyhood loss
    y_output_01 = (y_output + 1.0) / 2.0
    EPSILON = 1E-6
    log_prob_right = tf.multiply(tf.log(y_output_01 + EPSILON), tf.abs(x_last_win))
    log_prob_left  = tf.multiply(tf.log(1.0 - y_output_01 + EPSILON), tf.abs(x_last_win))
    log_likely = tf.reduce_sum(tf.multiply(y_01, log_prob_right)) + tf.reduce_sum(tf.multiply(1.0 - y_01, log_prob_left))
    loss = -log_likely

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Calculate error rate
with tf.name_scope('evaluation'):
    categories = tf.sign(masked_predictions)
    errors = tf.cast(tf.not_equal(categories, y_non_nans), tf.float32)
    num_errors = tf.reduce_sum(errors)
    num_notes = tf.maximum(tf.reduce_sum(tf.abs(y_non_nans)), 1)
    error_rate = num_errors / num_notes

# Summaries
error_rate_summary = tf.summary.scalar('error_rate', error_rate)
log_likely_summary = tf.summary.scalar('log_likely', log_likely)
image_placeholder = tf.placeholder(tf.uint8, fig2rgb_array(get_figure()).shape)
# image_summary = tf.summary.image('output', image_placeholder)
merged_summary = tf.summary.merge_all();

with tf.Session() as sess:
    writer = tf.summary.FileWriter(LOG_PATH)
    writer.add_graph(sess.graph)
    tf.global_variables_initializer().run()
    # Train
    for i in range(TRAINING_STEPS + 1):
        batch_xs, batch_ys = data.next_batch(BATCH_SIZE)
        batch_xs = batch_xs.reshape([BATCH_SIZE, 88 * (PAST_WINDOWS + 1)])
        _, merged_sum, result, error_rate_value, output_wildcard, any = sess.run([
            train_step,
            merged_summary,
            masked_predictions,
            error_rate,
            b_last,
            y_output
        ], feed_dict={x_in: batch_xs, y_labels: batch_ys})
        if i % 100 == 0:
            print('step =', i, ", error_rate =", error_rate_value)
            # print('wildcard = \n', output_wildcard)

        # Write output as image to summary
        # fig = get_figure()
        # plt.imshow(result.T, cmap='bwr', origin='lower', vmin=-1, vmax=1)
        # plt.tight_layout()
        # writer.add_summary(figure_to_summary(fig), i)

        # Write other values to summary
        if i % 50 == 0:
            writer.add_summary(merged_sum, i)
