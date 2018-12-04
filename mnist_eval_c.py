# -*- coding: utf-8 -*-#
import os
import tensorflow as tf
import numpy as np
import matplotlib as plt
import time
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference_c
import mnist_train_c

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略tf的AVX提示
# -------------------------------------------------------------------------------
# Name:         mnist_eval.py
# Author:       zlp
# Time:         2018/11/30
# Description:  
# -------------------------------------------------------------------------------
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,
                           [mnist_train_c.BATCH_SIZE, mnist_inference_c.IMAGE_SIZE, mnist_inference_c.IMAGE_SIZE,
                            mnist_inference_c.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference_c.OUTPUT_NODE], name='y-input')
        reshaped_validation = np.reshape(
            mnist.validation.images,
            (mnist_train_c.BATCH_SIZE, mnist_inference_c.IMAGE_SIZE, mnist_inference_c.IMAGE_SIZE,
             mnist_inference_c.NUM_CHANNELS)
        )
        validate_feed = {x: reshaped_validation, y_: mnist.validation.labels}
        y = mnist_inference_c.inference(x, None,None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        validate_averages = tf.train.ExponentialMovingAverage(
            mnist_train_c.MOVING_AVERAGE_DECAY)
        validate_to_restore = validate_averages.variables_to_restore()
        saver = tf.train.Saver(validate_to_restore)
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train_c.MODEL_SAVE_PATH
                )
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('After %s training step(s),validation accuracy = %g' % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('/data/mnist_data', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
