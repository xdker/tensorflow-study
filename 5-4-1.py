# -*- coding: utf-8 -*-#
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略tf的AVX提示
import tensorflow as tf
import numpy as np
import matplotlib as plt

# -------------------------------------------------------------------------------
# Name:         5-4-1.py
# Author:       zlp
# Time:         2018/11/28
# Description:  
# -------------------------------------------------------------------------------
v = tf.Variable(0, dtype=tf.float32, name="v")
# 在没有申明滑动平均模型时只有一个变量v，所以一下语句只会输出“v:0”
for variables in tf.global_variables():
    print(variables.name)
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
# 在申明滑动平均模型之后，TensorFlow会自动生成一个影子变量
# v/ExponentialMoving Average.于是一下语句会输出
# "v:0"和“v/ExponenttialMovingAverage:0”
for variables in tf.global_variables():
    print(variables.name)
saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存时，TensorFlow会将v：0和v/ExponentialMovingAverage:0两个变量保存下来
    saver.save(sess, "model/model.ckpt")
    print(sess.run([v, ema.average(v)]))
