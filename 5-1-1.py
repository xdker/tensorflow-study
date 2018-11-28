# -*- coding: utf-8 -*-#
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略tf的AVX提示
import tensorflow as tf
import numpy as np
import matplotlib as plt
# -------------------------------------------------------------------------------
# Name:         5-1-1.py
# Author:       zlp
# Time:         2018/11/22
# Description:  
# -------------------------------------------------------------------------------
a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
c = a + b
# 通过log_device_placement参数来输出运行每一个运算的设备。
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print (sess.run(c))
