# -*- coding: utf-8 -*-#
import os
import tensorflow as tf
import numpy as np
import matplotlib as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略tf的AVX提示
# -------------------------------------------------------------------------------
# Name:         5-4-5.py
# Author:       zlp
# Time:         2018/11/28
# Description:  
# -------------------------------------------------------------------------------
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(1.0, shape=[1]), name="v2")
result = v1 + v2
saver = tf.train.Saver()
saver.export_meta_graph("model/model.ckpt.mate.json")
