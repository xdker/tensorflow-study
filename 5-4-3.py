# -*- coding: utf-8 -*-#
import os
import tensorflow as tf
import numpy as np
import matplotlib as plt
from tensorflow.python.framework import graph_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略tf的AVX提示
# -------------------------------------------------------------------------------
# Name:         5-4-3.py
# Author:       zlp
# Time:         2018/11/28
# Description:  
# -------------------------------------------------------------------------------
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph_def, ['add']
    )
    with tf.gfile.GFile("model/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
