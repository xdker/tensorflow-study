# -*- coding: utf-8 -*-#
import os
import tensorflow as tf
import numpy as np
import matplotlib as plt
from tensorflow.python.platform import gfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略tf的AVX提示
# -------------------------------------------------------------------------------
# Name:         5-4-4.py
# Author:       zlp
# Time:         2018/11/28
# Description:  
# -------------------------------------------------------------------------------
with tf.Session() as sess:
    model_filename = "model/combined_model.pb"
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    result = tf.import_graph_def(graph_def, return_elements=['add:0'])
    print(sess.run(result))
