# -*- coding: utf-8 -*-#
import os
import tensorflow as tf
import numpy as np
import matplotlib as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略tf的AVX提示
# -------------------------------------------------------------------------------
# Name:         5-4-6.py
# Author:       zlp
# Time:         2018/11/29
# Description:  
# -------------------------------------------------------------------------------
reader = tf.train.NewCheckpointReader('model/model.ckpt')
global_variables = reader.get_variable_to_shape_map()
for variables_name in global_variables:
    print(variables_name, global_variables[variables_name])
print("Value for variable v1 is", reader.get_tensor('v'))
