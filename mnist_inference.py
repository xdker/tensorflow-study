# -*- coding: utf-8 -*-#
import os
import tensorflow as tf
import numpy as np
import matplotlib as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略tf的AVX提示
# -------------------------------------------------------------------------------
# Name:         mnist_inference.py
# Author:       zlp
# Time:         2018/11/29
# Description: 定义神经网络的前向传播的过程和参数
# -------------------------------------------------------------------------------
INPUT_NODE = 784
OUTPUT_NODE = 10
LAREY1_NODE = 500


def get_weight_variable(shape, regularizer):
    # 获取参数的形状并随机赋值
    weights = tf.get_variable(
        "weights", shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 将正则化损失加入loss集合
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


# 定义前向传播过程
def inference(input_tensor, regularizer):
    # 定义第一层神经网络
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
            [INPUT_NODE, LAREY1_NODE], regularizer)
        biases = tf.get_variable(
            "biases", [LAREY1_NODE], initializer=tf.constant_initializer(0.0)
        )
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    # 定义输出层
    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [LAREY1_NODE, OUTPUT_NODE], regularizer
        )
        biases = tf.get_variable(
            "biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0)
        )
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
