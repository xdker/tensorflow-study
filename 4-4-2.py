import tensorflow as tf
import numpy as np
import matplotlib as plt


def get_weight(shape, lamada):
    # 生成对应一层的权重变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # 将生成的新变量的l2正则化损失项加入集合；第一个为集合名字，第二个参数为内容
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamada)(var))
    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
# 定义了每一层网络的节点个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)
# 这个变量维护向前传播时最深层的节点，开始的时候就是输入层
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]
# 通过一个循环来生成5层全连接的神经网络的结构
for i in range(1, n_layers):
    # layer_dimension[i]为下一层的节点个数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合中
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 使用ReLU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前，将下一层的节点个数更新为当前层节点的个数
    in_dimension = layer_dimension[i]
# 在定义神经网络的前向传播的同时已经将所有的L2正则化损失加入了图上的集合，这里需要计算刻画模型在训练数据上表现的损失函数
mes_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
# 将均方误差损失函数加入损失集合
tf.add_to_collection('losses', mes_loss)
# get_collection返回一个列表，这个列表是所有这个集合中的元素，在这个样例中，
# 这些元素就是损失函数的不同部分，将他们加起来就可以得到最终的损失函数
loss = tf.add_n(tf.get_collection('losses'))
