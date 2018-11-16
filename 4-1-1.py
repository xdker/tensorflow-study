import tensorflow as tf
import numpy as np
import matplotlib as plt
with tf.Session() as sess:
    v1=tf.constant([[1.1,2.0],[2.1,4.2]])
    v2=tf.constant([[5.1,4.2],[3.2,4.2]])
    v3=(v1*v2).eval()
    print(v3)
    v4=tf.matmul(v1,v2).eval()
    print(v4)
    w=tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])
    print(tf.reduce_mean(w).eval())
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=v2,logits=v1)
