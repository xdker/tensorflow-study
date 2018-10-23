import tensorflow as tf
g1=tf.Graph()
with g1.as_default():
    #在计算图g1中定义变量“v”，并设置初始值为0。
    v=tf.get_variable(
        "v",shape=[1],initializer=tf.zeros_initializer)
g2=tf.Graph()
with g2.as_default():
    #在计算图g2中定义变量“v”，并设置初始值为0.
    v=tf.get_variable(
        "v",shape=[1],initializer=tf.zeros_initializer)
#在计算图g1中读取变量“v”的取值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))
with tf.Session(graph=g2) as sees:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print(sees.run(tf.global_variables("v")))