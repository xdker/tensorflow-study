import tensorflow as tf
#声明w1,w2两个变量。这里还通过seed参数设定了随机种子
w1=tf.Variable(tf.random_normal((2,3),stddev=1,seed=1))
w2=tf.Variable(tf.random_normal((3,1),stddev=1,seed=1))
x=tf.constant([[0.7,0.9]])
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
sess=tf.Session()
sess.run(w1.initializer)
sess.run(w2.initializer)
print(sess.run(y))
sess.close()

