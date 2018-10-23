import tensorflow as tf
a=tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0, 3.0], name="b")
result=tf.add(a,b,name="add")

#创建一个会话
sess=tf.Session()
print(sess.run(result))
sess.close()
#用上下文管理器来管理会话
with tf.Session().as_default():
    print(result.eval())
    #可以不用关闭会话
    