import numpy as np
import tensorflow as tf

sess = tf.Session()
'''
identity_matrix = tf.diag([1.0, 1.0, 1.0])
A = tf.truncated_normal([2,3])
B = tf.fill([2,3],5.0)
C = tf.random_uniform([3,2])
D = tf.convert_to_tensor(np.array([[1.,2.,3.],[-3.,-7.,-1.],[0., 5., -2.]]))

print(sess.run(identity_matrix))
print(sess.run(A))
print(sess.run(B))
print(sess.run(C))
print(sess.run(tf.transpose(C)))
print(sess.run(D))

print(sess.run(tf.matmul(B, C)))

print(sess.run(tf.div(tf.sin(3.1416/4.), tf.cos(3.1416/4.))))

def custom_polynomial(value):
    # tf.sub() --> tf.subtract()
    return tf.subtract(3*tf.square(value),value)+tf.constant(10)
print(sess.run(custom_polynomial(11)))
'''

print(sess.run(tf.nn.relu([-3.,3.,10.])))
print(sess.run(tf.nn.sigmoid([-1.,0.,1.])))