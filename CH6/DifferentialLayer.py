import tensorflow as tf
import numpy as np
sess = tf.Session()

data_size = 25
data_1d = np.random.normal(size=data_size)
x_input_1d = tf.placeholder(dtype=tf.float32, shape=[data_size])

def conv_layer_1d(input_1d, my_filter):
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)

    convolution_output = tf.nn.conv2d(input_4d, filter = my_filter, strides=[1,1,1,1],padding="VALID")
    conv_output_1d = tf.squeeze(convolution_output)
    return conv_output_1d
conv_size = 5
my_filter = tf.Variable(tf.random_normal(shape=[1,conv_size,1,1]))
my_convolution_output = conv_layer_1d(x_input_1d, my_filter)

def activation(input_1d):
    return tf.nn.relu(input_1d)
my_activation_output = activation(my_convolution_output)

def max_pool(input_1d, width):
    input_2d = tf.expand_dims(input_1d, 0)
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    pool_output = tf.nn.max_pool(input_4d, ksize=[1,1,width,1], strides=[1,1,1,1], padding='VALID')
    poll_output_1d = tf.squeeze(pool_output)
    return(poll_output_1d)
my_maxpool_output = max_pool(my_activation_output, width = 5)

def fully_connected(input_layer, num_outputs):
    weight_shape = tf.squeeze(tf.stack([tf.shape(input_layer),[num_outputs]]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])
    input_layer_2d = tf.expand_dims(input_layer,0)
    full_output = tf.add(tf.matmul(input_layer_2d, weight), bias)
    full_output_1d = tf.squeeze(full_output)
    return full_output_1d
my_full_output = fully_connected(my_maxpool_output, 5)

init = tf.initialize_all_variables()
sess.run(init)
feed_dict = {x_input_1d:data_1d}
print('Input = array of length 25')
print('Convolution w/filter, length 5, stride size = 1, results in an array of length 21 : ')
print(sess.run(my_convolution_output, feed_dict=feed_dict))
print('\nInput = the above array of length 21')
print('ReLU element wise returns the array of length 21 : ')
print(sess.run(my_activation_output, feed_dict= feed_dict))
print('\n Input = the above array of length 21')
print(' MaxPool, window length = 5, stride size = 1, results in the array of length : 17')
print(sess.run(my_maxpool_output, feed_dict=feed_dict))
print('\n Input = the above array of length 17')
print('Fully Connected layer on all four rows with five output : ')
print(sess.run(my_full_output, feed_dict=feed_dict))

tf.reset_default_graph()
sess = tf.Session()

data_size = [10,10]
data_2d = np.random.normal(size=data_size)
x_input_2d = tf.placeholder(dtype=tf.float32, shape=data_size)

def conv_layer_2d(input_2d, my_filter):
    input_3d = tf.expand_dims(input_2d,0)
    input_4d = tf.expand_dims(input_3d, 3)
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1,2,2,1], padding="VALID")
    conv_output_2d = tf.squeeze(convolution_output)
    return conv_output_2d
my_filter = tf.Variable(tf.random_normal(shape=[2,2,1,1]))
my_convolution_output = conv_layer_2d(x_input_2d, my_filter)

def activation(input_2d):
    return tf.nn.relu(input_2d)
my_activation_output = activation(my_convolution_output)

def max_pool(input_2d, width, height):
    input_3d= tf.expand_dims(input_2d, 0 )
    input_4d = tf.expand_dims(input_3d, 3)
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, height, width, 1], strides=[1,1,1,1], padding="VALID")
    pool_output_2d = tf.squeeze(pool_output)
    return pool_output_2d
my_maxpool_output = max_pool(my_activation_output, width=2, height=2)

def fully_connected(input_layer, num_outputs):
    flat_input = tf.reshape(input_layer, [-1])
    weight_shape = tf.squeeze(tf.stack([tf.shape(flat_input), [num_outputs]]))
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])
    input_2d = tf.expand_dims(flat_input, 0)
    full_output = tf.add(tf.matmul(input_2d, weight), bias)
    full_output_2d = tf.squeeze(full_output)
    return full_output_2d
my_full_output = fully_connected(my_maxpool_output, 5)
#
init = tf.initialize_all_variables()
sess.run(init)
feed_dict = {x_input_2d:data_2d}
print('\n')
print('Input = [10 x 10] array')
print('2x2 Convolution, stride size = [2x2], result in the [5x5] array : ')
print(sess.run(my_convolution_output, feed_dict=feed_dict))
print('\nInput = the above [5x5] array')
print('ReLU element wise returns the [5x5] array : ')
print(sess.run(my_activation_output, feed_dict= feed_dict))
print('\n Input = the above [5x5] array')
print(' MaxPool, stride size = [1x1], results in the [4x4] array')
print(sess.run(my_maxpool_output, feed_dict=feed_dict))
print('\n Input = the above [4x4] array')
print('Fully Connected layer on all four rows with five output : ')
print(sess.run(my_full_output, feed_dict=feed_dict))