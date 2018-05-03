import tensorflow as tf
import numpy as np

def generate_rand_img(batch_size, w, h, nc): # width, height, number of channels, TODO: make the mask actually a binary thing
    # return tf.random_uniform([batch_size, w, h, nc], minval=0, maxval=256, dtype=tf.float32)
    return np.zeros((batch_size, w, h, nc), dtype=np.float32)

'''
Input to the network is RGB image with binary channel indicating image completion (1 for pixel to be completed) (Iizuka)
Padding VALID: filter fits entirely, Padding SAME: preserves shape
'''

Z = tf.placeholder(tf.float32, shape=[None, 512, 512, 4], name='Z')
G_Wconv1_1 = tf.get_variable('G_Wconv1_1', shape=[5, 5, 4, 64])
G_bconv1_1 = tf.get_variable('G_bconv1_1', shape=[64])

init_op = tf.global_variables_initializer()

def generator(z):
    G_a1_1 = tf.nn.conv2d(z, G_Wconv1_1, strides=[1, 1, 1, 1], padding='SAME') + G_bconv1_1 # TODO: CHECK PADDING
    G_h1_1 = tf.nn.relu(G_a1_1)
    return G_h1_1

with tf.Session() as sess:
    sess.run(init_op)
    Z_ = generate_rand_img(1, 512, 512, 4)
    G_sample = generator(Z)
    output = sess.run([G_sample], feed_dict={Z: Z_})
    print(output[0].shape)
