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

G_Wconv2_1 = tf.get_variable('G_Wconv2_1', shape=[3, 3, 64, 128])
G_bconv2_1 = tf.get_variable('G_bconv2_1', shape=[128])

G_Wconv2_2 = tf.get_variable('G_Wconv2_2', shape=[3, 3, 128, 128])
G_bconv2_2 = tf.get_variable('G_bconv2_2', shape=[128])

G_Wconv3_1 = tf.get_variable('G_Wconv3_1', shape=[3, 3, 128, 256])
G_bconv3_1 = tf.get_variable('G_bconv3_1', shape=[256])

G_Wconv3_2 = tf.get_variable('G_Wconv3_2', shape=[3, 3, 256, 256])
G_bconv3_2 = tf.get_variable('G_bconv3_2', shape=[256])

G_Wconv3_3 = tf.get_variable('G_Wconv3_3', shape=[3, 3, 256, 256])
G_bconv3_3 = tf.get_variable('G_bconv3_3', shape=[256])

G_Wdlconv3_4 = tf.get_variable('G_Wdlconv3_4', shape=[3, 3, 256, 256])
G_bdlconv3_4 = tf.get_variable('G_bdlconv3_4', shape=[256])

G_Wdlconv3_5 = tf.get_variable('G_Wdlconv3_5', shape=[3, 3, 256, 256])
G_bdlconv3_5 = tf.get_variable('G_bdlconv3_5', shape=[256])

G_Wdlconv3_6 = tf.get_variable('G_Wdlconv3_6', shape=[3, 3, 256, 256])
G_bdlconv3_6 = tf.get_variable('G_bdlconv3_6', shape=[256])

G_Wdlconv3_7 = tf.get_variable('G_Wdlconv3_7', shape=[3, 3, 256, 256])
G_bdlconv3_7 = tf.get_variable('G_bdlconv3_7', shape=[256])

G_Wconv3_8 = tf.get_variable('G_Wconv3_8', shape=[3, 3, 256, 256])
G_bconv3_8 = tf.get_variable('G_bconv3_8', shape=[256])

G_Wconv3_9 = tf.get_variable('G_Wconv3_9', shape=[3, 3, 256, 256])
G_bconv3_9 = tf.get_variable('G_bconv3_9', shape=[256])

G_Wdeconv4_1 = tf.get_variable('G_Wdeconv4_1', shape=[4, 4, 256, 128])
G_bdeconv4_1 = tf.get_variable('G_bdeconv4_1', shape=[128])

G_Wconv4_2 = tf.get_variable('G_Wconv4_2', shape=[3, 3, 128, 128])
G_bconv4_2 = tf.get_variable('G_bconv4_2', shape=[128])

G_Wdeconv5_1 = tf.get_variable('G_Wdeconv5_1', shape=[4, 4, 128, 64])
G_bdeconv5_1 = tf.get_variable('G_bdeconv5_1', shape=[64])

G_Wconv5_2 = tf.get_variable('G_Wconv5_2', shape=[3, 3, 64, 32])
G_bconv5_2 = tf.get_variable('G_bconv5_2', shape=[32])

G_Wconv5_3 = tf.get_variable('G_Wconv5_3', shape=[3, 3, 32, 3])
G_bconv5_3 = tf.get_variable('G_bconv5_3', shape=[3])

init_op = tf.global_variables_initializer()

def generator(z):
    G_a1_1 = tf.nn.conv2d(z, G_Wconv1_1, strides=[1, 1, 1, 1], padding='SAME') + G_bconv1_1 # TODO: CHECK PADDING
    G_h1_1 = tf.nn.relu(G_a1_1) # (None, 512, 512, 64)

    G_a2_1 = tf.nn.conv2d(G_h1_1, G_Wconv2_1, strides=[1, 2, 2, 1], padding='SAME') + G_bconv2_1
    G_h2_1 = tf.nn.relu(G_a2_1) # (None, 256, 256, 128)

    G_a2_2 = tf.nn.conv2d(G_h2_1, G_Wconv2_2, strides=[1, 1, 1, 1], padding='SAME') + G_bconv2_2
    G_h2_2 = tf.nn.relu(G_a2_2) # (None, 256, 256, 128)

    G_a3_1 = tf.nn.conv2d(G_h2_2, G_Wconv3_1, strides=[1, 2, 2, 1], padding='SAME') + G_bconv3_1
    G_h3_1 = tf.nn.relu(G_a3_1) # (None, 128, 128, 256)

    G_a3_2 = tf.nn.conv2d(G_h3_1, G_Wconv3_2, strides=[1, 1, 1, 1], padding='SAME') + G_bconv3_2
    G_h3_2 = tf.nn.relu(G_a3_2) # (None, 128, 128, 256)

    G_a3_3 = tf.nn.conv2d(G_h3_2, G_Wconv3_3, strides=[1, 1, 1, 1], padding='SAME') + G_bconv3_3
    G_h3_3 = tf.nn.relu(G_a3_3) # (None, 128, 128, 256)

    G_a3_4 = tf.nn.conv2d(G_h3_3, G_Wdlconv3_4, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 2, 2, 1]) + G_bdlconv3_4
    G_h3_4 = tf.nn.relu(G_a3_4) # (None, 128, 128, 256)

    G_a3_5 = tf.nn.conv2d(G_h3_4, G_Wdlconv3_5, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 4, 4, 1]) + G_bdlconv3_5
    G_h3_5 = tf.nn.relu(G_a3_5) # (None, 128, 128, 256)

    G_a3_6 = tf.nn.conv2d(G_h3_5, G_Wdlconv3_6, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 8, 8, 1]) + G_bdlconv3_6
    G_h3_6 = tf.nn.relu(G_a3_6) # (None, 128, 128, 256)

    G_a3_7 = tf.nn.conv2d(G_h3_6, G_Wdlconv3_7, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 16, 16, 1]) + G_bdlconv3_7
    G_h3_7 = tf.nn.relu(G_a3_7) # (None, 128, 128, 256)

    G_a3_8 = tf.nn.conv2d(G_h3_7, G_Wconv3_8, strides=[1, 1, 1, 1], padding='SAME') + G_bconv3_8
    G_h3_8 = tf.nn.relu(G_a3_8) # (None, 128, 128, 256)

    G_a3_9 = tf.nn.conv2d(G_h3_8, G_Wconv3_9, strides=[1, 1, 1, 1], padding='SAME') + G_bconv3_9
    G_h3_9 = tf.nn.relu(G_a3_9) # (None, 128, 128, 256)

    # NOTE: https://github.com/tensorflow/tensorflow/issues/2118 on why we need to put output shape
    G_a4_1 = tf.nn.conv2d_transpose(G_h3_9, G_Wdeconv4_1, strides=[1, 2, 2, 1], padding='SAME', output_shape=[None, 256, 256, 128]) + G_bdeconv4_1
    G_h4_1 = tf.nn.relu(G_a4_1) # (None, 256, 256, 128)

    return G_h4_1

with tf.Session() as sess:
    sess.run(init_op)
    Z_ = generate_rand_img(1, 512, 512, 4)
    G_sample = generator(Z)
    output = sess.run([G_sample], feed_dict={Z: Z_})
    print(output[0].shape)
