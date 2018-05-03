import tensorflow as tf
import numpy as np

def generate_rand_img(batch_size, w, h, nc): # width, height, number of channels, TODO: make the mask actually a binary thing
    # return tf.random_uniform([batch_size, w, h, nc], minval=0, maxval=256, dtype=tf.float32)
    return np.zeros((batch_size, w, h, nc), dtype=np.float32)

'''
Input to the network is RGB image with binary channel indicating image completion (1 for pixel to be completed) (Iizuka)
Padding VALID: filter fits entirely, Padding SAME: preserves shape
'''

BATCH_SZ = 1

# Generator code
G_Z = tf.placeholder(tf.float32, shape=[BATCH_SZ, 512, 512, 4], name='G_Z')
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

G_Wdeconv4_1 = tf.get_variable('G_Wdeconv4_1', shape=[4, 4, 128, 256]) # NOTE: BACKWARDS, due to deconv
G_bdeconv4_1 = tf.get_variable('G_bdeconv4_1', shape=[128])

G_Wconv4_2 = tf.get_variable('G_Wconv4_2', shape=[3, 3, 128, 128])
G_bconv4_2 = tf.get_variable('G_bconv4_2', shape=[128])

G_Wdeconv5_1 = tf.get_variable('G_Wdeconv5_1', shape=[4, 4, 64, 128])
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
    G_a4_1 = tf.nn.conv2d_transpose(G_h3_9, G_Wdeconv4_1, strides=[1, 2, 2, 1], padding='SAME', output_shape=[BATCH_SZ, 256, 256, 128]) + G_bdeconv4_1
    G_h4_1 = tf.nn.relu(G_a4_1) # (None, 256, 256, 128)

    G_a4_2 = tf.nn.conv2d(G_h4_1, G_Wconv4_2, strides=[1, 1, 1, 1], padding='SAME') + G_bconv4_2
    G_h4_2 = tf.nn.relu(G_a4_2) # (None, 256, 256, 128)

    G_a5_1 = tf.nn.conv2d_transpose(G_h4_2, G_Wdeconv5_1, strides=[1, 2, 2, 1], padding='SAME', output_shape=[BATCH_SZ, 512, 512, 64]) + G_bdeconv5_1
    G_h5_1 = tf.nn.relu(G_a5_1) # (None, 512, 512, 64)

    G_a5_2 = tf.nn.conv2d(G_h5_1, G_Wconv5_2, strides=[1, 1, 1, 1], padding='SAME') + G_bconv5_2
    G_h5_2 = tf.nn.relu(G_a5_2) # (None, 512, 512, 32)

    G_a5_3 = tf.nn.conv2d(G_h5_2, G_Wconv5_3, strides=[1, 1, 1, 1], padding='SAME') + G_bconv5_3
    G_h5_3 = tf.nn.sigmoid(G_a5_3) # (None, 512, 512, 3)

    return G_h5_3

# Local Discriminator code
DL_X = tf.placeholder(tf.float32, shape=[BATCH_SZ, 128, 128, 3], name='DL_X')
DL_Wconv1 = tf.get_variable('DL_Wconv1', shape=[5, 5, 3, 64])
DL_bconv1 = tf.get_variable('DL_bconv1', shape=[64])

DL_Wconv2 = tf.get_variable('DL_Wconv2', shape=[5, 5, 64, 128])
DL_bconv2 = tf.get_variable('DL_bconv2', shape=[128])

DL_Wconv3 = tf.get_variable('DL_Wconv3', shape=[5, 5, 128, 256])
DL_bconv3 = tf.get_variable('DL_bconv3', shape=[256])

DL_Wconv4 = tf.get_variable('DL_Wconv4', shape=[5, 5, 256, 512])
DL_bconv4 = tf.get_variable('DL_bconv4', shape=[512])

DL_Wconv5 = tf.get_variable('DL_Wconv5', shape=[5, 5, 512, 512])
DL_bconv5 = tf.get_variable('DL_bconv5', shape=[512])

DL_Wdense6 = tf.get_variable('DL_Wdense6', shape=[8192, 1024])
DL_bdense6 = tf.get_variable('DL_bdense6', shape=[1024])

def local_discriminator(x): # Takes BATCH_SIZE x 128 x 128 x 3, outputs BATCH_SIZE x 1024
    pass

with tf.Session() as sess:
    sess.run(init_op)
    G_Z_ = generate_rand_img(1, 512, 512, 4)
    G_sample = generator(G_Z)
    output = sess.run([G_sample], feed_dict={G_Z: G_Z_})
    print(output[0].shape)
