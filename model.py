import tensorflow as tf
import numpy as np
from PIL import Image

def load_images(dir): # Outputs [m, 512, 512, 3]
    # TODO: change so it iterates over all images
    im = Image.open('images/city.jpg')
    # Crop it so it's 512 x 512, at the center.
    width, height = im.size
    left = (width - 512) / 2
    top = (height - 512) / 2
    im = im.crop((left, top, left + 512, top + 512))
    im_r = im.resize((256, 256))
    pix = np.array(im)
    pix_r = np.array(im_r)
    assert pix.shape == (512, 512, 3)
    assert pix_r.shape == (256, 256, 3)
    return pix[np.newaxis] / 255.0, pix_r[np.newaxis] / 255.0 # TODO: CHECK THIS

def preprocess_images(imgs): # Outputs [m, 512, 512, 4]
    # TODO: Randomize it, and decide how we're doing the mean
    pix_avg = np.mean(imgs)
    img = imgs[0] # TODO: Vectorize for entire thingy eventually
    img[64:-64, 64:-64, :] = pix_avg
    mask = np.zeros((512, 512, 1))
    mask[64:-64, 64:-64, :] = 1.0
    imgs_p = np.concatenate((img, mask), axis=2)
    return imgs_p[np.newaxis]

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

def generator(z):
    with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(
            inputs=z,
            filters=64,
            kernel_size=[5, 5],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv2_1 = tf.layers.conv2d(
            inputs=conv1,
            filters=128,
            kernel_size=[3, 3],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv2_2 = tf.layers.conv2d(
            inputs=conv2_1,
            filters=128,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv3_1 = tf.layers.conv2d(
            inputs=conv2_2,
            filters=256,
            kernel_size=[3, 3],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv3_2 = tf.layers.conv2d(
            inputs=conv3_1,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv3_3 = tf.layers.conv2d(
            inputs=conv3_2,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv3_4 = tf.layers.conv2d(
            inputs=conv3_3,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv3_5 = tf.layers.conv2d(
            inputs=conv3_4,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(4, 4),
            padding="same",
            activation=tf.nn.relu)

        conv3_6 = tf.layers.conv2d(
            inputs=conv3_5,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(8, 8),
            padding="same",
            activation=tf.nn.relu)

        conv3_7 = tf.layers.conv2d(
            inputs=conv3_6,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(16, 16),
            padding="same",
            activation=tf.nn.relu)

        conv3_8 = tf.layers.conv2d(
            inputs=conv3_7,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv3_9 = tf.layers.conv2d(
            inputs=conv3_8,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        deconv4_1 = tf.layers.conv2d_transpose(
            inputs=conv3_9,
            filters=128,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv4_2 = tf.layers.conv2d(
            inputs=deconv4_1,
            filters=128,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        deconv5_1 = tf.layers.conv2d_transpose(
            inputs=conv4_2,
            filters=64,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv5_2 = tf.layers.conv2d(
            inputs=deconv5_1,
            filters=32,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv5_3 = tf.layers.conv2d(
            inputs=conv5_2,
            filters=3,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.sigmoid)

    return conv5_3

# Local Discriminator code
DG_X = tf.placeholder(tf.float32, shape=[BATCH_SZ, 256, 256, 3], name='DG_X')

def global_discriminator(x):
    with tf.variable_scope('DG', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=128,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=256,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=512,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv5 = tf.layers.conv2d(
            inputs=conv4,
            filters=512,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv6 = tf.layers.conv2d(
            inputs=conv5,
            filters=512,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv6_flat = tf.layers.flatten(
            inputs=conv6)

        dense7 = tf.layers.dense(
            inputs=conv6_flat,
            units=1024,
            activation=tf.nn.relu)

    return dense7

def concatenator(global_x):
    with tf.variable_scope('C', reuse=tf.AUTO_REUSE):
        dense1 = tf.layers.dense(
            inputs=global_x,
            units=1,
            activation=tf.sigmoid)

    return dense1

imgs, imgs_r = load_images(None)
imgs_p = preprocess_images(imgs)

G_sample = generator(G_Z)
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

C_real = concatenator(global_discriminator(DG_X))
G_sample_r = tf.image.resize_images(G_sample, [256, 256]) # TODO: Check this isn't sketchy
C_fake = concatenator(global_discriminator(G_sample_r))
vars_DG = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DG')
vars_C = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='C')

with tf.Session() as sess:
    # TODO: Sample batches from training set
    C_loss = -tf.reduce_mean(tf.log(C_real) + tf.log(1. - C_fake))
    G_loss = -tf.reduce_mean(tf.log(C_fake))

    C_solver = tf.train.AdamOptimizer().minimize(C_loss, var_list=(vars_DG + vars_C))
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=vars_G)

    sess.run(tf.global_variables_initializer())

    _, C_loss_curr = sess.run([C_solver, C_loss], feed_dict={DG_X: imgs_r, G_Z: imgs_p})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={G_Z: imgs_p})
