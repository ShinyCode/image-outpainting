import tensorflow as tf
import numpy as np
from PIL import Image
import model
import util

'''
Input to the network is RGB image with binary channel indicating image completion (1 for pixel to be completed) (Iizuka)
Padding VALID: filter fits entirely, Padding SAME: preserves shape
'''

BATCH_SZ = 1

# Generator code
G_Z = tf.placeholder(tf.float32, shape=[BATCH_SZ, 512, 512, 4], name='G_Z')
DG_X = tf.placeholder(tf.float32, shape=[BATCH_SZ, 256, 256, 3], name='DG_X')

imgs, imgs_r = util.load_images(None)
imgs_p = util.preprocess_images(imgs)

G_sample = model.generator(G_Z)
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

C_real = model.concatenator(model.global_discriminator(DG_X))
G_sample_r = tf.image.resize_images(G_sample, [256, 256]) # TODO: Check this isn't sketchy
C_fake = model.concatenator(model.global_discriminator(G_sample_r))
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
