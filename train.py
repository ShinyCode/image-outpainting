import tensorflow as tf
import numpy as np
from PIL import Image
import model2 as model
import util

'''
Input to the network is RGB image with binary channel indicating image completion (1 for pixel to be completed) (Iizuka)
Padding VALID: filter fits entirely, Padding SAME: preserves shape
'''

BATCH_SZ = 1

# Generator code
G_Z = tf.placeholder(tf.float32, shape=[BATCH_SZ, 64, 64, 4], name='G_Z')
DG_X = tf.placeholder(tf.float32, shape=[BATCH_SZ, 64, 64, 3], name='DG_X')

imgs = util.load_images(None)
imgs_p = util.preprocess_images(imgs)

# FOR DEBUGGING:
'''
util.vis_image(imgs[0])
util.vis_image(imgs_p[0,:,:,:3])
util.vis_image(imgs_p[0,:,:,3], mode='L')
util.save_image(imgs_p[0,:,:,:3], 'masked.png')
util.save_image(imgs_p[0,:,:,3], 'mask.png', mode='L')
'''

G_sample = model.generator(G_Z)
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

C_real = model.concatenator(model.global_discriminator(DG_X))
C_fake = model.concatenator(model.global_discriminator(G_sample))
vars_DG = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DG')
vars_C = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='C')

C_loss = -tf.reduce_mean(tf.log(C_real) + tf.log(1. - C_fake))
G_MSE_loss = tf.losses.mean_squared_error(G_sample, DG_X) # TODO: MULTIPLY with mask
ALPHA = 0.0004
G_loss = G_MSE_loss - ALPHA * tf.reduce_mean(tf.log(C_fake))

C_solver = tf.train.AdamOptimizer().minimize(C_loss, var_list=(vars_DG + vars_C))
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=vars_G)
G_MSE_solver = tf.train.AdamOptimizer().minimize(G_MSE_loss, var_list=vars_G)

N_ITERS = 300 # TODO: make this more than 1
N_ITERS_P1 = 100 # How many iterations to train in phase 1
N_ITERS_P2 = 100 # How many iterations to train in phase 2
INTV_PRINT = 10 # How often to print

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(N_ITERS):
        # TODO: Sample batches from training set
        G_sample_ = None
        C_loss_curr, G_loss_curr, G_MSE_loss_curr = None, None, None
        if i < N_ITERS_P1: # Stage 1 - Train Generator Only
            if i == 0:
                print('------------------> Beginning Phase 1...')
            _, G_MSE_loss_curr, G_sample_ = sess.run([G_MSE_solver, G_MSE_loss, G_sample], feed_dict={DG_X: imgs, G_Z: imgs_p})
        elif i < N_ITERS_P1 + N_ITERS_P2: # Stage 2 - Train Discriminator Only
            if i == N_ITERS_P1:
                print('------------------> Beginning Phase 2...')
            _, C_loss_curr = sess.run([C_solver, C_loss], feed_dict={DG_X: imgs, G_Z: imgs_p})
        else: # Stage 3 - Train both Generator and Discriminator
            if i == N_ITERS_P1 + N_ITERS_P2:
                print('------------------> Beginning Phase 3...')
            _, C_loss_curr = sess.run([C_solver, C_loss], feed_dict={DG_X: imgs, G_Z: imgs_p})
            _, G_loss_curr, G_sample_ = sess.run([G_solver, G_loss, G_sample], feed_dict={DG_X: imgs, G_Z: imgs_p})

        if i % INTV_PRINT == 0:
            if G_sample_ is not None:
                util.save_image(G_sample_[0], 'output/G%d.png' % i) # TODO: Sample images from output
            print('Iteration [%d/%d]:' % (i, N_ITERS))
            if G_MSE_loss_curr is not None:
                print('\tG_MSE_loss = %f' % G_MSE_loss_curr)
            if G_loss_curr is not None:
                print('\tG_loss = %f' % G_loss_curr)
            if C_loss_curr is not None:
                print('\tC_loss = %f' % C_loss_curr)
