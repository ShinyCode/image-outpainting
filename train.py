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
G_loss = -tf.reduce_mean(tf.log(C_fake))

C_solver = tf.train.AdamOptimizer().minimize(C_loss, var_list=(vars_DG + vars_C))
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=vars_G)

N_EPOCHS = 100
N_BATCHES = 1 # TODO: make this more than 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(N_EPOCHS):
        for i in range(N_BATCHES):
            # TODO: Sample batches from training set
            _, C_loss_curr, G_sample_ = sess.run([C_solver, C_loss, G_sample], feed_dict={DG_X: imgs, G_Z: imgs_p})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={G_Z: imgs_p})

            if epoch % 1 == 0:
                util.save_image(G_sample_[0], 'output/G%d.png' % epoch) # TODO: Sample images from output
                print('Epoch [%d/%d]:\n\tC_loss = %f\n\tG_loss = %f' % (epoch, N_EPOCHS, C_loss_curr, G_loss_curr))
