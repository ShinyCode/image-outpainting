import tensorflow as tf
import numpy as np
from PIL import Image
import model3 as model
import util
import os

'''
Input to the network is RGB image with binary channel indicating image completion (1 for pixel to be completed) (Iizuka)
Padding VALID: filter fits entirely, Padding SAME: preserves shape
'''

# np.random.seed(0)
# tf.set_random_seed(0)

BATCH_SZ = 16
VERBOSE = True
EPSILON = 1e-9
IMAGE_SZ = 128
OUT_DIR = 'output'

if os.path.isdir(OUT_DIR) and len(os.listdir(OUT_DIR)) > 1:
    print('Warning, OUT_DIR already exists. Aborting.')
    exit()

if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

# Generator code
G_Z = tf.placeholder(tf.float32, shape=[None, IMAGE_SZ, IMAGE_SZ, 4], name='G_Z')
DG_X = tf.placeholder(tf.float32, shape=[None, IMAGE_SZ, IMAGE_SZ, 3], name='DG_X')

imgs = np.load('places/places_128_small.npy') # Originally from http://data.csail.mit.edu/places/places365/val_256.tar
imgs_p = util.preprocess_images_outpainting(imgs)

test_img = imgs[0, np.newaxis]
test_img_p = imgs_p[0, np.newaxis]

'''
imgs = util.load_city_image()
imgs_p = util.preprocess_images_outpainting(imgs)

test_img = imgs.copy()
test_img_p = imgs_p.copy()
'''

util.save_image(test_img[0], os.path.join(OUT_DIR, 'test_img.png'))

imgs = imgs[1:]
imgs_p = imgs_p[1:]

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

# http://www.cvc.uab.es/people/joans/slides_tensorflow/tensorflow_html/gan.html
C_loss = -tf.reduce_mean(tf.log(tf.maximum(C_real, EPSILON)) + tf.log(tf.maximum(1. - C_fake, EPSILON)))
G_MSE_loss = tf.losses.mean_squared_error(G_sample, DG_X, weights=tf.expand_dims(G_Z[:,:,:,3], -1)) # TODO: MULTIPLY with mask. Actually see if we want to remove this.
ALPHA = 0.0004
G_loss = G_MSE_loss - ALPHA * tf.reduce_mean(tf.log(tf.maximum(C_fake, EPSILON)))

C_solver = tf.train.AdamOptimizer().minimize(C_loss, var_list=(vars_DG + vars_C))
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=vars_G)
G_MSE_solver = tf.train.AdamOptimizer().minimize(G_MSE_loss, var_list=vars_G)

N_ITERS = 3000
N_ITERS_P1 = 1000 # How many iterations to train in phase 1
N_ITERS_P2 = 400 # How many iterations to train in phase 2
INTV_PRINT = 20 # How often to print

train_MSE_loss = []
dev_MSE_loss = []

last_output_PATH = None

assert N_ITERS > N_ITERS_P1 + N_ITERS_P2

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(N_ITERS):
        # TODO: Sample batches from training set
        batch, batch_p = util.sample_random_minibatch(imgs, imgs_p, BATCH_SZ)
        G_sample_ = None
        C_loss_curr, G_loss_curr, G_MSE_loss_curr = None, None, None
        if i < N_ITERS_P1: # Stage 1 - Train Generator Only
            if i == 0:
                print('------------------> Beginning Phase 1...')
            _, G_MSE_loss_curr, G_sample_ = sess.run([G_MSE_solver, G_MSE_loss, G_sample], feed_dict={DG_X: batch, G_Z: batch_p})
        elif i < N_ITERS_P1 + N_ITERS_P2: # Stage 2 - Train Discriminator Only
            if i == N_ITERS_P1:
                print('------------------> Beginning Phase 2...')
            _, C_loss_curr, C_real_, C_fake_ = sess.run([C_solver, C_loss, C_real, C_fake], feed_dict={DG_X: batch, G_Z: batch_p})
            if VERBOSE:
                print((i, C_loss_curr, np.min(C_real_), np.max(C_real_), np.min(C_fake_), np.max(C_fake_)))
        else: # Stage 3 - Train both Generator and Discriminator
            if i == N_ITERS_P1 + N_ITERS_P2:
                print('------------------> Beginning Phase 3...')
            _, C_loss_curr, C_real_, C_fake_ = sess.run([C_solver, C_loss, C_real, C_fake], feed_dict={DG_X: batch, G_Z: batch_p})
            if VERBOSE:
                print((i, C_loss_curr, 'D', np.min(C_real_), np.max(C_real_), np.min(C_fake_), np.max(C_fake_)))
            _, G_loss_curr, G_MSE_loss_curr, G_sample_, C_fake_ = sess.run([G_solver, G_loss, G_MSE_loss, G_sample, C_fake], feed_dict={DG_X: batch, G_Z: batch_p})
            if VERBOSE:
                print((i, G_loss_curr, 'G', np.min(C_fake_), np.max(C_fake_)))


        if i % INTV_PRINT == 0:
            G_MSE_loss_curr_dev = None
            if G_sample_ is not None:
                output, G_MSE_loss_curr_dev = sess.run([G_sample, G_MSE_loss], feed_dict={DG_X: test_img, G_Z: test_img_p})
                util.save_image(output[0], os.path.join(OUT_DIR, 'G%d.png' % i))
                last_output_PATH = os.path.join(OUT_DIR, 'G%d.png' % i)
            print('Iteration [%d/%d]:' % (i, N_ITERS))
            if G_MSE_loss_curr is not None:
                print('\tG_MSE_loss (train) = %f' % G_MSE_loss_curr)
            if G_MSE_loss_curr_dev is not None:
                print('\tG_MSE_loss (dev) = %f' % G_MSE_loss_curr_dev)
            if G_loss_curr is not None:
                print('\tG_loss = %f' % G_loss_curr)
            if C_loss_curr is not None:
                print('\tC_loss = %f' % C_loss_curr)

        # Keep track of losses for logging
        if G_MSE_loss_curr is not None:
            train_MSE_loss.append([i, G_MSE_loss_curr])
        if G_MSE_loss_curr_dev is not None:
            dev_MSE_loss.append([i, G_MSE_loss_curr_dev])

# Save the loss
np.savez(os.path.join(OUT_DIR, 'loss.npz'), train_MSE_loss=np.array(train_MSE_loss), dev_MSE_loss=np.array(dev_MSE_loss))
# Save the final blended output, and make a graph of the loss.
util.plot_loss(os.path.join(OUT_DIR, 'loss.npz'), 'MSE Loss During Training', os.path.join(OUT_DIR, 'loss_plot.png'))
util.postprocess_images_outpainting(os.path.join(OUT_DIR, 'test_img.png'), last_output_PATH, os.path.join(OUT_DIR, 'out_paste.png'), blend=False)
util.postprocess_images_outpainting(os.path.join(OUT_DIR, 'test_img.png'), last_output_PATH, os.path.join(OUT_DIR, 'out_blend.png'), blend=True)
