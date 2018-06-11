# proj:    image-outpainting
# file:    train_ld.py
# authors: Mark Sabini, Gili Rusak
# desc:    Train the model specified in model_ld.py, which
#          uses both global and local discriminators.
# -------------------------------------------------------------
import tensorflow as tf
import numpy as np
from PIL import Image
import model_ld as model
import util
import os
import sys

tf.reset_default_graph()

# Places365 Training Hyperparameters
BATCH_SZ = 16
VERBOSE = False
EPSILON = 1e-9
IMAGE_SZ = 128
OUT_DIR = 'output'
MODEL_DIR = os.path.join(OUT_DIR, 'models')
INFO_PATH = os.path.join(OUT_DIR, 'run.txt')
N_TEST = 10
N_ITERS = 64000
N_ITERS_P1 = 20000 # How many iterations to train in phase 1
N_ITERS_P2 = 4000 # How many iterations to train in phase 2
INTV_PRINT = 200 # How often to print
INTV_SAVE = 1000 # How often to save the model
ALPHA = 0.0004

'''
# City Training Hyperparameters
BATCH_SZ = 1
VERBOSE = False
EPSILON = 1e-9
IMAGE_SZ = 128
OUT_DIR = 'output'
MODEL_DIR = os.path.join(OUT_DIR, 'models')
INFO_PATH = os.path.join(OUT_DIR, 'run.txt')
N_TEST = 1
N_ITERS = 5000
N_ITERS_P1 = 1000 # How many iterations to train in phase 1
N_ITERS_P2 = 400 # How many iterations to train in phase 2
INTV_PRINT = 50 # How often to print
INTV_SAVE = 10000 # How often to save the model
ALPHA = 0.0004
'''

# Check that we don't clobber a pre-existing run
if len(sys.argv) < 2 and os.path.isdir(OUT_DIR) and len(os.listdir(OUT_DIR)) > 2:
    print('Warning, OUT_DIR already exists. Aborting.')
    exit()

# Load in a model if specified as the second argument.
start_iter = 0
model_filename = None
if len(sys.argv) >= 2:
    start_iter = int(sys.argv[1])
    model_filename = os.path.join(MODEL_DIR, 'model%d.ckpt' % start_iter)

# Generator code
G_Z = tf.placeholder(tf.float32, shape=[None, IMAGE_SZ, IMAGE_SZ, 4], name='G_Z')
DG_X = tf.placeholder(tf.float32, shape=[None, IMAGE_SZ, IMAGE_SZ, 3], name='DG_X')

# Load Places365 data
data = np.load('places/places_128.npz')
imgs = data['imgs_train'] # Originally from http://data.csail.mit.edu/places/places365/val_256.tar
imgs_p = util.preprocess_images_outpainting(imgs)

test_imgs = data['imgs_test']
test_imgs_p = util.preprocess_images_outpainting(test_imgs)

test_img = test_imgs[:N_TEST]
test_img_p = test_imgs_p[:N_TEST]

train_img = imgs[4, np.newaxis]
train_img_p = imgs_p[4, np.newaxis]

'''
# Load city image data
imgs = util.load_city_image()
imgs_p = util.preprocess_images_outpainting(imgs)

test_imgs = util.load_city_image()
test_imgs_p = util.preprocess_images_outpainting(test_imgs)

test_img = test_imgs
test_img_p = test_imgs_p

train_img = imgs
train_img_p = imgs_p
'''

# Write training and testing sample ground truths as reference
util.save_image(train_img[0], os.path.join(OUT_DIR, 'train_img.png'))
for i_test in range(N_TEST):
    util.save_image(test_imgs[i_test], os.path.join(OUT_DIR, 'test_img_%d.png' % i_test))

G_sample = model.generator(G_Z)
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

C_real = model.concatenator(model.global_discriminator(DG_X), model.local_discriminator(DG_X[:, :, :IMAGE_SZ // 2, :]), model.local_discriminator(tf.reverse(DG_X[:, :, -IMAGE_SZ // 2:, :], axis=[2])))
C_fake = model.concatenator(model.global_discriminator(G_sample), model.local_discriminator(G_sample[:, :, :IMAGE_SZ // 2, :]), model.local_discriminator(tf.reverse(G_sample[:, :, -IMAGE_SZ // 2:, :], axis=[2])))
vars_DG = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DG')
vars_DL = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DL')
vars_C = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='C')

C_loss = -tf.reduce_mean(tf.log(tf.maximum(C_real, EPSILON)) + tf.log(tf.maximum(1. - C_fake, EPSILON)))
G_MSE_loss = tf.losses.mean_squared_error(G_sample, DG_X, weights=tf.expand_dims(G_Z[:,:,:,3], -1)) # TODO: MULTIPLY with mask. Actually see if we want to remove this.
G_loss = G_MSE_loss - ALPHA * tf.reduce_mean(tf.log(tf.maximum(C_fake, EPSILON)))

C_solver = tf.train.AdamOptimizer().minimize(C_loss, var_list=(vars_DG + vars_DL + vars_C))
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=vars_G)
G_MSE_solver = tf.train.AdamOptimizer().minimize(G_MSE_loss, var_list=vars_G)

train_MSE_loss = []
dev_MSE_loss = []

last_output_PATH = [None] * N_TEST

assert N_ITERS > N_ITERS_P1 + N_ITERS_P2

# Saver to save the session
saver = tf.train.Saver()

with tf.Session() as sess:
    if model_filename is None:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, model_filename)
    for i in range(start_iter, N_ITERS + 1):
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

        # Periodically test the generator on held-out images
        if i % INTV_PRINT == 0:
            G_MSE_loss_curr_dev = None
            if G_sample_ is not None:
                # Print out the dev image
                output, G_MSE_loss_curr_dev = sess.run([G_sample, G_MSE_loss], feed_dict={DG_X: test_img, G_Z: test_img_p})
                for i_test in range(N_TEST):
                    util.save_image(output[i_test], os.path.join(OUT_DIR, 'dev_%d_%d.png' % (i_test, i)))
                    last_output_PATH[i_test] = os.path.join(OUT_DIR, 'dev_%d_%d.png' % (i_test, i))
                # Also save the train image
                output, = sess.run([G_sample], feed_dict={DG_X: train_img, G_Z: train_img_p})
                util.save_image(output[0], os.path.join(OUT_DIR, 'train%d.png' % i))
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

        # Save the model every so often
        if i % INTV_SAVE == 0 and i > 0:
            save_path = saver.save(sess, os.path.join(MODEL_DIR, 'model%d.ckpt' % i))
            print('Model saved in path: %s' % save_path)

        # Save the loss every so often
        if i % INTV_SAVE == 0:
            np.savez(os.path.join(OUT_DIR, 'loss.npz'), train_MSE_loss=np.array(train_MSE_loss), dev_MSE_loss=np.array(dev_MSE_loss))

# Save the loss
np.savez(os.path.join(OUT_DIR, 'loss.npz'), train_MSE_loss=np.array(train_MSE_loss), dev_MSE_loss=np.array(dev_MSE_loss))
# Save the final blended output, and make a graph of the loss.
util.plot_loss(os.path.join(OUT_DIR, 'loss.npz'), 'MSE Loss During Training', os.path.join(OUT_DIR, 'loss_plot.png'))
for i_test in range(N_TEST):
    util.postprocess_images_outpainting(os.path.join(OUT_DIR, 'test_img_%d.png' % i_test), last_output_PATH[i_test], os.path.join(OUT_DIR, 'out_paste_%d.png' % i_test), blend=False)
    util.postprocess_images_outpainting(os.path.join(OUT_DIR, 'test_img_%d.png' % i_test), last_output_PATH[i_test], os.path.join(OUT_DIR, 'out_blend_%d.png' % i_test), blend=True)
