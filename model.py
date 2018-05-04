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

theta_G = [G_Wconv1_1, G_bconv1_1, G_Wconv2_1, G_bconv2_1, G_Wconv2_2, G_bconv2_2, G_Wconv3_1, G_bconv3_1,
           G_Wconv3_2, G_bconv3_2, G_Wconv3_3, G_bconv3_3, G_Wdlconv3_4, G_bdlconv3_4, G_Wdlconv3_5, G_bdlconv3_5,
           G_Wdlconv3_6, G_bdlconv3_6, G_Wdlconv3_7, G_bdlconv3_7, G_Wconv3_8, G_bconv3_8, G_Wconv3_9, G_bconv3_9,
           G_Wdeconv4_1, G_bdeconv4_1, G_Wconv4_2, G_bconv4_2, G_Wdeconv5_1, G_bdeconv5_1, G_Wconv5_2, G_bconv5_2,
           G_Wconv5_3, G_bconv5_3]
assert len(theta_G) == 34

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

    G_a3_4 = tf.nn.conv2d(G_h3_3, G_Wdlconv3_4, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 1, 2, 2]) + G_bdlconv3_4
    G_h3_4 = tf.nn.relu(G_a3_4) # (None, 128, 128, 256)

    G_a3_5 = tf.nn.conv2d(G_h3_4, G_Wdlconv3_5, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 1, 4, 4]) + G_bdlconv3_5
    G_h3_5 = tf.nn.relu(G_a3_5) # (None, 128, 128, 256)

    G_a3_6 = tf.nn.conv2d(G_h3_5, G_Wdlconv3_6, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 1, 8, 8]) + G_bdlconv3_6
    G_h3_6 = tf.nn.relu(G_a3_6) # (None, 128, 128, 256)

    G_a3_7 = tf.nn.conv2d(G_h3_6, G_Wdlconv3_7, strides=[1, 1, 1, 1], padding='SAME', dilations=[1, 1, 16, 16]) + G_bdlconv3_7
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
DG_X = tf.placeholder(tf.float32, shape=[BATCH_SZ, 256, 256, 3], name='DG_X')
DG_Wconv1 = tf.get_variable('DG_Wconv1', shape=[5, 5, 3, 64])
DG_bconv1 = tf.get_variable('DG_bconv1', shape=[64])

DG_Wconv2 = tf.get_variable('DG_Wconv2', shape=[5, 5, 64, 128])
DG_bconv2 = tf.get_variable('DG_bconv2', shape=[128])

DG_Wconv3 = tf.get_variable('DG_Wconv3', shape=[5, 5, 128, 256])
DG_bconv3 = tf.get_variable('DG_bconv3', shape=[256])

DG_Wconv4 = tf.get_variable('DG_Wconv4', shape=[5, 5, 256, 512])
DG_bconv4 = tf.get_variable('DG_bconv4', shape=[512])

DG_Wconv5 = tf.get_variable('DG_Wconv5', shape=[5, 5, 512, 512])
DG_bconv5 = tf.get_variable('DG_bconv5', shape=[512])

DG_Wconv6 = tf.get_variable('DG_Wconv6', shape=[5, 5, 512, 512])
DG_bconv6 = tf.get_variable('DG_bconv6', shape=[512])

DG_Wdense7 = tf.get_variable('DG_Wdense7', shape=[8192, 1024])
DG_bdense7 = tf.get_variable('DG_bdense7', shape=[1024])

theta_DG = [DG_Wconv1, DG_bconv1, DG_Wconv2, DG_bconv2, DG_Wconv3, DG_bconv3, DG_Wconv4, DG_bconv4,
            DG_Wconv5, DG_bconv5, DG_Wconv6, DG_bconv6, DG_Wdense7, DG_bdense7]

def global_discriminator(x): # Takes BATCH_SIZE x 128 x 128 x 3, outputs BATCH_SIZE x 1024
    DG_a1 = tf.nn.conv2d(x, DG_Wconv1, strides=[1, 2, 2, 1], padding='SAME') + DG_bconv1
    DG_h1 = tf.nn.relu(DG_a1) # (None, 128, 128, 64)

    DG_a2 = tf.nn.conv2d(DG_h1, DG_Wconv2, strides=[1, 2, 2, 1], padding='SAME') + DG_bconv2
    DG_h2 = tf.nn.relu(DG_a2) # (None, 64, 64, 128)

    DG_a3 = tf.nn.conv2d(DG_h2, DG_Wconv3, strides=[1, 2, 2, 1], padding='SAME') + DG_bconv3
    DG_h3 = tf.nn.relu(DG_a3) # (None, 32, 32, 256)

    DG_a4 = tf.nn.conv2d(DG_h3, DG_Wconv4, strides=[1, 2, 2, 1], padding='SAME') + DG_bconv4
    DG_h4 = tf.nn.relu(DG_a4) # (None, 16, 16, 512)

    DG_a5 = tf.nn.conv2d(DG_h4, DG_Wconv5, strides=[1, 2, 2, 1], padding='SAME') + DG_bconv5
    DG_h5 = tf.nn.relu(DG_a5) # (None, 8, 8, 512)

    DG_a6 = tf.nn.conv2d(DG_h5, DG_Wconv6, strides=[1, 2, 2, 1], padding='SAME') + DG_bconv6
    DG_h6 = tf.nn.relu(DG_a6) # (None, 4, 4, 512)

    DG_h6_flat = tf.reshape(DG_h6, [BATCH_SZ, 8192])

    DG_a7 = tf.matmul(DG_h6_flat, DG_Wdense7) + DG_bdense7
    DG_h7 = tf.nn.relu(DG_a7) # (None, 1024)
    return DG_h7

C_Wdense1 = tf.get_variable('C_Wdense1', shape=[1024, 1])
C_bdense1 = tf.get_variable('C_bdense1', shape=[1])

theta_C = [C_Wdense1, C_bdense1]

def concatenator(global_x):
    C_a1 = tf.matmul(global_x, C_Wdense1) + C_bdense1 # logits
    C_h1 = tf.sigmoid(C_a1) # (None, 1)
    return C_h1, C_a1

imgs, imgs_r = load_images(None)
imgs_p = preprocess_images(imgs)

with tf.Session() as sess:
    # TODO: Sample batches from training set
    G_sample = generator(G_Z)
    C_real, _ = concatenator(global_discriminator(DG_X))
    G_sample_r = tf.image.resize_images(G_sample, [256, 256]) # TODO: Check this isn't sketchy
    C_fake, _ = concatenator(global_discriminator(G_sample_r))

    C_loss = -tf.reduce_mean(tf.log(C_real) + tf.log(1. - C_fake))
    G_loss = -tf.reduce_mean(tf.log(C_fake))

    C_solver = tf.train.AdamOptimizer().minimize(C_loss, var_list=(theta_DG + theta_C))
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    sess.run(tf.global_variables_initializer())

    _, C_loss_curr = sess.run([C_solver, C_loss], feed_dict={DG_X: imgs_r, G_Z: imgs_p})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={G_Z: imgs_p})
