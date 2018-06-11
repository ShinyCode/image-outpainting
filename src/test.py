# proj:    image-outpainting
# file:    test.py
# authors: Mark Sabini, Gili Rusak
# desc:    Script for simulating the training pipeline. Masks out
#          the sides of an image, feeds it through the network, and
#          compares the network output to the original image.
# -------------------------------------------------------------
import tensorflow as tf
import numpy as np
from PIL import Image
import model
import util
import os
import sys

if len(sys.argv) != 4:
    print('Usage: python test.py [model_PATH] [in_PATH] [out_PATH]')
    exit()

_, model_PATH, in_PATH, out_PATH = sys.argv

tf.reset_default_graph()

IMAGE_SZ = 128

img = np.array(Image.open(in_PATH).convert('RGB'))[np.newaxis] / 255.0
img_p = util.preprocess_images_outpainting(img)

G_Z = tf.placeholder(tf.float32, shape=[None, IMAGE_SZ, IMAGE_SZ, 4], name='G_Z')
G_sample = model.generator(G_Z)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_PATH)
    output, = sess.run([G_sample], feed_dict={G_Z: img_p})
    util.save_image(output[0], out_PATH)
