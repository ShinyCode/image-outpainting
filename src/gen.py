# proj:    image-outpainting
# file:    gen.py
# authors: Mark Sabini, Gili Rusak
# desc:    Script for generating new images. Pads the image, creates
#          a mask, feeds it through the network, and postprocesses.
# -------------------------------------------------------------
import tensorflow as tf
import numpy as np
from PIL import Image
import model
import util
import os
import sys

if len(sys.argv) != 4:
    print('Usage: python gen.py [model_PATH] [in_PATH] [out_PATH]')
    exit()

_, model_PATH, in_PATH, out_PATH = sys.argv

tf.reset_default_graph()

IMAGE_SZ = 128

img = np.array(Image.open(in_PATH).convert('RGB'))
img_p = util.preprocess_images_gen(img / 255.0)

G_Z = tf.placeholder(tf.float32, shape=[1, img_p.shape[1], img_p.shape[2], 4], name='G_Z')
G_sample = model.generator(G_Z)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_PATH)
    output, = sess.run([G_sample], feed_dict={G_Z: img_p})
    output = util.norm_image(output[0])
    output_p = util.postprocess_images_gen(img, output, blend=True)
    img_o = Image.fromarray(output_p, 'RGB')
    img_o.save(out_PATH, format='PNG')
