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
