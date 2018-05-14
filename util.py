# import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.misc

IMAGE_SZ = 64 # A power of 2, please!

def load_images(dir): # Outputs [m, IMAGE_SZ, IMAGE_SZ, 3]
    # TODO: change so it iterates over all images
    im = Image.open('images/city_64.png').convert('RGB')
    width, height = im.size
    left = (width - IMAGE_SZ) / 2
    top = (height - IMAGE_SZ) / 2
    im = im.crop((left, top, left + IMAGE_SZ, top + IMAGE_SZ))
    pix = np.array(im)
    assert pix.shape == (IMAGE_SZ, IMAGE_SZ, 3)
    return pix[np.newaxis] / 255.0 # TODO: CHECK THIS

def preprocess_images(imgs): # Outputs [m, IMAGE_SZ, IMAGE_SZ, 4]
    # TODO: Randomize it, and decide how we're doing the mean
    imgs = np.array(imgs, copy=True) # Don't want to overwrite
    pix_avg = np.mean(imgs)
    img = imgs[0] # TODO: Vectorize for entire thingy eventually
    img[int(3 * IMAGE_SZ / 8):int(-3 * IMAGE_SZ / 8), int(3 * IMAGE_SZ / 8):int(-3 * IMAGE_SZ / 8), :] = pix_avg
    mask = np.zeros((IMAGE_SZ, IMAGE_SZ, 1))
    mask[int(3 * IMAGE_SZ / 8):int(-3 * IMAGE_SZ / 8), int(3 * IMAGE_SZ / 8):int(-3 * IMAGE_SZ / 8), :] = 1.0
    imgs_p = np.concatenate((img, mask), axis=2)
    return imgs_p[np.newaxis]

def generate_rand_img(batch_size, w, h, nc): # width, height, number of channels, TODO: make the mask actually a binary thing
    # return tf.random_uniform([batch_size, w, h, nc], minval=0, maxval=256, dtype=tf.float32)
    return np.zeros((batch_size, w, h, nc), dtype=np.float32)

def norm_image(img_r):
    min_val = np.min(img_r)
    max_val = np.max(img_r)
    # img_norm = (255.0 * (img_r - min_val) / (max_val - min_val)).astype(np.int8)
    img_norm = (img_r * 255.0).astype(np.int8)
    return img_norm

def vis_image(img_r, mode='RGB'): # img should have 3 channels. Values will be normalized and truncated to [0, 255]
    img_norm = norm_image(img_r)
    img = Image.fromarray(img_norm, mode)
    img.show()

def save_image(img_r, name, mode='RGB'):
    img_norm = norm_image(img_r)
    img = Image.fromarray(img_norm, mode)
    img.save(name, format='PNG')

def read_in_CIFAR(file_name, class_label=None): # Returns numpy array of size (m, 32, 32, 3)
    # TODO: See https://www.cs.toronto.edu/~kriz/cifar.html
    # TODO: Decide if we want to filter by class (in the labels list)
    import pickle
    with open(file_name, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        raw_images = data[b'data']
        classes = np.array(data[b'labels'])
        class_images = raw_images[classes == class_label]
        class_images_np = np.array(class_images)
        class_images_resized = np.reshape(class_images_np, (-1, 32, 32, 3))
    return class_images_resized

def upsample_CIFAR(batch): # Rescales (m, 32, 32, 3) -> (m, 64, 64, 3)
    return np.array([scipy.misc.imresize(img, (64, 64, 3), interp='cubic') for img in batch])

def sample_random_minibatch(data, m): # Returns numpy array of size (m, 64, 64, 3)
    indices = np.random.randint(0, data.shape[0], m)
    return data[indices]
