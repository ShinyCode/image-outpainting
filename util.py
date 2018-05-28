import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import cv2
import wget
import os
from urllib.error import HTTPError
from urllib.error import URLError

IMAGE_SZ = 128 # A power of 2, please!
CIFAR_SZ = 32

def load_test_image(): # Outputs [m, IMAGE_SZ, IMAGE_SZ, 3]
    # TODO: change so it iterates over all images
    im = Image.open('images/city_64.png').convert('RGB')
    width, height = im.size
    left = (width - IMAGE_SZ) / 2
    top = (height - IMAGE_SZ) / 2
    im = im.crop((left, top, left + IMAGE_SZ, top + IMAGE_SZ))
    pix = np.array(im)
    assert pix.shape == (IMAGE_SZ, IMAGE_SZ, 3)
    return pix[np.newaxis] / 255.0 # Need to normalize images to [0, 1]

def load_images(in_PATH):
    imgs = []
    for filename in os.listdir(in_PATH):
        full_filename = os.path.join(os.path.abspath(in_PATH), filename)
        img = Image.open(full_filename).convert('RGB')
        pix = np.array(img)
        pix_norm = pix / 255.0
        imgs.append(pix_norm)
    return np.array(imgs)

def preprocess_images_inpainting(imgs, crop=True): # Outputs [m, IMAGE_SZ, IMAGE_SZ, 4]
    m = imgs.shape[0]
    imgs = np.array(imgs, copy=True) # Don't want to overwrite
    pix_avg = np.mean(imgs, axis=(1, 2, 3))
    if crop:
        imgs[:, int(3 * IMAGE_SZ / 8):int(-3 * IMAGE_SZ / 8), int(3 * IMAGE_SZ / 8):int(-3 * IMAGE_SZ / 8), :] = pix_avg[:, np.newaxis, np.newaxis, np.newaxis]
    mask = np.zeros((m, IMAGE_SZ, IMAGE_SZ, 1))
    mask[:, int(3 * IMAGE_SZ / 8):int(-3 * IMAGE_SZ / 8), int(3 * IMAGE_SZ / 8):int(-3 * IMAGE_SZ / 8), :] = 1.0
    imgs_p = np.concatenate((imgs, mask), axis=3)
    return imgs_p

def preprocess_images_outpainting(imgs, crop=True): # Outputs [m, IMAGE_SZ, IMAGE_SZ, 4]
    m = imgs.shape[0]
    imgs = np.array(imgs, copy=True) # Don't want to overwrite
    pix_avg = np.mean(imgs, axis=(1, 2, 3))
    if crop:
        imgs[:, :, :int(2 * IMAGE_SZ / 8), :] = imgs[:, :, int(-2 * IMAGE_SZ / 8):, :] = pix_avg[:, np.newaxis, np.newaxis, np.newaxis]

    mask = np.zeros((m, IMAGE_SZ, IMAGE_SZ, 1))
    mask[:, :, :int(2 * IMAGE_SZ / 8), :] = mask[:, :, int(-2 * IMAGE_SZ / 8):, :] = 1.0
    mask[:, :, int(2 * IMAGE_SZ / 8), :] = mask[:, :, int(-2 * IMAGE_SZ / 8), :] = 1.0
    mask[:, :, int(2 * IMAGE_SZ / 8) + 1, :] = mask[:, :, int(-2 * IMAGE_SZ / 8) - 1, :] = 0.8
    mask[:, :, int(2 * IMAGE_SZ / 8) + 2, :] = mask[:, :, int(-2 * IMAGE_SZ / 8) - 2, :] = 0.5
    mask[:, :, int(2 * IMAGE_SZ / 8) + 3, :] = mask[:, :, int(-2 * IMAGE_SZ / 8) - 3, :] = 0.2
    imgs_p = np.concatenate((imgs, mask), axis=3)
    return imgs_p

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

def upsample_CIFAR(batch): # Rescales (m, CIFAR_SZ, CIFAR_SZ, 3) -> (m, IMAGE_SZ, IMAGE_SZ, 3)
    return np.array([scipy.misc.imresize(img, (IMAGE_SZ, IMAGE_SZ, 3), interp='cubic') for img in batch])

def read_in_CIFAR(file_name, class_label=None): # Returns numpy array of size (m, IMAGE_SZ, IMAGE_SZ, 3)
    # NOTE: See https://www.cs.toronto.edu/~kriz/cifar.html
    import pickle
    with open(file_name, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        raw_images = data[b'data']
        classes = np.array(data[b'labels'])
    if class_label is not None:
        raw_images = raw_images[classes == class_label]
    class_images_np = np.array(raw_images)
    class_images_resized = np.transpose(np.reshape(class_images_np, (-1, 3, CIFAR_SZ, CIFAR_SZ)), (0, 2, 3, 1))
    class_images_upsampled = upsample_CIFAR(class_images_resized) # Upsample when loading data to save time during training
    return class_images_upsampled / 255.0 # Need to normalize images to [0, 1]

def sample_random_minibatch(data, data_p, m): # Returns two numpy arrays of size (m, 64, 64, 3)
    indices = np.random.randint(0, data.shape[0], m)
    return data[indices], data_p[indices]

def plot_loss(loss_filename, title, out_filename):
    loss = np.load(loss_filename)
    assert 'train_MSE_loss' in loss and 'dev_MSE_loss' in loss
    train_MSE_loss = loss['train_MSE_loss']
    dev_MSE_loss = loss['dev_MSE_loss'] # TODO: Deal with dev_MSE_loss not changing during Phase 2
    label_train, = plt.plot(train_MSE_loss[:, 0], train_MSE_loss[:, 1], label='Training MSE loss')
    label_dev, = plt.plot(dev_MSE_loss[:, 0], dev_MSE_loss[:, 1], label='Dev MSE loss')
    plt.legend(handles=[label_train, label_dev])
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    plt.title(title)
    plt.savefig(out_filename)
    plt.clf()

def postprocess_images_outpainting(img_PATH, img_o_PATH, out_PATH, blend=False): # img, img_0 are (64, 64, 3), mask is (64, 64, 1)
    src = cv2.imread(img_PATH)[:, int(2 * IMAGE_SZ / 8):-int(2 * IMAGE_SZ / 8), :]
    dst = cv2.imread(img_o_PATH)
    if blend:
        mask = np.ones(src.shape, src.dtype) * 255
        center = (int(IMAGE_SZ / 2) - 1, int(IMAGE_SZ / 2) - 1)
        out = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
    else:
        out = dst.copy()
        out[:, int(2 * IMAGE_SZ / 8):-int(2 * IMAGE_SZ / 8), :] = src
    cv2.imwrite(out_PATH, out)

def download_images(url_list_PATH, out_PATH, prefix):
    with open(url_list_PATH, 'r') as fp:
        for i, line in enumerate(fp):
            url = line.strip()
            _, ext = os.path.splitext(url)
            dst = os.path.abspath(os.path.join(out_PATH, prefix + str(i) + ext))
            try:
                filename = wget.download(url, dst)
            except:
                continue
            print('Downloaded %s' % filename)

def delete_blank_images(url_PATH, ref_img_PATH):
    ref_img = Image.open(os.path.abspath(ref_img_PATH)).convert('RGB')
    ref_img_array = np.array(ref_img)
    for filename in os.listdir(url_PATH):
        try:
            img = Image.open(os.path.join(os.path.abspath(url_PATH), filename)).convert('RGB')
        except:
            print('Invalid file: Deleting %s' % filename)
            os.remove(os.path.join(os.path.abspath(url_PATH), filename))
            continue
        img_array = np.array(img)
        if img_array.shape == ref_img_array.shape and np.sum(img_array) == np.sum(ref_img_array):
            print('Deleting %s' % filename)
            os.remove(os.path.join(os.path.abspath(url_PATH), filename))

def resize_images(src_PATH, dst_PATH):
    for filename in os.listdir(src_PATH):
        print('Processing %s' % filename)
        full_filename = os.path.join(os.path.abspath(src_PATH), filename)
        img_raw = Image.open(full_filename).convert('RGB')
        w, h = img_raw.size
        if w <= h:
            dim = w
            y_start = int((h - dim) / 2)
            img_crop = img_raw.crop(box=(0, y_start, dim, y_start + dim))
        else: # w > h
            dim = h
            x_start = int((w - dim) / 2)
            img_crop = img_raw.crop(box=(x_start, 0, x_start + dim, dim))
        img_scale = img_crop.resize((IMAGE_SZ, IMAGE_SZ), Image.ANTIALIAS)
        full_outfilename = os.path.join(os.path.abspath(dst_PATH), filename)
        img_scale.save(full_outfilename, format='PNG')
