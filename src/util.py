# proj:    image-outpainting
# file:    util.py
# authors: Mark Sabini, Gili Rusak
# desc:    Various utility functions for all sorts of things.
# -------------------------------------------------------------
import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import cv2
import os
import re
import imageio

IMAGE_SZ = 128 # Should be a power of 2

# Loads the city image.
# Returns: normalized numpy array of size (1, IMAGE_SZ, IMAGE_SZ, 3)
def load_city_image():
    im = Image.open('images/city_128.png').convert('RGB')
    width, height = im.size
    left = (width - IMAGE_SZ) / 2
    top = (height - IMAGE_SZ) / 2
    im = im.crop((left, top, left + IMAGE_SZ, top + IMAGE_SZ))
    pix = np.array(im)
    assert pix.shape == (IMAGE_SZ, IMAGE_SZ, 3)
    return pix[np.newaxis] / 255.0 # Need to normalize images to [0, 1]

# Loads multiple images from a directory.
# Returns: normalized numpy array of size (m, IMAGE_SZ, IMAGE_SZ, 3)
def load_images(in_PATH, verbose=False):
    imgs = []
    for filename in sorted(os.listdir(in_PATH)):
        if verbose:
            print('Processing %s' % filename)
        full_filename = os.path.join(os.path.abspath(in_PATH), filename)
        img = Image.open(full_filename).convert('RGB')
        pix = np.array(img)
        pix_norm = pix / 255.0
        imgs.append(pix_norm)
    return np.array(imgs)

# Reads in all the images in a directory and saves them to an .npy file.
def compile_images(in_PATH, out_PATH):
    imgs = load_images(in_PATH, verbose=True)
    np.save(out_PATH, imgs)

# Masks and preprocesses an (m, IMAGE_SZ, IMAGE_SZ, 3) batch of images for image outpainting.
# Returns: numpy array of size (m, IMAGE_SZ, IMAGE_SZ, 4)
def preprocess_images_outpainting(imgs, crop=True):
    m = imgs.shape[0]
    imgs = np.array(imgs, copy=True)
    pix_avg = np.mean(imgs, axis=(1, 2, 3))
    if crop:
        imgs[:, :, :int(2 * IMAGE_SZ / 8), :] = imgs[:, :, int(-2 * IMAGE_SZ / 8):, :] = pix_avg[:, np.newaxis, np.newaxis, np.newaxis]
    mask = np.zeros((m, IMAGE_SZ, IMAGE_SZ, 1))
    mask[:, :, :int(2 * IMAGE_SZ / 8), :] = mask[:, :, int(-2 * IMAGE_SZ / 8):, :] = 1.0
    imgs_p = np.concatenate((imgs, mask), axis=3)
    return imgs_p

# Expands and preprocesses a single (h, w, 3) image for image outpainting.
# Returns: numpy array of size (h, w + 2 * dw, 4)
def preprocess_images_gen(img):
    img = np.array(img, copy=True)
    pix_avg = np.mean(img)
    dw = int(2 * IMAGE_SZ / 8) # Amount that will be outpainted on each side
    img_expand = np.ones((img.shape[0], img.shape[1] + 2 * dw, img.shape[2])) * pix_avg
    img_expand[:, dw:-dw, :] = img
    mask = np.zeros((img_expand.shape[0], img_expand.shape[1], 1))
    mask[:, :int(2 * IMAGE_SZ / 8), :] = mask[:, int(-2 * IMAGE_SZ / 8):, :] = 1.0
    img_p = np.concatenate((img_expand, mask), axis=2)
    return img_p[np.newaxis]

# Renormalizes an image to [0, 255].
def norm_image(img_r):
    img_norm = (img_r * 255.0).astype(np.uint8)
    return img_norm

# Visualize an image.
def vis_image(img_r, mode='RGB'):
    img_norm = norm_image(img_r)
    img = Image.fromarray(img_norm, mode)
    img.show()

# Save an image as a .png file.
def save_image(img_r, name, mode='RGB'):
    img_norm = norm_image(img_r)
    img = Image.fromarray(img_norm, mode)
    img.save(name, format='PNG')

# Sample a random minibatch from data.
# Returns: Two numpy arrays, representing examples and their corresponding
#          preprocessed arrays.
def sample_random_minibatch(data, data_p, m):
    indices = np.random.randint(0, data.shape[0], m)
    return data[indices], data_p[indices]

# Plots the loss and saves the plot.
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

# Plots the loss and saves the plot, but fancier.
def plot_loss2(loss_filename, title, out_filename):
    loss = np.load(loss_filename)
    itrain_MSE_loss, train_MSE_loss = loss['itrain_MSE_loss'], loss['train_MSE_loss']
    idev_MSE_loss, dev_MSE_loss = loss['idev_MSE_loss'], loss['dev_MSE_loss']
    iG_loss, G_loss = loss['iG_loss'], loss['G_loss']
    iD_loss, D_loss = loss['iD_loss'], loss['D_loss']
    label_train, = plt.plot(itrain_MSE_loss, train_MSE_loss, label='Training MSE loss')
    label_dev, = plt.plot(idev_MSE_loss, dev_MSE_loss, label='Dev MSE loss')
    label_G, = plt.plot(iG_loss, G_loss, label='Generator loss')
    label_D, = plt.plot(iD_loss, D_loss, label='Discriminator loss')
    plt.legend(handles=[label_train, label_dev, label_G, label_D])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(out_filename)
    plt.clf()

# Use seamless cloning to improve the generator's output.
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

# Use seamless cloning to improve the generator's output.
def postprocess_images_gen(img, img_o, blend=False):
    src = img[:, :, ::-1].copy()
    dst = img_o[:, :, ::-1].copy()
    if blend:
        mask = np.ones(src.shape, src.dtype) * 255
        center = (int(dst.shape[1] / 2) - 1, int(dst.shape[0] / 2) - 1)
        out = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
    else:
        out = dst.copy()
        out[:, int(2 * IMAGE_SZ / 8):-int(2 * IMAGE_SZ / 8), :] = src
    return out[:, :, ::-1].copy()

# Crop and resize all the images in a directory.
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

# Parse the output of train.py to extract the various losses.
def parse_log(in_PATH, out_PATH):
    data = []
    curr_list = []
    with open(in_PATH, 'r') as fp:
        for i, line in enumerate(fp):
            if i == 0:
                continue
            line = line.strip()
            if line.startswith('----'):
                continue
            elif line.startswith('Model'):
                continue
            elif line.startswith('Iteration'):
                if len(curr_list):
                    data.append(curr_list)
                    curr_list = []
                curr_list.append(line)
            else:
                curr_list.append(line)
        if len(curr_list):
            data.append(curr_list)
    G_MSE_train, G_MSE_dev, G, C = None, None, None, None
    G_MSE_train_s, G_MSE_dev_s, G_s, C_s = [], [], [], []
    G_MSE_train_is, G_MSE_dev_is, G_is, C_is = [], [], [], []
    def extract_loss(str):
        return float(re.findall('= ([\d, .]+)', str)[0])
    for entry in data:
        i = int(re.findall('\[(\d+)/', entry[0])[0])
        if len(entry) == 3: # Phase 1
            G_MSE_train = extract_loss(entry[1])
            G_MSE_dev = extract_loss(entry[2])
        elif len(entry) == 2: # Phase 2
            C = extract_loss(entry[1])
        elif len(entry) == 5: # Phase 3
            G_MSE_train = extract_loss(entry[1])
            G_MSE_dev = extract_loss(entry[2])
            G = extract_loss(entry[3])
            C = extract_loss(entry[4])
        if G_MSE_train is not None:
            G_MSE_train_s.append(G_MSE_train)
            G_MSE_train_is.append(i)
        if G_MSE_dev is not None:
            G_MSE_dev_s.append(G_MSE_dev)
            G_MSE_dev_is.append(i)
        if G is not None:
            G_s.append(G)
            G_is.append(i)
        if C is not None:
            C_s.append(C)
            C_is.append(i)
    G_MSE_train_sm = np.array(G_MSE_train_s)
    G_MSE_dev_sm = np.array(G_MSE_dev_s)
    G_sm = np.array(G_s)
    C_sm = np.array(C_s)
    G_MSE_train_ism = np.array(G_MSE_train_is)
    G_MSE_dev_ism = np.array(G_MSE_dev_is)
    G_ism = np.array(G_is)
    C_ism = np.array(C_is)
    np.savez(out_PATH, train_MSE_loss=G_MSE_train_sm, dev_MSE_loss=G_MSE_dev_sm, G_loss=G_sm, D_loss=C_sm,
             itrain_MSE_loss=G_MSE_train_ism, idev_MSE_loss=G_MSE_dev_ism, iG_loss=G_ism, iD_loss=C_ism)

# Smoothes the MSE loss in the output loss file to make plotting easier.
def smooth_MSE_loss(loss_file, window_size, outfile):
    losses = np.load(loss_file)
    train = losses['train_MSE_loss']
    dev = losses['dev_MSE_loss']
    num_train = train.shape[0]
    new_train_list = []
    for i in range(0, num_train, window_size):
        window_avg = np.sum(train[i:i+window_size, 1]) / float(window_size)
        window_avg_val = np.sum(train[i:i+window_size, 0]) / float(window_size)
        new_train_list.append([window_avg_val, window_avg])
    np_train = np.array(new_train_list[:-2])
    np.savez(outfile, train_MSE_loss=np_train, dev_MSE_loss=dev)

# Create a GIF to enable visualization of generator outputs over the course of training.
def create_GIF(in_PATH, prefix, out_PATH):
    indices = range(0, 227401, 200)
    images = []
    for index in indices:
        full_filename = os.path.join(os.path.abspath(in_PATH), prefix + str(index) + '.png')
        try:
            images.append(imageio.imread(full_filename))
        except:
            continue
    images = images[:50] + images[50::10] + [images[-1]]
    imageio.mimwrite(out_PATH, images, loop=1, duration=0.1)

# Compute the RMSE between a ground truth and outpainted image.
def compute_RMSE(image_gt_PATH, image_o_PATH):
    im_gt = np.array(Image.open(image_gt_PATH).convert('RGB')).astype(np.float64)
    im_o = np.array(Image.open(image_o_PATH).convert('RGB')).astype(np.float64)
    assert im_gt.shape == (128, 128, 3)
    assert im_o.shape == (128, 128, 3)
    M = np.ones((128, 128, 3))
    M[:, 32:96, :] = 0
    num_pixels = 128 * 64 * 3
    return np.sqrt(np.sum(((im_gt - im_o) * M) ** 2) / num_pixels)
