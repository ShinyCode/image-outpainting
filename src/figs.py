# proj:    image-outpainting
# file:    figs.py
# authors: Mark Sabini, Gili Rusak
# desc:    Collection of utilities for generating figures.
# -------------------------------------------------------------
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import util

IMAGE_SZ = 128

def resize(in_PATH, out_PATH):
    img = Image.open(in_PATH).convert('RGB')
    img_scale = img.resize((IMAGE_SZ, IMAGE_SZ), Image.ANTIALIAS)
    img_scale.save(out_PATH, format='PNG')

def mask(in_PATH, out_PATH):
    img = np.array(Image.open(in_PATH).convert('RGB'))
    pix_avg = np.mean(img)
    img[:, :int(2 * IMAGE_SZ / 8), :] = img[:, int(-2 * IMAGE_SZ / 8):, :] = pix_avg
    img = Image.fromarray(img.astype(np.uint8), 'RGB')
    img.save(out_PATH, format='PNG')
