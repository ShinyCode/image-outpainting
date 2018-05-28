import sys
import util
import os

if len(sys.argv) < 4:
    print('Usage: python download.py [synset-PATH] [output-PATH] [refimg-PATH]')
    exit()

synset_PATH = sys.argv[1]
output_PATH = sys.argv[2]
ref_img_PATH = sys.argv[3]
tmp_PATH = output_PATH + '_t'

if not os.path.exists(tmp_PATH):
    os.makedirs(tmp_PATH)

if not os.path.exists(output_PATH):
    os.makedirs(output_PATH)

util.download_images(synset_PATH, tmp_PATH, 'img')
util.delete_blank_images(tmp_PATH, ref_img_PATH)
util.resize_images(tmp_PATH, output_PATH)
