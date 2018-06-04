import numpy as np
import matplotlib.pyplot as plt
from squeezenet import SqueezeNet
import tensorflow as tf
import os

# tf.reset_default_graph() # remove all existing variables in the graph

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
def preprocess_image(img):
    return (img.astype(np.float32) - SQUEEZENET_MEAN) / SQUEEZENET_STD

def content_loss(content_weight, content_current, content_original):
    loss = content_weight * tf.reduce_sum(tf.square(content_original - content_current))
    return loss

def gram_matrix(features, normalize=True):
    shape = tf.shape(features)
    _, H, W, C = shape[0], shape[1], shape[2], shape[3]
    features_altered = tf.reshape(features, [H * W, C])
    gram = tf.matmul(tf.transpose(features_altered), features_altered)
    if normalize:
        return gram / tf.cast(H * W * C, tf.float32)
    return gram

def style_loss(feats, style_layers, style_targets, style_weights):
    style_loss = 0
    for i in range(len(style_layers)):
        gram = gram_matrix(feats[style_layers[i]])
        style_loss += content_loss(style_weights[i], gram, style_targets[i])
    return style_loss

def tv_loss(img, tv_weight):
    shape = tf.shape(img)
    _, H, W, C = shape[0], shape[1], shape[2], shape[3]
    img_down = img[:, :H-1, 1:W, :]
    img_up = img[:, 1:, 1:W, :]
    img_left = img[:, 1:H, :W-1, :]
    img_right = img[:, 1:H, 1:, :]
    loss = tv_weight * tf.reduce_sum(tf.square(img_down - img_up) + tf.square(img_left - img_right))
    return loss

CONTENT_LAYER = 3
CONTENT_WEIGHT = 5e-2
STYLE_LAYERS = (1, 4, 6, 7)
STYLE_WEIGHTS = (20000, 500, 12, 1)
TV_WEIGHT = 5e-2

def all_style_loss(sq, im_out, im_gt, sess):
    # Extract features of ground truth image for content loss
    gt_img = preprocess_image(im_gt)
    feats = sq.extract_features(model.image)
    content_target = sess.run(feats[CONTENT_LAYER], {model.image: gt_img[None]})

    # Extract features of ground truth image for style loss
    style_feat_vars = [feats[idx] for idx in STYLE_LAYERS]
    style_target_vars = []
    for style_feat_var in style_feat_vars:
        style_target_vars.append(gram_matrix(style_feat_var))
    style_targets = sess.run(style_target_vars, {model.image: gt_img[None]})

    # Extract features on generated image
    out_img = preprocess_image(im_out)
    feats = sq.extract_features(out_img)
    # Compute loss
    c_loss = content_loss(CONTENT_WEIGHT, feats[content_layer], content_target)
    s_loss = style_loss(feats, STYLE_LAYERS, style_targets, STYLE_WEIGHTS)
    t_loss = tv_loss(out_img, TV_WEIGHT)
    return c_loss + s_loss + t_loss
