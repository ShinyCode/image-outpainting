import tensorflow as tf
import numpy as np
from PIL import Image

print('Imported vanilla model.')

def generator(z):
    with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(
            inputs=z,
            filters=64,
            kernel_size=[5, 5],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv2_1 = tf.layers.conv2d(
            inputs=conv1,
            filters=128,
            kernel_size=[3, 3],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv2_2 = tf.layers.conv2d(
            inputs=conv2_1,
            filters=128,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv3_1 = tf.layers.conv2d(
            inputs=conv2_2,
            filters=256,
            kernel_size=[3, 3],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv3_2 = tf.layers.conv2d(
            inputs=conv3_1,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv3_3 = tf.layers.conv2d(
            inputs=conv3_2,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv3_4 = tf.layers.conv2d(
            inputs=conv3_3,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv3_5 = tf.layers.conv2d(
            inputs=conv3_4,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(4, 4),
            padding="same",
            activation=tf.nn.relu)

        conv3_6 = tf.layers.conv2d(
            inputs=conv3_5,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(8, 8),
            padding="same",
            activation=tf.nn.relu)

        conv3_7 = tf.layers.conv2d(
            inputs=conv3_6,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(16, 16),
            padding="same",
            activation=tf.nn.relu)

        conv3_8 = tf.layers.conv2d(
            inputs=conv3_7,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv3_9 = tf.layers.conv2d(
            inputs=conv3_8,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        deconv4_1 = tf.layers.conv2d_transpose(
            inputs=conv3_9,
            filters=128,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv4_2 = tf.layers.conv2d(
            inputs=deconv4_1,
            filters=128,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        deconv5_1 = tf.layers.conv2d_transpose(
            inputs=conv4_2,
            filters=64,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv5_2 = tf.layers.conv2d(
            inputs=deconv5_1,
            filters=32,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv5_3 = tf.layers.conv2d(
            inputs=conv5_2,
            filters=3,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.sigmoid)

    return conv5_3

# Local Discriminator code
def global_discriminator(x):
    with tf.variable_scope('DG', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=128,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=256,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=512,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv5 = tf.layers.conv2d(
            inputs=conv4,
            filters=512,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv6 = tf.layers.conv2d(
            inputs=conv5,
            filters=512,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv6_flat = tf.layers.flatten(
            inputs=conv6)

        dense7 = tf.layers.dense(
            inputs=conv6_flat,
            units=1024,
            activation=tf.nn.relu)

    return dense7

def concatenator(global_x):
    with tf.variable_scope('C', reuse=tf.AUTO_REUSE):
        dense1 = tf.layers.dense(
            inputs=global_x,
            units=1,
            activation=tf.sigmoid)

    return dense1
