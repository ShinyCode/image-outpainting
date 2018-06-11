# proj:    image-outpainting
# file:    model.py
# authors: Mark Sabini, Gili Rusak
# desc:    Model for outpainting on 128x128 images with only
#          a global discriminator.
# -------------------------------------------------------------
import tensorflow as tf

print('Imported model (for Places365, 128x128 images)')

def generator(z):
    with tf.variable_scope('G', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(
            inputs=z,
            filters=64,
            kernel_size=[5, 5],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=128,
            kernel_size=[3, 3],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv5 = tf.layers.conv2d(
            inputs=conv4,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(4, 4),
            padding="same",
            activation=tf.nn.relu)

        conv5_p = tf.layers.conv2d(
            inputs=conv5,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            dilation_rate=(8, 8),
            padding="same",
            activation=tf.nn.relu)

        conv6 = tf.layers.conv2d(
            inputs=conv5_p,
            filters=256,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        deconv7 = tf.layers.conv2d_transpose(
            inputs=conv6,
            filters=128,
            kernel_size=[4, 4],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv8 = tf.layers.conv2d(
            inputs=deconv7,
            filters=64,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu)

        out = tf.layers.conv2d(
            inputs=conv8,
            filters=3,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            activation=tf.sigmoid)

    return out

def global_discriminator(x):
    with tf.variable_scope('DG', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=64,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=64,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=64,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv5 = tf.layers.conv2d(
            inputs=conv4,
            filters=64,
            kernel_size=[5, 5],
            strides=(2, 2),
            padding="same",
            activation=tf.nn.relu)

        conv5_flat = tf.layers.flatten(
            inputs=conv5)

        dense6 = tf.layers.dense(
            inputs=conv5_flat,
            units=512,
            activation=tf.nn.relu)

    return dense6

def concatenator(global_x):
    with tf.variable_scope('C', reuse=tf.AUTO_REUSE):
        dense1 = tf.layers.dense(
            inputs=global_x,
            units=1,
            activation=tf.sigmoid)

    return dense1
