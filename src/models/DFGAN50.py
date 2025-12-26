import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Lambda,
    multiply,
    Activation
)
from tensorflow.keras.models import Model
from tensorflow.keras import Input

from models.common import fft2d, fftshift2d


def global_average_pooling2d(x):
    return tf.reduce_mean(x, axis=[1, 2], keepdims=False)


def FCALayer(input, channel, reduction=16, size_psc=64):
    absfft1 = Lambda(
        fft2d,
        output_shape=lambda s: s
    )(input)

    absfft1 = Lambda(
        fftshift2d,
        arguments={'size_psc': size_psc},
        output_shape=lambda s: s
    )(absfft1)

    W = Lambda(global_average_pooling2d)(absfft1)
    W = Dense(channel // reduction, activation='relu')(W)
    W = Dense(channel, activation='sigmoid')(W)

    return multiply([input, W])


def FCAB(input, channel, size_psc):
    conv = Conv2D(channel, kernel_size=3, padding='same')(input)
    conv = Activation(tf.nn.gelu)(conv)
    att = FCALayer(conv, channel=channel, size_psc=size_psc)
    return conv + att


def ResidualGroup(input, channel, size_psc):
    conv = input
    for _ in range(4):
        conv = FCAB(conv, channel, size_psc)
    return conv + input


def Generator(input_shape, scale=2):
    inputs = Input(shape=input_shape)
    conv = Conv2D(64, kernel_size=3, padding='same')(inputs)

    conv = ResidualGroup(conv, 64, size_psc=64)

    conv = Conv2D(64 * scale * scale, kernel_size=3, padding='same')(conv)

    upsampled = Lambda(
        lambda x: tf.nn.depth_to_space(x, block_size=scale),
        output_shape=lambda s: (
            s[0],
            s[1] * scale,
            s[2] * scale,
            s[3] // (scale * scale)
        )
    )(conv)

    outputs = Conv2D(1, kernel_size=3, padding='same')(upsampled)

    return Model(inputs, outputs)
