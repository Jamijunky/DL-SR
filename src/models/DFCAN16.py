from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input, add, multiply, Lambda
from tensorflow.keras.layers import Conv2D
from .common import fft2d, fftshift2d, gelu, pixel_shiffle, global_average_pooling2d


def FCALayer(input, channel, reduction=16, size_psc=128):
    absfft1 = Lambda(
    fft2d,
    arguments={'gamma': 0.8},
    output_shape=lambda s: s
    )(input)
    absfft1 = Lambda(
    fftshift2d,
    arguments={'size_psc': size_psc},
    output_shape=lambda s: s
    )(absfft1)
    absfft2 = Conv2D(channel, kernel_size=3, activation='relu', padding='same')(absfft1)
    W = Lambda(global_average_pooling2d)(absfft2)
    W = Conv2D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
    W = Conv2D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
    mul = multiply([input, W])
    return mul


def FCAB(input, channel, size_psc=128):
    conv = Conv2D(channel, kernel_size=3, padding='same')(input)
    
    conv = Activation(tf.nn.gelu)(conv)
    conv = Conv2D(channel, kernel_size=3, padding='same')(conv)
    
    conv = Activation(tf.nn.gelu)(conv)
    att = FCALayer(conv, channel, reduction=16, size_psc=size_psc)
    output = add([att, input])
    return output


def ResidualGroup(input, channel, size_psc=128):
    conv = input
    n_RCAB = 4
    for _ in range(n_RCAB):
        conv = FCAB(conv, channel=channel, size_psc=size_psc)
    conv = add([conv, input])
    return conv


def DFCAN(input_shape, scale=2, size_psc=128):
    inputs = Input(input_shape)
    conv = Conv2D(64, kernel_size=3, padding='same')(inputs)
    n_ResGroup = 4
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, size_psc)
    conv = Conv2D(64 * (scale ** 2), kernel_size=3, padding='same')(conv)
    conv = Activation(tf.nn.gelu)(conv)
    upsampled = Lambda(
    lambda x: tf.nn.depth_to_space(x, block_size=scale),
    output_shape=lambda s: (s[0], s[1]*scale, s[2]*scale, s[3]//(scale**2))
)(conv)

    conv = Conv2D(1, kernel_size=3, padding='same')(upsampled)
    output = Activation('sigmoid')(conv)
    model = Model(inputs=inputs, outputs=output)
    return model
