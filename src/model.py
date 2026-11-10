import tensorflow as tf
from tensorflow.keras import layers, Model

def residual_mixer_block(x, channels, kernel_size=5):
    residual = x


    x = layers.DepthwiseConv2D(kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Add()([x, residual])

    # Pointwise (1x1) convolution
    residual = x
    x = layers.Conv2D(channels, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Add()([x, residual])
    return x


def build_model(channels=128, depth=6, kernel_size=5, num_classes=10):
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(channels, 3, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for _ in range(depth):
        x = residual_mixer_block(x, channels, kernel_size)
        
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="CompactMixer")
    return model