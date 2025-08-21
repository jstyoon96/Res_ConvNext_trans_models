import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,BatchNormalization, Dropout, Add
from scipy.signal import butter, filtfilt
######################
#preprocessing
######################
def extract_features_with_meta_and_filter(signal):
    fs = 500
    r = 0
    while r < signal.shape[0] and np.all(signal[r, :] == 0):
        r += 1
    s = signal.shape[0]
    while s > r and np.all(signal[s-1, :] == 0):
        s -= 1
    signal = signal[r:s, :] if r < s else np.zeros((5000, 12), dtype=np.float32)

    try:
        nyquist = 0.5 * fs
        normal_cutoff = 0.5 / nyquist
        b, a = butter(1, normal_cutoff, btype='high', analog=False)
        signal = filtfilt(b, a, signal, axis=0)
    except:
        signal -= np.mean(signal, axis=0, keepdims=True)

    signal -= np.mean(signal, axis=0, keepdims=True)
    max_val = np.max(np.abs(signal), axis=0, keepdims=True)
    signal /= (max_val + 1e-8)

    if signal.shape[0] < 5000:
        pad_width = 5000 - signal.shape[0]
        signal = np.pad(signal, ((0, pad_width), (0, 0)), mode='constant')
    elif signal.shape[0] > 5000:
        signal = signal[:5000, :]


    return signal.astype(np.float32)
####################################################################################
#resnet50
####################################################################################
def resnet_CNN_1(x,k,Ch):
    y = Conv2D(Ch, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    y = MaxPooling2D(pool_size=(1, 2), padding='same')(y)
    shortcut = y

    for i in range(3):
        if (i == 0):
            x = BatchNormalization(axis=3)(x)
            x = Activation('relu')(x)
            x = Conv2D(Ch, kernel_size=(1, k), strides=(1, 1), padding='same')(x)

            x = BatchNormalization(axis=3)(x)
            x = Activation('relu')(x)
            x = Conv2D(Ch, kernel_size=(1, k), strides=(1, 2), padding='same')(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            x = Dropout(0.1)(x)

        elif (i == 1):
            y = Conv2D(Ch, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
            y = MaxPooling2D(pool_size=(1, 2), padding='same')(y)
            shortcut = y

            x = BatchNormalization(axis=3)(x)
            x = Activation('relu')(x)
            x = Conv2D(Ch, kernel_size=(1, k), strides=(1, 1), padding='same')(x)

            x = BatchNormalization(axis=3)(x)
            x = Activation('relu')(x)
            x = Conv2D(Ch, kernel_size=(1, k), strides=(1, 2), padding='same')(x)

            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            x = Dropout(0.1)(x)


        else:
            x = Dropout(0.1)(x)

    return x

def resnet_CNN_2(x):
    x = Conv2D(256, kernel_size=(1, 8), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    return x


def resnet_CNN_3(x):
    x = Conv2D(16, kernel_size=(1, 8), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    return x

####################################################################################
#ConvNext
####################################################################################
class StochasticDepth(layers.Layer):
    """Stochastic Depth module.

    It is also referred to as Drop Path in `timm`.
    References:
        (1) github.com:rwightman/pytorch-image-models
    """

    def __init__(self, drop_path, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class Block(tf.keras.Model):
    """ConvNeXt block.

    References:
        (1) https://arxiv.org/abs/2201.03545
        (2) https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.dim = dim
        if layer_scale_init_value > 0:
            self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim,)))
        else:
            self.gamma = None
        self.dw_conv_1 = layers.Conv2D(
            filters=dim, kernel_size=(1,7), padding="same", groups=dim
        )
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.pw_conv_1 = layers.Dense(4 * dim)
        self.act_fn = layers.Activation("gelu")
        self.pw_conv_2 = layers.Dense(dim)
        self.drop_path = (
            StochasticDepth(drop_path)
            if drop_path > 0.0
            else layers.Activation("linear")
        )

    def call(self, inputs):
        x = inputs

        x = self.dw_conv_1(x)
        x = self.layer_norm(x)
        x = self.pw_conv_1(x)
        x = self.act_fn(x)
        x = self.pw_conv_2(x)

        if self.gamma is not None:
            x = self.gamma * x

        return inputs + self.drop_path(x)


def get_convnext_model(
    model_name="convnext_tiny_1k",
    input_shape=(1, 5000, 12),
    num_classes=1,
    depths=[3, 3, 9, 3],
    # dims=[64, 128, 256, 512, 1024], #dims=[96, 192, 384, 768],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.5,
    layer_scale_init_value=1e-6,
) -> keras.Model:
    """Implements ConvNeXt family of models given a configuration.

    References:
        (1) https://arxiv.org/abs/2201.03545
        (2) https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Note: `predict()` fails on CPUs because of group convolutions. The fix is recent at
    the time of the development: https://github.com/keras-team/keras/pull/15868. It's
    recommended to use a GPU / TPU.
    """

    inputs = layers.Input(input_shape)
    stem = keras.Sequential(
        [
            layers.Conv2D(dims[0], kernel_size=(1,4), strides=(1,4)),
            layers.LayerNormalization(epsilon=1e-6),
        ],
        name="stem",
    )

    downsample_layers = []
    downsample_layers.append(stem)
    for i in range(3):
        downsample_layer = keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6),
                layers.Conv2D(dims[i + 1], kernel_size=(1,2), strides=(1,2)),
            ],
            name=f"downsampling_block_{i}",
        )
        downsample_layers.append(downsample_layer)

    stages = []
    dp_rates = [x for x in tf.linspace(0.0, drop_path_rate, sum(depths))]
    cur = 0
    for i in range(4):
        stage = keras.Sequential(
            [
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        name=f"convnext_block_{i}_{j}",
                    )
                    for j in range(depths[i])
                ]
            ],
            name=f"convnext_stage_{i}",
        )
        stages.append(stage)
        cur += depths[i]

    x = inputs
    for i in range(len(stages)):
        x = downsample_layers[i](x)
        x = stages[i](x)

    x = layers.GlobalAvgPool2D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    outputs = layers.Dense(num_classes, name="classification_head", activation='sigmoid')(x)

    return keras.Model(inputs, outputs, name=model_name)

####################################################################################
#CNN transformer encoder
####################################################################################
################################################################################
# Focal Loss with Sample Weights (Precompiled)
################################################################################
def get_weighted_focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred, sample_weight):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal = alpha * tf.pow((1 - p_t), gamma) * bce
        return tf.reduce_mean(focal * sample_weight)
    return loss

weighted_focal_loss_fn = get_weighted_focal_loss(alpha=0.5, gamma=2.0)

################################################################################
# Positional Encoding
################################################################################
class PositionalEncoding(Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = None

    def build(self, input_shape):
        _, seq_len, d_model = input_shape
        self.d_model = d_model

        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates

        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        if d_model % 2 == 0:
            pos_encoding = tf.concat([sines, cosines], axis=-1)
        else:
            pos_encoding = tf.concat([sines, cosines], axis=-1)
            pos_encoding = pos_encoding[:, :d_model]

        self.pos_encoding = tf.expand_dims(pos_encoding, axis=0)

    def call(self, x):
        return x + self.pos_encoding

################################################################################
# Transformer Block
################################################################################
def transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

################################################################################
# Model Architecture
################################################################################
def build_model(input_shape, num_classes=1, num_transformer_blocks=2):
    ecg_input = keras.Input(shape=input_shape, name='ecg_input')
    meta_input = keras.Input(shape=(2,), name='meta_input')

    x = layers.Conv1D(32, 5, padding="same", activation="relu")(ecg_input)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)

    x = PositionalEncoding()(x)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.concatenate([x, meta_input])
    x = layers.LayerNormalization()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation="sigmoid")(x)

    return keras.Model(inputs=[ecg_input, meta_input], outputs=output)

