from keras.layers import (
    Input, Activation, Dense,
    Reshape, Embedding,
    Permute, Add, Concatenate, Lambda,
    GlobalAveragePooling3D, GlobalMaxPooling3D, multiply,
    Conv2D,
)
import keras
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
# from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.models import Model
from keras.layers.convolutional import Conv3D
# from keras.engine.topology import Layer
from tensorflow.python.keras.layers import Layer
import numpy as np
from keras import backend as K
import tensorflow as tf

K.set_image_data_format('channels_first')


class iLayer(Layer):
    '''
    final weighted sum
    '''

    def __init__(self, **kwargs):
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.train_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape


class Recalibration(Layer):
    '''
    channel-wise recalibration for closeness component
    '''

    def __init__(self, **kwargs):
        super(Recalibration, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        input_shape: (batch, c, h, w)
        '''
        initial_weight_value = np.random.random((input_shape[1], 2, input_shape[2], input_shape[3]))  # (c,2,h,w)
        self.W = K.variable(initial_weight_value)
        self.train_weights = [self.W]

        super(Recalibration, self).build(input_shape)

    def call(self, x):
        '''
        x: (batch, c, h,w)
        '''
        double_x = tf.stack([x, x], axis=2)  # [(batch,c,h,w), (batch, c,h,w)] => (batch,c,2,h,w)
        return tf.reduce_sum(double_x * self.W, 1)  # (batch,2,h,w)

    def compute_output_shape(self, input_shape):
        return input_shape[0], 2, input_shape[2], input_shape[3]  # (batch_size,2,h,w)


class Recalibration_T(Layer):
    '''
    channel-wise recalibration for weekly period component:
    '''

    def __init__(self, channel, **kwargs):
        super(Recalibration_T, self).__init__(**kwargs)
        self.channel = channel

    def build(self, input_shape):
        '''
        input_shape: (batch, c, h, w)
        '''
        initial_weight_value = np.random.random(input_shape[1] * 2)  # [2c,]:because output 2 channel
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

        super(Recalibration_T, self).build(input_shape)

    def call(self, x):
        '''
        x: (batch, c, h, w)
        '''
        nb_channel = self.channel
        _, _, map_height, map_width = x.shape
        W = tf.reshape(tf.tile(self.W, [map_height * map_width]), (nb_channel, 2, map_height,
                                                                   map_width))  # sharing channel-wsie weight on different positions in the weekly-period recalibration block
        double_x = tf.stack([x, x], axis=2)  # stack [(batch, c,h, w)] = (batch, c, 2, h,w)
        return tf.reduce_sum(double_x * W, 1)  # (batch, 2, h, w)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2, input_shape[2], input_shape[3])  # (batch_size,2,h,w)


def _shortcut(input, residual):
    return keras.layers.Add()([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        '''
        input: (batch,c,h,w)
        '''
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=subsample, padding="same")(
            activation)

    return f


def _residual_unit(nb_filter):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)

    return f


def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            input = residual_unit(nb_filter=nb_filter)(input)
        return input

    return f


def ftd1(x):
    return x[:, 0:16, :, :, :]


def ftd2(x):
    return x[:, 16:32, :, :, :]




def ChannelAttention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling3D()(input_feature)
    avg_pool = Reshape((1, 1, 1, channel))(avg_pool)
    # assert avg_pool.shape[1:] == (1, 1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    # assert avg_pool.shape[1:] == (1, 1, 1,channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    # assert avg_pool.shape[1:] == (1, 1, 1, channel)

    max_pool = GlobalMaxPooling3D()(input_feature)
    max_pool = Reshape((1, 1, 1, channel))(max_pool)
    # assert max_pool.shape[1:] == (1, 1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    # assert max_pool.shape[1:] == (1, 1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    # assert max_pool.shape[1:] == (1, 1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((4, 1, 2, 3))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def SpatialAttention(input_feature):
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 4, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    # assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    # assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    # assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=32,
                          kernel_size=kernel_size,
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    # assert cbam_feature.shape[-1] == 32

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((4, 1, 2, 3))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in CBAM: Convolutional Block Attention Module.
    """
    cbam_feature = ChannelAttention(cbam_feature, ratio)
    # cbam_feature = SpatialAttention(cbam_feature, )
    return cbam_feature


def ST3DNet(c_conf=(6, 2, 16, 8), p_conf=(6, 2, 16, 8), t_conf=(4, 2, 16, 8), external_dim=8, nb_residual_unit=4):
    # ST3DNet(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf, external_dim=external_dim, nb_residual_unit=nb_residual_unit)
    len_closeness, nb_flow, map_height, map_width = c_conf

    main_inputs = []
    outputs = []
    if len_closeness > 0:
        input = Input(shape=(nb_flow, len_closeness, map_height, map_width))  # (2,t_c,h,w)
        main_inputs.append(input)
        # Conv1 3D
        conv = Conv3D(filters=32, kernel_size=(4, 3, 3), strides=(1, 1, 1), padding="same",
                      kernel_initializer='random_uniform')(input)
        conv1 = Activation("relu")(conv)

        cbam = cbam_block(conv1)
        conv1 = keras.layers.Add()([conv1, cbam])

        conv_1 = keras.layers.Lambda(ftd1)(conv1)

        conv_2 = keras.layers.Lambda(ftd2)(conv1)

        #
        conv21 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 1, 1),
                        kernel_initializer='random_uniform')(conv_1)
        conv21 = Activation("relu")(conv21)
        conv21 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv21)

        conv21 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 2, 2),
                        kernel_initializer='random_uniform')(conv21)
        conv21 = Activation("relu")(conv21)
        conv21 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv21)

        conv21 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 5, 5),
                        kernel_initializer='random_uniform')(conv21)
        conv21 = Activation("relu")(conv21)
        conv21 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv21)

        conv22 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 1, 1),
                        kernel_initializer='random_uniform')(conv_2)
        conv22 = Activation("relu")(conv22)
        conv22 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv22)

        conv22 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 2, 2),
                        kernel_initializer='random_uniform')(conv22)
        conv22 = Activation("relu")(conv22)
        conv22 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv22)

        conv22 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 5, 5),
                        kernel_initializer='random_uniform')(conv22)
        conv22 = Activation("relu")(conv22)
        conv22 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv22)

        conv = Concatenate(axis=1)([conv21, conv22])
        conv = keras.layers.Add()([conv1, conv])

        conv4 = Conv3D(filters=2, kernel_size=(4, 3, 3), strides=(2, 1, 1), padding="same",
                       kernel_initializer='random_uniform')(conv)
        conv = keras.layers.Dropout(0.25)(conv4)

        outputs.append(conv)

    if p_conf is not None:
        len_period, nb_flow, map_height, map_width = p_conf
        input = Input(shape=(nb_flow, len_period, map_height, map_width))
        main_inputs.append(input)
        conv = Conv3D(filters=32, kernel_size=(len_period, 3, 3), strides=(1, 1, 1), padding="same",
                      kernel_initializer='random_uniform')(input)
        conv1 = Activation("relu")(conv)

        cbam = cbam_block(conv1)
        conv1 = keras.layers.Add()([conv1, cbam])

        # conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same",
        #                kernel_initializer='random_uniform')(conv1)

        conv_1 = keras.layers.Lambda(ftd1)(conv1)

        conv_2 = keras.layers.Lambda(ftd2)(conv1)



        #
        conv21 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 1, 1),
                        kernel_initializer='random_uniform')(conv_1)
        conv21 = Activation("relu")(conv21)
        conv21 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv21)

        conv21 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 3, 3),
                        kernel_initializer='random_uniform')(conv21)
        conv21 = Activation("relu")(conv21)
        conv21 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv21)

        conv21 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 5, 5),
                        kernel_initializer='random_uniform')(conv21)
        conv21 = Activation("relu")(conv21)
        conv21 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv21)

        conv22 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 1, 1),
                        kernel_initializer='random_uniform')(conv_2)
        conv22 = Activation("relu")(conv22)
        conv22 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv22)

        conv22 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 3, 3),
                        kernel_initializer='random_uniform')(conv22)
        conv22 = Activation("relu")(conv22)
        conv22 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv22)

        conv22 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 5, 5),
                        kernel_initializer='random_uniform')(conv22)
        conv22 = Activation("relu")(conv22)
        conv22 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv22)
        conv = Concatenate(axis=1)([conv21, conv22])
        conv = keras.layers.Add()([conv1, conv])

        conv4 = Conv3D(filters=2, kernel_size=(1, 3, 3), strides=(2, 1, 1), padding="same",
                       kernel_initializer='random_uniform')(conv)
        conv = keras.layers.Dropout(0.25)(conv4)

        outputs.append(conv)

    if t_conf is not None:
        len_trend, nb_flow, map_height, map_width = t_conf
        input = Input(shape=(nb_flow, len_trend, map_height, map_width))
        main_inputs.append(input)
        conv = Conv3D(filters=32, kernel_size=(len_trend, 3, 3), strides=(1, 1, 1), padding="same",
                      kernel_initializer='random_uniform')(input)
        conv1 = Activation("relu")(conv)

        cbam = cbam_block(conv1)
        conv1 = keras.layers.Add()([conv1, cbam])

        conv_1 = keras.layers.Lambda(ftd1)(conv1)

        conv_2 = keras.layers.Lambda(ftd2)(conv1)

        conv21 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 1, 1),
                        kernel_initializer='random_uniform')(conv_1)
        conv21 = Activation("relu")(conv21)
        conv21 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv21)

        conv21 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 3, 3),
                        kernel_initializer='random_uniform')(conv21)
        conv21 = Activation("relu")(conv21)
        conv21 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv21)

        conv21 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 5, 5),
                        kernel_initializer='random_uniform')(conv21)
        conv21 = Activation("relu")(conv21)
        conv21 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv21)

        conv22 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 1, 1),
                        kernel_initializer='random_uniform')(conv_2)
        conv22 = Activation("relu")(conv22)
        conv22 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv22)

        conv22 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 3, 3),
                        kernel_initializer='random_uniform')(conv22)
        conv22 = Activation("relu")(conv22)
        conv22 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv22)

        conv22 = Conv3D(filters=8, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                        dilation_rate=(1, 5, 5),
                        kernel_initializer='random_uniform')(conv22)
        conv22 = Activation("relu")(conv22)
        conv22 = Conv3D(filters=16, kernel_size=(2, 1, 1), strides=(1, 1, 1), padding="same",
                        kernel_initializer='random_uniform')(conv22)
        conv = Concatenate(axis=1)([conv21, conv22])
        conv = keras.layers.Add()([conv1, conv])



        conv4 = Conv3D(filters=2, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding="same",
                       kernel_initializer='random_uniform')(conv)
        conv = keras.layers.Dropout(0.25)(conv4)

        outputs.append(conv)



    if len(outputs) == 1:
        main_output = outputs[0]
        main_output = keras.layers.Flatten(data_format='channels_first')(main_output)
        main_output = Dense(units=2 * map_height * map_width)(main_output)
        main_output = Activation('relu')(main_output)
    else:
        # from .iLayer import iLayer
        new_outputs = []
        # for output in outputs:
        #     new_outputs.append(iLayer()(output))
        #####genghai
        main_output = keras.layers.Add()(outputs)
        # main_output = Dense(output_dim=2 * map_height * map_width)(main_output)
        main_output = keras.layers.Flatten(data_format='channels_first')(main_output)
        # main_output = Dense(output_dim=2 * map_height * map_width)(main_output)
        main_output = Dense(units=2 * map_height * map_width)(main_output)
        main_output = Activation('relu')(main_output)

    # main_output = keras.layers.Permute((2, 1, 3, 4))(main_output)
    #
    # main_output = keras.layers.ConvLSTM2D(filters=2, kernel_size=(3, 3), padding="same",
    #                                data_format='channels_first', return_sequences=False,
    #                                recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0),
    #                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0),
    #                                activity_regularizer=tf.keras.regularizers.l1_l2(l1=0, l2=0)
    #                                )(main_output)

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        # embedding = Dense(output_dim=10)(external_input)
        embedding = Dense(units=10)(external_input)
        embedding = Activation('relu')(embedding)
        # h1 = Dense(output_dim=2 * map_height * map_width)(embedding)
        h1 = Dense(units=2 * map_height * map_width)(embedding)
        external_output = Activation('relu')(h1)
        # external_output = Reshape((2, map_height, map_width))(external_output)
        # main_output = keras.layers.Add()([main_output, external_output])
    else:
        print('external_dim:', external_dim)

    # embedding = keras.layers.Flatten(data_format='channels_first')(main_output)
    # embedding = Activation('relu')(embedding)
    # embedding = Dense(output_dim=16 * map_height * map_width)(embedding)
    # embedding = Activation('relu')(embedding)
    # embedding = Dense(output_dim=4 * map_height * map_width)(embedding)
    # embedding = Activation('relu')(embedding)
    #
    # embedding = Dense(output_dim=2048)(embedding)
    # embedding = Activation('relu')(embedding)
    # main_output = Dense(units=2048)(main_output)
    main_output = keras.layers.Add()([main_output, external_output])
    # main_output = Dense(output_dim=2 * map_height * map_width)(main_output)
    conv = Reshape((nb_flow, map_height, map_width))(main_output)
    # if mask.shape[0] != 0:
    #   conv = Lambda(lambda el: el * mask)(conv)
    conv = Activation('relu')(conv)
    # model = Model(input=main_inputs, output=conv)
    model = Model(inputs=main_inputs, outputs=conv)

    return model

# def ST3DNetE(c_conf=(6, 2, 16, 8), t_conf=(4, 2, 16, 8), external_dim=8, nb_residual_unit=4):
#     len_closeness, nb_flow, map_height, map_width = c_conf
#     nb = map_height * map_width
#     # main input
#     main_inputs = []
#     outputs = []
#     if len_closeness > 0:
#         input = Input(shape=(nb_flow, len_closeness, map_height, map_width))  # (2,t_c,h,w)
#         main_inputs.append(input)
#         # embedding
#         input = Input(shape=(nb))  # (2,t_c,h,w)
#         emb_input = np.arange(nb).reshape(-1, nb)  # (1, nb=hw)
#         emb_input = Embedding(nb, 64, input_length=nb)(emb_input)  # (1, nb=hw, 64)
#         emb_input = emb_input.reshape(1, map_height, map_width, 64)  # (1, h, w, 64)
#         emb_input = Permute((2, 3), input_shape=(map_height, map_width, 64))(emb_input)  # (1, h, 64, w)
#         emb_input = Permute((1, 2), input_shape=(map_height, map_width, 64))(emb_input)  # (1, 64, h, w)
#
#         # Conv1 3D
#         conv = Conv3D(filters=64, kernel_size=(6, 3, 3), strides=(1, 1, 1), border_mode="same",
#                       kernel_initializer='random_uniform')(input)
#
#         # add embed and con1 3D
#         conv = emb_input + conv
#
#         conv = Activation("relu")(conv)
#
#         # Conv2 3D
#         conv = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(conv)
#         conv = Activation("relu")(conv)
#
#         # Conv3 3D
#         conv = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(3, 1, 1), border_mode="same")(conv)
#
#         # (filter,1,height,width)
#         reshape = Reshape((64, map_height, map_width))(conv)
#
#         # Residual 2D [nb_residual_unit] Residual Units
#         residual_output = ResUnits(_residual_unit, nb_filter=64, repetations=nb_residual_unit)(reshape)
#
#         output_c = Recalibration()(residual_output)
#         outputs.append(output_c)
#
#     if t_conf is not None:
#         len_seq, nb_flow, map_height, map_width = t_conf
#         input = Input(shape=(nb_flow, len_seq, map_height, map_width))
#         main_inputs.append(input)
#
#         conv = Conv3D(nb_filter=8, kernel_dim1=len_seq, kernel_dim2=1, kernel_dim3=1, border_mode="valid")(input)
#         conv = Activation('relu')(conv)
#
#         output_t = Reshape((8, map_height, map_width))(conv)
#         output_t = Recalibration_T(8)(output_t)
#
#         outputs.append(output_t)
#
#     # parameter-matrix-based fusion
#     if len(outputs) == 1:
#         main_output = outputs[0]
#     else:
#         # from .iLayer import iLayer
#         new_outputs = []
#         for output in outputs:
#             new_outputs.append(iLayer()(output))
#         main_output = keras.layers.Add()(new_outputs)
#
#     # fusing with external component
#     if external_dim != None and external_dim > 0:
#         # external input
#         external_input = Input(shape=(external_dim,))
#         main_inputs.append(external_input)
#         embedding = Dense(output_dim=10)(external_input)
#         embedding = Activation('relu')(embedding)
#         h1 = Dense(output_dim=nb_flow * map_height * map_width)(embedding)
#         activation = Activation('relu')(h1)
#         external_output = Reshape((nb_flow, map_height, map_width))(activation)
#         main_output = keras.layers.Add()([main_output, external_output])
#     else:
#         print('external_dim:', external_dim)
#
#     main_output = Activation('relu')(main_output)
#     model = Model(input=main_inputs, output=main_output)
#
#     return model
