#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dense, Conv2D, LSTM, Attention
import tensorflow as tf

class Generator_CNN(Model):
    
    def __init__(self, rows=14, columns=8, channels=16):
        super(Generator_CNN, self).__init__()
        self.rows, self.columns, self.channels = rows, columns, channels
        self.dense = Dense(units=self.rows*self.columns*self.channels, use_bias=False)
        self.batchnorm_1 = BatchNormalization()
        self.batchnorm_2 = BatchNormalization()
        self.batchnorm_3 = BatchNormalization()
        self.lrl = LeakyReLU(0.2)
        self.conv_1 = Conv2D(filters=8, kernel_size=3, strides=(1, 1), \
                                          padding='same', use_bias=False)
        self.conv_2 = Conv2D(filters=1, kernel_size=3, strides=(1,1), \
                                          padding='same', use_bias=False)
    def trim(self, bar_data):#ohlc
        return tf.clip_by_value(bar_data, \
                      tf.expand_dims(bar_data[:, :, 2, :], axis=-1), \
                      tf.expand_dims(bar_data[:, :, 1, :], axis=-1))
    
    def call(self, noise_seed, training):
        raw_matrix = tf.reshape(self.lrl(self.batchnorm_1(self.dense(noise_seed), training)), \
                                [-1, self.rows, self.columns, self.channels])
        conv_1 = self.lrl(self.batchnorm_2(self.conv_1(raw_matrix), training))
        return tf.reshape(self.lrl(self.batchnorm_3(self.conv_2(conv_1), training)), (-1, self.rows, 4))


class Generator_LSTM(Model):     
    
    def __init__(self, output_dims=4, rows=14):
        super(Generator_LSTM, self).__init__()
        self.output_dims = output_dims
        self.rows = rows
        self.lstm_forward_dims = self.output_dims * 2
        self.lstm_backward_dims = self.output_dims
        self.batchnorm_h = BatchNormalization()
        self.batchnorm_c = BatchNormalization()
        self.batchnorm_3 = BatchNormalization()
        self.lstm_forward_init = LSTM(self.lstm_forward_dims, return_state=True)
        self.lstm_forward = LSTM(self.lstm_forward_dims, return_state=True, activation='relu')
        self.lstm_backward = LSTM(self.lstm_backward_dims, return_sequences=True, activation='relu')
        self.dense_1 = Dense(units=self.output_dims * 2)
        self.attention = Attention()
        self.dense_2 = Dense(units=self.output_dims, activation='tanh')
    def trim(self, bar_data):#ohlc
        return tf.clip_by_value(bar_data, \
                                tf.expand_dims(bar_data[:, :, 2], axis=-1), \
                                tf.expand_dims(bar_data[:, :, 1], axis=-1))
        
    def call(self, noise_seed, training):
        bars = []
        _, h, c = self.lstm_forward_init(noise_seed)
        h, c = self.batchnorm_h(h, training), self.batchnorm_c(c, training)
        for _ in range(self.rows):
            _, h, c = self.lstm_forward(tf.reshape(h, [-1, 1, self.lstm_forward_dims]), initial_state=[h, c])
            bars.append(h)
        return self.lstm_backward(self.batchnorm_3(tf.stack(bars[::-1], axis=1), training))
