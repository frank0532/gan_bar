#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Conv1D, LSTM, Attention

class Discriminator(Model):
    
    def __init__(self, standard_dims=16):
        super(Discriminator, self).__init__()
        self.batchnorm_1 = BatchNormalization()
        self.batchnorm_2 = BatchNormalization()
        self.batchnorm_3 = BatchNormalization()
        
        self.lstm_1 = LSTM(standard_dims, return_sequences=True)
        self.cnn_q = Conv1D(filters=standard_dims, kernel_size=1)
        self.cnn_v = Conv1D(filters=standard_dims, kernel_size=1)
        self.attention = Attention()
        self.lstm_2 = LSTM(standard_dims)
        self.dense_1 = Dense(units=standard_dims)
        self.dense_2 = Dense(units=standard_dims // 2)
        self.dense_3 = Dense(units=1, activation='sigmoid')

    def call(self, sequence_data, training):
        lstm1 = self.batchnorm_1(self.lstm_1(sequence_data), training)
        atten = self.batchnorm_2(self.attention([self.cnn_q(lstm1), self.cnn_v(lstm1)]), training)
        lstm2 = self.batchnorm_3(self.lstm_2(lstm1 + atten), training)
        return self.dense_3(self.dense_2(self.dense_1(lstm2)))

