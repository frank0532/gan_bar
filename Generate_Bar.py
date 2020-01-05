#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import mpl_finance as mpf
import matplotlib.pyplot as plt


class Generate_Bar():

    def __init__(self, base_close=10.0):
        self.base_close = base_close
    
    def generate_bars(self, generator, num_samples, sequence_length, rand_dims=8):
        noise_seed = tf.random.normal([num_samples, 1, rand_dims])
        data = generator(noise_seed, False) + 1.0
        return data * self.base_close
#		data_shape = data.shape
#		base_multiplier = tf.reshape(tf.concat([tf.ones([data_shape[0], 1]), \
#		                      tf.math.cumprod(data[:, :-1, 3], axis=-1)], \
#		                      axis=-1), \
#		                (data_shape[0], data_shape[1], 1)) * self.base_close
#		return data * base_multiplier

    def plot1bar(self, data, ax):
        L = len(data)
        candle_data=np.column_stack([list(range(L)), data])
        mpf.candlestick_ohlc(ax, candle_data, width=0.5, colorup='r', colordown='g')
        xindex = range(L)
        if L<=5:
            xindex=[0, L-2]
        else:
            xindex=list(range(0, L, L//5))
        ax.set_xticks(xindex)
        ax.grid()

    def plot_all_bars(self, data, save_file):
        data_shape = data.shape
        _, ax = plt.subplots(data_shape[0], 1, figsize=(5, data_shape[0] * 4))
        if data_shape[0] > 1:
            for i in range(data_shape[0]):
                self.plot1bar(data[i], ax[i])
        else:
            self.plot1bar(data[0], ax)
        plt.savefig(save_file)
        plt.clf()

    def __call__(self, generator=None, num_samples=10, sequence_length=5 , save_file=''):
        self.plot_all_bars(self.generate_bars(generator, num_samples, sequence_length), save_file)