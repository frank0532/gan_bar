#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tushare as ts
import pandas_datareader.data as web
import numpy as np
import tensorflow as tf
import os, pickle

class Sample():
    
    def __init__(self, code='JPM', base_dir=''):
        data_dir = base_dir + 'gan_bars/data/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        save_file = data_dir + code        
        try:
            with open(save_file, 'rb') as tmp:
                self.data = pickle.load(tmp)
        except:            
            if code[:6].isdigit() and len(code) == 9:
                ts.pro_api('your_tushare_token')
                self.data = ts.pro_bar(ts_code=code, adj='qfq', asset='E', start_date='20000101').\
                    set_index('trade_date')[['open', 'high', 'low', 'close']].loc[::-1]
            else:
                data = web.DataReader(code, 'yahoo', '20000101')[['Open', 'High', 'Low', 'Close', 'Adj Close']]
                ratio = (data['Adj Close'] / data['Close']).values.reshape(-1, 1)
                self.data = data.iloc[:, :-1] * ratio
            with open(save_file, 'wb') as tmp:
                pickle.dump(self.data, tmp)
        self.data_v = self.data.values
        self.L = len(self.data)
#        assert self.L > 500, 'too few bars of ' + code
        print('load {} bars of {} in all.'.format(self.L, code))
        
    def __call__(self,  num, length):
        samples = []
        for  ix in np.random.choice(self.L - length - 1, num, replace=False):
            samples.append(tf.slice(self.data_v, [ix, 0], [length + 1, -1]))
        samples = tf.stack(samples,axis=0)
        return samples[:, 1:, :] / tf.reshape(samples[:, 0, -1], [-1, 1, 1]) - 1.0
