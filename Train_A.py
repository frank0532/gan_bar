#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Generate_Bar import Generate_Bar
from Gan_Bar import Gan_Bar
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
import mpl_finance as mpf
import os, pickle


class Train_A():
    
    def __init__(self, base_dir = '/content/drive/My Drive/', base_close=10.0):
        self.base_dir = base_dir
        self.tasks_dir = self.base_dir + 'gan_bars/tasks/'
        self.result_dir = self.base_dir + 'gan_bars/results/'
        if os.path.exists(self.base_dir + 'pc.num'):
            with open(self.base_dir + 'pc.num', 'rb') as tmp:
                self.pc = pickle.load(tmp)
        else:
            self.pc = input('Please put the unique number for this computer:')
            with open(self.base_dir + 'pc.num', 'wb') as tmp:
                pickle.dump(self.pc, tmp)
        if not os.path.exists(self.tasks_dir):
            os.makedirs(self.tasks_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.gen_bars = Generate_Bar(base_close=base_close)
    
    def __call__(self, mode='master', num_pc = 7):
        if mode.upper() == 'MASTER':
            left_codes = []
            if os.path.exists(self.tasks_dir + self.pc):
                for fi in os.listdir(self.tasks_dir):
                    with open(self.tasks_dir + fi, 'rb') as tmp:
                        left_codes.extend(pickle.load(tmp))
            if not left_codes:
                pro = ts.pro_api('your_tushare_token')
                left_codes = pro.stock_basic(exchange='', list_status='L', fields='ts_code').values.reshape(-1)
            codes_split = np.array_split(left_codes, num_pc)            
            pcs = range(-1, num_pc-1)
            for pci in range(num_pc):
                with open(self.tasks_dir + str(pcs[pci]), 'wb') as tmp:
                    pickle.dump(codes_split[pci].tolist(), tmp)
        elif mode.upper() == 'SLAVE':
            while 1:
                with open(self.tasks_dir + self.pc, 'rb') as tmp:
                    codes = pickle.load(tmp)
                if len(codes) > 0:
                    code = codes[0]
                else:
                    print('task for '+ self.pc + ' is completed.')
                    break
                suffix = code[:6] if code[:6].isdigit() else code
                
                gan_bar = Gan_Bar(code=code, base_dir=self.base_dir)
#                try:
#                    gan_bar = Gan_Bar(code=code, base_dir=self.base_dir)
#                except:
#                    time.sleep(60)
#                    gan_bar = Gan_Bar(code=code, base_dir=self.base_dir)
#                i += 1
#                print(i, end=':')
#                with open(self.tasks_dir + self.pc, 'wb') as tmp:
#                    pickle.dump(codes[1:], tmp)
#                continue

                gen_loss, dis_loss = gan_bar.train(gen_name='CNN', epochs=800, num_samples=20, sequence_length_list=[5, 10, 15])
                if gen_loss and dis_loss:
                    self.save_gen_dis_fig(gen_loss, dis_loss, self.result_dir + suffix + '_gen_dis_loss.png')
                    gan_bar.save_weights(self.result_dir + suffix + '.weights')
                    self.gen_bars(generator=gan_bar.generators[-1], num_samples=10, sequence_length=5, save_file=self.result_dir + suffix + '_generate_fig.png')
                    self.save_bars_indicator(gan_bar, 100, 10, code, self.result_dir + suffix + '_indicator.png')
                with open(self.tasks_dir + self.pc, 'wb') as tmp:
                    pickle.dump(codes[1:], tmp)
        else:
            print("'Mode' is error!")
            
    def save_gen_dis_fig(self, gen_loss, dis_loss, save_file):
        plt.figure(figsize=(10, 3))
        plt.subplot(121)
        plt.plot(gen_loss)
        plt.xlabel('gen')
        plt.subplot(122)
        plt.plot(dis_loss)
        plt.xlabel('dis')
        plt.savefig(save_file)
        plt.clf()
    
    def save_bars_indicator(self, gan_bar, num_bars, num4indicator, title, fig_name):
        raw_bars = gan_bar.sampler.data.iloc[1-num_bars-num4indicator:, :]
        predict_data = []
        for i in range(num4indicator):
            predict_data.append(raw_bars[i:i+num_bars])
        predict_data = np.stack(predict_data, axis=1)
        indicator = gan_bar.discriminator(predict_data, False)
        fig = plt.figure(figsize=(num_bars * 0.8, 8))
        ax1 = fig.add_axes([0.1, 0.3, 0.8, 0.6])
        Data = raw_bars.iloc[-num_bars:, :]
        dates = Data.index
        L=len(dates)
        candleData=np.column_stack([list(range(L)),Data])
        mpf.candlestick_ohlc(ax1, candleData, width=0.5, colorup='r', colordown='g')
        if L<=5:
            xindex=[0, L-2]
        else:
            xindex=list(range(0, L, L//5))
        ax1.set_xticks(xindex)
        ax1.grid() 
        plt.title(title, color='r')
        ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.2])
        ax2.plot(range(L), indicator)
        ax2.set_xticks(xindex)
        ax2.set_xticklabels(dates[xindex], rotation=20)
        ax2.grid()
        plt.savefig(fig_name)
        plt.clf()

T = Train_A(base_dir = '/home/caofa/Documents/codes/git/')
T('slave', None)
