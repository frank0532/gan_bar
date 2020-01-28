# Generative Adversarial Networks on Stock Bars
## 1.General
Use GAN technology on stock bars (Chinese A stocks, stocks of USA and HK) to train two models: generator and discriminator.
### 1.1 Discriminator 
which can be used as an indicator on a stock, similar with KDJ.
![image](https://raw.githubusercontent.com/frank0532/gan_bar/master/results/000002_indicator.png)
![image](https://raw.githubusercontent.com/frank0532/gan_bar/master/results/600600_indicator.png)
![image](https://raw.githubusercontent.com/frank0532/gan_bar/master/results/300246_indicator.png)
### 1.2 Generator 
which can be used to select stocks or group them.

## 2.File
### 2.1 Discriminator.py
build discriminator model;
### 2.2 Generator.py
build generator model;
### 2.3 Generate_Bar.py
generate faked bars data from random seed by Generator and plot bars figure;
### 2.4 Sample.py
select sequential real bars data from real stocks trading data which are saved in ‘data’ Dir or would be downloaded by ‘tushare’ / ‘pandas_datareader’ if not in this Dir;
### 2.5 Gan_Bar.py
train GAN model on a certain stock data;
### 2.6 Train_A.py 
train GAN model on all Chinese A stocks one by one(all needed data had been saved in ‘data’ Dir); It runs on Colab to train GAN on 7 accounts simultaneously and it has two mode: Master ans slave; ‘Master’ distributes task for each account(recorded in ‘task’ Dir); ‘Slave’ train GAN on own distributed stocks one by one. 

## 3.Dir
### 3.1 Data
save stocks data which are downloaded by ‘tushare’ / ‘pandas_datareader’;
### 3.2 Results
each model weights, train loss, generate figs and discriminator indicator after complete GAN model training;
#### 3.2.1 Train Loss 
![image](https://raw.githubusercontent.com/frank0532/gan_bar/master/results/600600_gen_dis_loss.png)
#### 3.2.2 Generate Fig
![image](https://raw.githubusercontent.com/frank0532/gan_bar/master/results/600600_generate_fig.png)
#### 3.2.3 Discriminator Indicator
![image](https://raw.githubusercontent.com/frank0532/gan_bar/master/results/600600_indicator.png)
### 3.3 Task
each task which is distributed by Train_A.py when it run on mode ‘Master’ and modified on mode ‘Slave’ on Colab for multi-training.



