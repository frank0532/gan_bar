#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Sample import Sample
from Generator import Generator_CNN, Generator_LSTM
from Discriminator import Discriminator
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tqdm import tqdm


class Gan_Bar(Model):
    
    def __init__(self, code='C', base_dir='', rand_dims=8):
        super(Gan_Bar, self).__init__()
        self.rand_dims = rand_dims
        self.sampler = Sample(code, base_dir)
        self.cross_entropy = BinaryCrossentropy(from_logits=True)
        self.gen_optimizer = SGD(lr=1e-4, momentum=0.0, decay=0.0, nesterov=False)
        self.dis_optimizer = SGD(lr=5e-5, momentum=0.0, decay=0.0, nesterov=False)

#    @tf.function
    def train_dis(self, sli):
        noise_seed = tf.random.normal([self.num_samples, 1, self.rand_dims])
        real_samples =  self.sampler(self.num_samples, self.sequence_length_list[sli])
        with tf.GradientTape() as g:
            faked_samples =  self.generators[sli](noise_seed, True)
            faked_output = self.discriminator(faked_samples, True)
            real_output = self.discriminator(real_samples, True)
            dis_loss = self.cross_entropy(tf.ones_like(real_output), real_output) + \
                        self.cross_entropy(tf.zeros_like(faked_output), faked_output)
        dis_gradients = g.gradient(dis_loss, self.discriminator.trainable_variables)
        self.dis_optimizer.apply_gradients(zip(dis_gradients, self.discriminator.trainable_variables))
        return dis_loss

#    @tf.function
    def loss_dis(self, sli):
        noise_seed = tf.random.normal([self.num_samples, 1, self.rand_dims])
        real_samples =  self.sampler(self.num_samples, self.sequence_length_list[sli])
        faked_samples =  self.generators[sli](noise_seed, True)
        faked_output = self.discriminator(faked_samples, True)
        real_output = self.discriminator(real_samples, True)
        dis_loss = self.cross_entropy(tf.ones_like(real_output), real_output) + \
                    self.cross_entropy(tf.zeros_like(faked_output), faked_output)
        return dis_loss / 2

#    @tf.function
    def train_gen(self, sli):
        noise_seed = tf.random.normal([self.num_samples, 1, self.rand_dims])
        with tf.GradientTape() as  g:
            faked_samples =  self.generators[sli](noise_seed, True)
            faked_output = self.discriminator(faked_samples, True)
            gen_loss =  self.cross_entropy(tf.ones_like(faked_output), faked_output)
        gen_gradients = g.gradient(gen_loss, self.generators[sli].trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generators[sli].trainable_variables))
        return gen_loss

    def train_step(self, target_loss):
        dis_loss = tf.reduce_mean([self.train_dis(sli) for sli in range(self.num_gen)])
        for i in range(100):
            gen_loss = tf.reduce_mean([self.train_gen(sli) for sli in range(self.num_gen)])
            if tf.reduce_mean([self.loss_dis(sli) for sli in range(self.num_gen)]) > target_loss:
                break
        return gen_loss, dis_loss

    def train(self, gen_name='CNN', epochs=10000, num_samples=20, sequence_length_list=[5, 10, 15]):
        if len(self.sampler.data) < 500:
            return None, None
        self.epochs = epochs
        self.num_samples = num_samples
        self.sequence_length_list = sequence_length_list
        self.num_gen = len(self.sequence_length_list)
        assert gen_name in ['CNN', 'LSTM'], 'gen_name is wrong, please check it.'
        if gen_name == 'CNN':
            self.generators = [Generator_CNN(rows=sli) for sli in sequence_length_list]
        else:
            self.generators = [Generator_LSTM(rows=sli) for sli in sequence_length_list]
        self.discriminator = Discriminator()
            
        target_loss = tf.reduce_mean([self.loss_dis(sli) for sli in range(self.num_gen)])
        gen_loss_sequence, dis_loss_sequence = [], []
        for _ in tqdm(range(self.epochs)):
            gen_loss, dis_loss = self.train_step(target_loss)
            gen_loss_sequence.append(gen_loss.numpy())
            dis_loss_sequence.append(dis_loss.numpy())
            # if np.mean(gen_loss_sequence[-5:]) > 98:
            #     break
        return gen_loss_sequence, dis_loss_sequence