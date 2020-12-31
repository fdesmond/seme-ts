#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:49:01 2020

@author: yiye
"""

import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model
from keras.layers import Input, RepeatVector 
from keras.layers import LSTM
from keras.layers.core import Dense, Lambda 
from keras import objectives
from keras.optimizers import RMSprop

class lstm_autoencoder_vae:
    def __init__(self, input_dim, timesteps,\
                 batch_size, intermediate_dim, \
                 latent_dim, epsilon_std=1., learning_rate=0.001):
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        self.learning_rate = learning_rate

    def sampling(self, args):
        # z_log_sigma = log(sigma^2)
        z_mean, z_log_sigma = args 
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim),\
                                          mean=0.,stddev=self.epsilon_std)
        return z_mean + z_log_sigma * epsilon
        
    def generate_model(self): 
        # Encoder 
        x = Input(shape=(self.timesteps, self.input_dim), name="Main_input_VAE")
        h = LSTM(self.intermediate_dim, \
                 kernel_initializer='random_uniform',\
                 input_shape=(self.timesteps,self.input_dim,)
                 )(x)
        z_mean = Dense(self.latent_dim)(h)
        z_log_sigma = Dense(self.latent_dim)(h)
            
        z = Lambda(self.sampling,output_shape=(self.latent_dim,) )([z_mean, z_log_sigma])
        
        # Decoder                               
        decoder_h = LSTM(self.intermediate_dim, \
                    kernel_initializer='random_uniform', \
                    input_shape=(self.timesteps,self.latent_dim,), \
                    return_sequences=True
                    ) 
        decoder_mean = LSTM(self.input_dim, \
                    kernel_initializer='random_uniform', \
                    input_shape=(self.timesteps,self.intermediate_dim,), \
                    return_sequences=True
                    ) 
        h_decoded = RepeatVector(self.timesteps)(z)
        h_decoded = decoder_h(h_decoded)           
        # decoded layer
        x_decoded_mean = decoder_mean(h_decoded)
            
        # end-to-end autoencoder
        vae = Model(x, x_decoded_mean)
            
        # encoder, from inputs to latent space
        encoder = Model(x, [z_mean, z_log_sigma, z])
            
        # generator, from latent space to reconstructed inputs: 
        # sample from the learned latent space distributions
        decoder_input = Input(shape=(self.latent_dim,)) 
        _h_decoded = RepeatVector(self.timesteps)(decoder_input)
        _h_decoded = decoder_h(_h_decoded)
        _x_decoded_mean = decoder_mean(_h_decoded) 
        generator = Model(decoder_input, _x_decoded_mean)
        
        def vae_loss(x_loss, x_decoded_mean_loss): 
            """
            Loss function for the Variational AUto-Encoder 
            :param x_loss:
            :param x_decoded_mean_loss:
            :return:
            """
            xent_loss = objectives.mse(x_loss, x_decoded_mean_loss)
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)) 
            loss = xent_loss + kl_loss
            return loss
        opt_rmsprop = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-4, decay=0)
        vae.compile(optimizer=opt_rmsprop, loss=vae_loss)
        
        return vae, encoder, generator


def get_data():
    # read data from file
    data = np.fromfile('sample_data.dat').reshape(419,13)
    timesteps = 3
    dataX = []
    for i in range(len(data) - timesteps - 1):
        x = data[i:(i+timesteps), :]
        dataX.append(x)
    return np.array(dataX)


if __name__ == "__main__":
    x = get_data()
    input_dim = x.shape[-1] # 13
    timesteps = x.shape[1] # 3
    batch_size = 1

    m = lstm_autoencoder_vae(input_dim, 
        timesteps=timesteps, 
        batch_size=batch_size, 
        intermediate_dim=32,
        latent_dim=100,
        epsilon_std=1.)
    vae, enc, dec = m.generate_model()

    vae.fit(x, x, epochs=50)

    preds = vae.predict(x, batch_size=batch_size)
    preds2 = vae.predict(x, batch_size=batch_size)
    z_mean, z_log_sigma, z = enc.predict(x, batch_size=batch_size)
    preds3 = dec.predict(z, batch_size=batch_size)

    # Change the sampling std: distortion level in this method
    distortion_level = 2.
    z_ = np.zeros(shape=(x.shape[0], m.latent_dim))
    for n in range(x.shape[0]):
        epsilon = np.random.normal(0., scale=distortion_level, size = m.latent_dim)
        z_[n:,] = z_mean[n:,] + z_log_sigma[n:,] * epsilon
    preds4 = dec.predict(z_, batch_size=batch_size)

    # pick a column to plot.
    print("[plotting...]")
    print("x: %s, preds: %s" % (x.shape, preds.shape))
    plt.plot(x[:,0,3], label='data')
    plt.plot(preds[:,0,3], label='predict')
    plt.plot(preds2[:,0,3], label='predict2')
    plt.plot(preds3[:,0,3], label='predict3')
    plt.plot(preds4[:,0,3], label='predict4')
    plt.legend()
    plt.show()




