#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:22:08 2021

@author: yiye
"""
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, RepeatVector 
from keras.layers import LSTM
from keras.layers.core import Dense, Lambda 
from keras.optimizers import RMSprop

class lstm_vae:
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
        z_mean, z_sigma = args 
        epsilon = tf.random_normal(shape=(self.batch_size, self.latent_dim),\
                                          mean=0.,stddev=self.epsilon_std)
        return z_mean + z_sigma * epsilon
        
    def generate_model(self): 
        # Encoder 
        x = Input(shape=(self.timesteps, self.input_dim), name="Main_input_VAE")
        h = LSTM(self.intermediate_dim, \
                 kernel_initializer='random_uniform',\
                 input_shape=(self.timesteps,self.input_dim,)
                 )(x)
        z_mean = Dense(self.latent_dim)(h)
        z_sigma = Dense(self.latent_dim)(h)
            
        z = Lambda(self.sampling,output_shape=(self.latent_dim,) )([z_mean, z_sigma])
        
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
        encoder = Model(x, [z_mean, z_sigma, z])
            
        # generator, from latent space to reconstructed inputs: 
        # sample from the learned latent space distributions
        decoder_input = Input(shape=(self.latent_dim,)) 
        _h_decoded = RepeatVector(self.timesteps)(decoder_input)
        _h_decoded = decoder_h(_h_decoded)
        _x_decoded_mean = decoder_mean(_h_decoded) 
        generator = Model(decoder_input, _x_decoded_mean)
        
        def vae_loss(x_loss, x_decoded_mean_loss): 
            """
            Loss function for the Variational Auto-Encoder 
            :param x_loss:
            :param x_decoded_mean_loss:
            :return:
            """
            # here we additionally take mean over the number of input features and the one of latent features respectively
            # since it brings us the better result
            xent_loss = tf.reduce_mean(tf.square(x_loss - x_decoded_mean_loss))
            kl_loss = - 0.5 * tf.reduce_mean(1 + tf.log(tf.square(z_sigma)) - tf.square(z_mean) - tf.square(z_sigma))
            loss = xent_loss + kl_loss
            return loss
        opt_rmsprop = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-4, decay=0)
        vae.compile(optimizer=opt_rmsprop, loss=vae_loss)
        
        return vae, encoder, generator
    
def get_data(X, timesteps):
    input_dim = X.shape[-1]
    sample = np.zeros(shape=(len(X)-timesteps+1, timesteps, input_dim), dtype = np.float32)
    for i in range(len(X)-timesteps+1):
        sample[i,:,:] = np.copy(X[i:(i+timesteps),:])
    return sample
