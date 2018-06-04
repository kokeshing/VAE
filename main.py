import os
import sys
import tensorflow as tf
import vae_tensorflow

SAVE_DIR = "C:/Users/ennpe/src/VAE/result/"
IMG_SAVE_DIR = "./result_img/"

# parameters
lr = 0.001
mn = 0.9
steps = 10000
batch_size = 64

# network structure
image_dim = 784
hidden_dim = 512
latent_dim = 2

vae = vae_tensorflow.VAE(input_dim=image_dim, hidden_dim=hidden_dim, z_dim=latent_dim)

vae.vae_train(batch_size=batch_size, steps=steps, lr=lr, mn=mn, save_dir=SAVE_DIR)

vae.generate_image(generate_num=20, save_dir=IMG_SAVE_DIR, batch_size=batch_size)
