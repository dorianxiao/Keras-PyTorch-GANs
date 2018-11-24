from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import argparse

# ------
# Create GAN
# ------
class GAN():
    def __init__(self, opt):
        self.opt = opt

        # define a optimizer
        optimizer = Adam(self.opt.lr, self.opt.b1, self.opt.b2)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        
        # Build the generator
        self.generator = self.build_generator()

        # The generator takes generated images as input and generates images
        z = Input(shape=(self.opt.latent_dim,))
        img = self.generator(z)

        # For te combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The conbined mode (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy',
                              optimizer=optimizer)

    def build_generator(self):
        # ------
        # Build Generator Network
        # ------
        model = Sequential()

        model.add(Dense(256, input_dim=self.opt.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.opt.image_shape), activation='tanh'))
        model.add(Reshape(self.opt.image_shape))

        # print the network
        model.summary()

        noise = Input(shape=(self.opt.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.opt.image_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        # print the network
        model.summary()

        img = Input(shape=self.opt.image_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self):
        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Reshape -1 to 1
        X_train = X_train / 127.5 - 1
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((self.opt.batch_size, 1))
        fake = np.zeros_like(valid)

        for epoch in range(self.opt.n_epochs):
            # -----
            # Train Discriminator
            # -----

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], self.opt.batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (self.opt.batch_size, self.opt.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            # -----
            # Train Generator
            # -----
            noise = np.random.normal(0, 1, (self.opt.batch_size, self.opt.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the process
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % self.opt.sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.opt.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%08d.png" % epoch)
        plt.close()



def main():
    os.makedirs('images', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs',
                        type=int,
                        default=20000,
                        help='number of epoches of training')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='size of the batches')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0002,
                        help='adam: learning rate')
    parser.add_argument('--b1', 
                        type=float, 
                        default=0.5, 
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', 
                        type=float, 
                        default=0.999, 
                        help='adam: decay of second order momentum of gradient')
    parser.add_argument('--n_cpu', 
                        type=int, 
                        default=8, 
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', 
                        type=int, 
                        default=100, 
                        help='dimensionality of the latent space')
    parser.add_argument('--img_size', 
                        type=int, 
                        default=28, 
                        help='size of each image dimension')
    parser.add_argument('--channels', 
                        type=int, 
                        default=1, 
                        help='number of image channels')
    parser.add_argument('--sample_interval', 
                        type=int, 
                        default=40, 
                        help='interval betwen image samples')
    opt = parser.parse_args()

    # channel last
    opt.image_shape = (opt.img_size, opt.img_size, opt.channels)
    print(opt)

    gan = GAN(opt)
    gan.train()


if __name__ == '__main__':
    main()