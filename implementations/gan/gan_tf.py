import argparse
import os
import numpy as np
import math

import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs',  
                    type=int,
                    default=200,
                    help='number of epochs of training')
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
                    default=1,
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
                    help='interval between image samples')
opt = parser.parse_args()
print(opt)

img_shape = (opt.img_size, opt.img_size, opt.channels)

def save_images(gen_imgs, epochs):
    r, c = 5, 5
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


def block(inputs, out_shape, name=None, normalize=True, reuse=False):
    with tf.variable_scope(name) as sp:
        x = tf.layers.dense(inputs, out_shape, name=sp.name+'fully_connected', reuse=reuse)
        if normalize:
            x = tf.layers.batch_normalization(x, momentum=0.8, name=sp.name+'batch_normalized')
        x = tf.nn.leaky_relu(x, name=sp.name+'leaky_relu')
    # tf.summary.histogram(name, x)
    return x

with tf.variable_scope('Generator'):
    # accept noise
    noise_in = tf.placeholder(tf.float32, [None, opt.latent_dim], name='noise')
    x = block(noise_in, 128, 'first', normalize=False)
    x = block(x, 256, 'second')
    x = block(x, 512, 'third')
    x = block(x, 1024, 'forth')
    x = tf.layers.dense(x, int(np.prod(img_shape)))
    G_out = tf.nn.tanh(x)
    # tf.summary.histogram('G_out', G_out)

with tf.variable_scope('Discriminator'):
    # accept real image
    real_img = tf.placeholder(tf.float32, [None, int(np.prod(img_shape))], name='real')

    x = block(real_img, 512, 'first', normalize=False)
    x = block(x, 256, 'second', normalize=False)
    prob_real = tf.layers.dense(x, 1, tf.nn.sigmoid, name='out')

    x = block(G_out, 512, 'first', normalize=False, reuse=True)
    x = block(x, 256, 'second', normalize=False, reuse=True)
    prob_fake = tf.layers.dense(x, 1, tf.nn.sigmoid, name='out', reuse=True)

D_loss = -tf.reduce_mean(tf.log(prob_real)+tf.log(1-prob_fake))
tf.summary.histogram('discriminator loss', D_loss)

G_loss = tf.reduce_mean(tf.log(1-prob_fake))
tf.summary.histogram('generator loss', G_loss)

train_D = tf.train.AdamOptimizer(opt.lr, ).minimize(D_loss, 
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))

train_G = tf.train.AdamOptimizer(opt.lr).minimize(G_loss,
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../data/mnist', one_hot=False)
r, c = 5, 5
fig, axs = plt.subplots(r, c)
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    # writer the network to a file
    writer = tf.summary.FileWriter('../../logs', sess.graph)
    sess.run(tf.global_variables_initializer())

    total_batch = int(len(mnist.train.labels) / opt.batch_size)

    for epoch in range(opt.n_epochs):
        for i in range(total_batch):

            # ------
            # Train Discriminator, many times
            # ------
            p_r, p_f = 0, 1
            cnt = 0
            
            while(np.mean(p_r) < 0.55 or np.mean(p_f) > 0.45):
                # prepare the data
                noise = np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))
                real_image, _ = mnist.train.next_batch(batch_size=opt.batch_size)

                d_l, p_r, p_f, _ = sess.run([D_loss, prob_real, prob_fake, train_D], {noise_in:noise, real_img:real_image})
                cnt = cnt + 1

            print('Discriminator: %d'%(cnt))
            print('p_r: %f, p_f: %f'%(np.mean(p_r), np.mean(p_f)))
            

            # ------
            # Train Generator, only once
            # ------
            prob_f_ = 0
            cnt = 0
            
            # while(np.mean(prob_f_) < 0.5):
            noise = np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))
            g_l, prob_f_, _ = sess.run([G_loss, prob_fake, train_G], {noise_in:noise})
            cnt = cnt + 1
        
            # print('Generator: %d'%(cnt))
            # print('FAKE: %f(1)' % np.mean(prob_f_))

            batches_done = epoch * total_batch + i
            if i % opt.sample_interval == 0:
                noise = np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))
                real_image, _ = mnist.train.next_batch(batch_size=opt.batch_size)
                d_l, g_l, p_r, p_f, _, _, results = sess.run([D_loss, G_loss, prob_real, prob_fake, train_D, train_G, merged], {noise_in:noise, real_img:real_image})
                writer.add_summary(results, i)
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f (-1.38)] [G loss: %f], [prob real: %f(1)], [prob fake: %f(0)]"
                        % (epoch, opt.n_epochs, i, total_batch, -d_l, g_l, np.mean(p_r), np.mean(p_f)))
                noise = np.random.normal(0, 1, (r * c, opt.latent_dim))
                gen_imgs = sess.run([G_out], {noise_in:noise})
                # gen_imgs, _ = mnist.train.next_batch(batch_size=25)
                # Rescale images 0 - 1
                gen_imgs = np.add(0.5,  np.multiply(0.5, gen_imgs))
                gen_imgs = np.reshape(gen_imgs, [r*c, *img_shape])
                
                cnt = 0
                for i in range(r):
                    for j in range(c):
                        axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                        axs[i,j].axis('off')
                        cnt += 1
                fig.savefig("../../images/%08d.png" % batches_done)

plt.close()
