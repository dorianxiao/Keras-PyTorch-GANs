import argparse
import os
import numpy as np
import math
import itertools

import tensorflow as tf
import matplotlib.pyplot as plt


def block(inputs, out_shape, name=None, normalize=True, reuse=False):
    with tf.variable_scope(name):
        x = tf.layers.dense(
            inputs, out_shape, name='fully_connected', reuse=reuse)
        if normalize:
            x = tf.layers.batch_normalization(
                x, momentum=0.8, name='batch_normalization')
        x = tf.nn.leaky_relu(x, name='leaky_relu')

    return x


def main():

    os.makedirs('images', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=200,
        help='number of epochs of training')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument(
        '--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument(
        '--b1',
        type=float,
        default=0.5,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--b2',
        type=float,
        default=0.999,
        help='adam: decay of second order momentum of gradient')
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=100,
        help='dimensionality of the latent space')
    parser.add_argument(
        '--img_size',
        type=int,
        default=28,
        help='size of each image dimension')
    parser.add_argument(
        '--sample_interval',
        type=int,
        default=40,
        help='interval between image samples')
    opt = parser.parse_args()
    # Channel last
    opt.image_shape = (opt.img_size, opt.img_size,
                       1)  # (width, height, channels)
    print(opt)

    # ------
    # Create Generator Network
    # ------
    with tf.variable_scope('Encoder'):
        # Accept real image
        real_in = tf.placeholder(
            tf.float32, [None, *opt.image_shape], name='real_in')

        x = tf.reshape(real_in, [-1, np.prod(opt.image_shape)])
        x = block(x, 512, 'first_layer', normalize=False)
        x = block(x, 512, 'second_layer')

        mu = tf.layers.dense(x, opt.latent_dim, name='encoder_mu')
        logvar = tf.layers.dense(x, opt.latent_dim, name='encoder_logvar')

        random = tf.random.normal(np.array([opt.latent_dim]), 0, 1)

        logstd = tf.multiply(random, tf.exp(logvar / 2), name='logvar_std')

        x = tf.add(logstd, mu)

        encoder_code = tf.nn.sigmoid(x, name='encoder_code')

    with tf.variable_scope('Decoder'):
        # Accept encoder code
        x = block(encoder_code, 512, 'first_layer', normalize=False)
        x = block(x, 512, 'second_layer')
        x = tf.layers.dense(x, int(np.prod(opt.image_shape)))
        x = tf.nn.tanh(x, name='last_layer')

        decoder_image = tf.reshape(
            x, [-1, *opt.image_shape], name='reshape_image')

    # ------
    # Create Discriminator Network
    # ------
    with tf.variable_scope('Discriminator'):
        # Accept a encoder_code / noise
        noise_code_in = tf.placeholder(
            tf.float32, [None, opt.latent_dim], name='noise_code_in')

        x = block(encoder_code, 512, 'first_layer', normalize=False)
        x = block(x, 256, 'second_layer', normalize=False)
        prob_code = tf.layers.dense(x, 1, name='prob')

        x = block(
            noise_code_in, 512, 'first_layer', normalize=False, reuse=True)
        x = block(x, 256, 'second_layer', normalize=False, reuse=True)
        prob_noise = tf.layers.dense(x, 1, name='prob', reuse=True)

    # Define Loss
    adversarial_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=prob_code, labels=tf.ones_like(prob_code)),
        name='adversalrial_loss')

    pixelwise_loss = tf.reduce_mean(
        tf.abs(tf.subtract(decoder_image, real_in)), name='pixelwise_loss')

    generator_loss = tf.add(
        0.001 * adversarial_loss,
        0.999 * pixelwise_loss,
        name='generator_loss')
    tf.summary.histogram('generator loss', generator_loss)

    # prod_noise is groudtruth, prod_code is fake
    discriminator_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=prob_noise, labels=tf.ones_like(prob_noise)))
    discriminator_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=prob_code, labels=tf.zeros_like(prob_code)))
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    tf.summary.histogram('discriminator loss', discriminator_loss)

    # Define Optimizer
    train_discriminator = tf.train.AdamOptimizer(
        opt.lr, opt.b1, opt.b2, name="discriminator_opti").minimize(
            discriminator_loss,
            var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))

    train_generator = tf.train.AdamOptimizer(
        opt.lr, opt.b1, opt.b2, name="generator_opti").minimize(
            generator_loss,
            var_list=[
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder'),
                tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')
            ])
    # Load data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('data/mnist', one_hot=False)
    # picture set
    r, c = 8, 8
    fig, axs = plt.subplots(r, c)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs', sess.graph)

        # run init operation
        sess.run(tf.global_variables_initializer())

        total_batch = int(len(mnist.train.labels) / opt.batch_size)

        for epoch in range(opt.n_epochs):
            for batch in range(total_batch):
                # prepare data
                code_noise = np.random.normal(0, 1,
                                              (opt.batch_size, opt.latent_dim))
                real_image, _ = mnist.train.next_batch(
                    batch_size=opt.batch_size)
                # rescale to -1, 1
                real_image = real_image * 2 - 1
                real_image = np.reshape(real_image, [-1, *opt.image_shape])
                # print("%f, %f" % (np.max(real_image), np.min(real_image)))
                discriminator_loss_value, generator_loss_value, results, gen_image, _, _ = \
                    sess.run([discriminator_loss, generator_loss, merged, decoder_image, train_discriminator, train_generator],
                        {real_in: real_image, noise_code_in: code_noise})

                if batch % opt.sample_interval == 0:
                    writer.add_summary(results, batch)
                    print("[Epoch %d/%d] [Batch %.2f] \n[dis loss: %f][gen loss: %f]" % (\
                        epoch, opt.n_epochs, float(batch) / float(total_batch), discriminator_loss_value, generator_loss_value,))

                    gen_image = gen_image / 2 + 0.5
                    gen_image = np.reshape(gen_image,
                                           [opt.batch_size, *opt.image_shape])
                    cnt = 0
                    for i in range(r):
                        for j in range(c):
                            axs[i, j].imshow(
                                gen_image[cnt, :, :, 0], cmap='gray')
                            axs[i, j].axis('off')
                            cnt += 1
                    fig.savefig(
                        "images/%08d.png" % (epoch * total_batch + batch))
    plt.close()


if __name__ == '__main__':
    main()