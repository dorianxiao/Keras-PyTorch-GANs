import argparse
import os
import numpy as np
import math

import tensorflow as tf
import matplotlib.pyplot as plt

def block(inputs, out_shape, name=None, normalize=True, reuse=False):
    with tf.variable_scope(name):
        x = tf.layers.dense(inputs, out_shape, name='fully_connected', reuse=reuse)
        if normalize:
            x = tf.layers.batch_normalization(x, momentum=0.8, name='batch_normalization')
        x = tf.nn.leaky_relu(x, name='leaky_relu')

    return x


def main():

    os.makedirs('../../images', exist_ok=True)

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
    parser.add_argument('--latent_dim',
                        type=int, 
                        default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--img_size',
                        type=int,
                        default=28,
                        help='size of each image dimension')
    parser.add_argument('--sample_interval',
                        type=int,
                        default=40,
                        help='interval between image samples')
    opt = parser.parse_args()
    print(opt)

    # Channel last
    image_shape = (opt.img_size, opt.img_size, 1) # (width, height, channels)

    # ------
    # Create Generator Network
    # ------
    with tf.variable_scope('Generator'):
        # Accept noise
        noise_in = tf.placeholder(tf.float32, [None, opt.latent_dim], name='noise_in')
        x = block(noise_in, 128, 'first_layer', normalize=False)
        x = block(x, 256, 'second_layer')
        x = block(x, 512, 'third_layer')
        x = block(x, 1024, 'last_layer')
        fake_image = tf.layers.dense(x, int(np.prod(image_shape)), activation=tf.nn.tanh, name='fake_image')

    # ------
    # Create Discriminator Network
    # ------
    with tf.variable_scope('Discriminator'):
        # Accept a real image
        real_in = tf.placeholder(tf.float32, [None, int(np.prod(image_shape))], name='real_in')

        x = block(real_in, 512, 'first_layer', normalize=False)
        x = block(x, 256, 'second_layer', normalize=False)
        prob_real = tf.layers.dense(x, 1, name='prob')

        x = block(fake_image, 512, 'first_layer', normalize=False, reuse=True)
        x = block(x, 256, 'second_layer', normalize=False, reuse=True)
        prob_fake = tf.layers.dense(x, 1, name='prob', reuse=True)

    # Define Loss
    # discriminator_loss = -tf.reduce_mean(tf.log(prob_real) + tf.log(1 - prob_fake))
    discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=prob_real, labels=tf.ones_like(prob_real)))
    discriminator_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=prob_fake, labels=tf.zeros_like(prob_fake)))
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    tf.summary.histogram('discriminator loss', discriminator_loss)

    # generator_loss = -tf.reduce_mean(tf.log(prob_fake))
    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=prob_fake, labels=tf.ones_like(prob_fake)))
    tf.summary.histogram('generator loss', generator_loss)

    # Define Optimizer
    train_discriminator = tf.train.AdamOptimizer(opt.lr, opt.b1, opt.b2, name="discriminator_opti").minimize(discriminator_loss, 
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))

    train_generator = tf.train.AdamOptimizer(opt.lr, opt.b1, opt.b2, name="generator_opti").minimize(generator_loss,
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))
    
    # Load data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('../../data/mnist', one_hot=False)
    # picture set
    r, c = 5, 5
    fig, axs = plt.subplots(r, c)

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('../../logs', sess.graph)

        # run init operation
        sess.run(tf.global_variables_initializer())

        total_batch = int(len(mnist.train.labels) / opt.batch_size)

        for epoch in range(opt.n_epochs):
            for batch in range(total_batch):
                # prepare data
                noise_image = np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))
                real_image, _ = mnist.train.next_batch(batch_size=opt.batch_size)
                # resacale to -1, 1
                real_image = real_image / 127.5 - 1

                # ------
                # Train
                # According to the paper, it's better for us to train 
                # Discriminator Network many times, and Generator once.
                # However, it works well if we only train both Discriminator
                # and Generator Network once.
                # ------
                discriminator_loss_value, generator_loss_value, prob_fake_value, prob_real_value, results, _, _ = \
                    sess.run([discriminator_loss, generator_loss, prob_fake, prob_real, merged, train_discriminator, train_generator],
                        {noise_in:noise_image, real_in:real_image})

                if batch % opt.sample_interval == 0:
                    writer.add_summary(results, batch)
                    print("[Epoch %d/%d] [Batch %.2f] \n[dis loss: %f][gen loss: %f][prob_fake: %f][prob_real: %f]\n" % (\
                        epoch, opt.n_epochs, float(batch) / float(total_batch), discriminator_loss_value, generator_loss_value,
                        np.mean(prob_fake_value), np.mean(prob_real_value)))

                    noise = np.random.normal(0, 1, (r * c, opt.latent_dim))
                    gen_imgs = sess.run([fake_image], {noise_in:noise})
                    
                    # scale to 0, 1
                    gen_imgs = np.add(0.5,  np.multiply(0.5, gen_imgs))

                    gen_imgs = np.reshape(gen_imgs, [r*c, *image_shape])
                    
                    cnt = 0
                    for i in range(r):
                        for j in range(c):
                            axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                            axs[i,j].axis('off')
                            cnt += 1
                    fig.savefig("../../images/%08d.png" % (epoch * total_batch + batch) )
    plt.close()
if __name__ == '__main__':
    main()