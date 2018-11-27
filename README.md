<p align="center"><img src="assets/logo.jpg" width="480"\></p>

## Keras-PyTorch-GANs
The implementation of variable GANs by Keras and PyTorch. Contributions and suggestions of GANs to implement are very welcome.

<b> Merged from </b> [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN) and [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)

## Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [GAN](#gan)
    + [Adversarial Autoencoder](#adversarial-autoencoder)

## Installation
    $ git clone https://github.com/LittleLampChen/Keras-PyTorch-GANs.git
    $ cd Keras-PyTorch-GANs
    $ sudo pip3 install -r requirements.txt

## Implementations
### GAN
_Generative Adversarial Network_

#### Authors
Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio

#### Abstract
We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.

[[Paper]](https://arxiv.org/abs/1406.2661) [[Code_torch]](implementations/gan/gan_torch.py) [[Code_keras]](implementations/gan/gan_keras.py)[[Code_tf]](implementations/gan/gan_tf.py)

#### Run Example
```
$ cd implementations/gan/
$ python3 gan.py
```

<p align="center">
    <img src="assets/gan_torch.gif" width="640"\>
</p>

### Adversarial Autoencoder
Implementation of _Adversarial Autoencoder_.

[Code](aae/aae.py)

Paper: https://arxiv.org/abs/1511.05644

#### Example
```
$ cd aae/
$ python3 aae.py
```

<p align="center">
    <img src="assets/aae.gif" width="640"\>
</p>