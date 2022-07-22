---
layout: post
title: VAEs as Regularized Autoencoders: a Discrete Approach
---
 
About a year ago, I read an interesting paper titled ["From Variational To Deterministic Autoencoders"](https://arxiv.org/abs/1903.12436) by Ghosh et al. It is a really interesting paper that I suggest others read, since it explores the variational autoencoder at a low level and derives fromt it a general class of generative models.

In short, I decided to explore the idea of VAEs as "autoencoders with latent and decoder regularization" and applied it to the discrete domain.

Before delving into the findings and implications of my experiments, it would be noteworthy to highlight the main points of the aforementioned paper.

# Regularized Autoencoder: Simple but Genius

On the surface, variational autoencoders utilize an autoencoder setup in the context of variational inference. 

From a probabilistic perspective, we have a generative model p(x, z) that produces data, and an inference model q(z|x) that infers latent variables by approximating the true posterior distribution p(z|x).

From the deep neural network perspective, we have an encoder that compresses the data into latent code, and a decoder that reconstructs the data from the latent code. Additionally, there is a regularization term that shapes the latent space.

One key property of VAEs that separate them from vanilla autoencoders is that the encoder does not output a latent variable, but a distribution (more specifically the parameters of the distribution). To obtain a latent code, we take a random sample from this distribution.

There is a problem that arises here: it is not possible to take the gradient of a random sample. To combat this, we can take a sample with respect to the output of the encoder.

For Gaussian VAEs, this mean taking a sample from a unit Gaussian, and scaling it by the variance and adding the mean which is outputed by the encoder. It is now possible to train the network end-to-end.

The authors of the paper noticed an intriguing property of the reparameterization trick: by making the variance constant (set to one), reparameterization can be interpreted as injecting noise into the latent variable, or in other words, decoder regularization.

They went on to explain the importance of decoder regularization in smoothing out the latent space of the generative model, and subsequently demonstrated its effectiveness by replacing the noise injection with other explicit and implicit regularization methods, showing that the generative capacity of regularized autoencoders rivals that of VAEs.

The main takeaway here is that VAEs are a specific implementation of a regularized autoencoder, one in which decoder regularization is acheived through (probabilistic) noise injection. Of course, the downside is that regularized autoencoders lose their generative ability, requiring ex-post density estimation to sample data.

The authors of the paper used regularized autoencoders in the continuous setting, but did not not explore them in the discrete domain. In this hobby project of mine, curiosity got the best of me and I uncovered some interesting properties of RAEs.

# Discrete VAEs

