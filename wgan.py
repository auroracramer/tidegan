import torch
from torch import autograd
from torch import optim
import numpy as np
import pprint


def compute_discr_loss_terms(model_dis, model_gen, real_data_v, noise_v,
                             batch_size, latent_dim,
                             lmbda, use_cuda, compute_grads=False):
    # Convenient values for
    one = torch.FloatTensor([1])
    neg_one = one * -1
    if use_cuda:
        one = one.cuda()
        neg_one = neg_one.cuda()

    # Reset gradients
    model_dis.zero_grad()

    # a) Compute loss contribution from real training data and backprop
    # (negative of the empirical mean, w.r.t. the data distribution, of the discr. output)
    D_real = model_dis(real_data_v)
    D_real = D_real.mean()
    # Negate since we want to _maximize_ this quantity
    if compute_grads:
        D_real.backward(neg_one)

    # b) Compute loss contribution from generated data and backprop
    # (empirical mean, w.r.t. the generator distribution, of the discr. output)
    # Generate noise in latent space

    # Generate data by passing noise through the generator
    fake = autograd.Variable(model_gen(noise_v).data)
    inputv = fake
    D_fake = model_dis(inputv)
    D_fake = D_fake.mean()
    if compute_grads:
        D_fake.backward(one)

    # c) Compute gradient penalty and backprop
    gradient_penalty = calc_gradient_penalty(model_dis, real_data_v.data,
                                             fake.data, batch_size, lmbda,
                                             use_cuda=use_cuda)

    if compute_grads:
        gradient_penalty.backward(one)

    # Compute metrics and record in batch history
    D_cost = D_fake - D_real + gradient_penalty
    Wasserstein_D = D_real - D_fake

    return D_cost, Wasserstein_D


def compute_gener_loss_terms(model_dis, model_gen, batch_size, latent_dim,
                             use_cuda, compute_grads=False):
    # Convenient values for
    one = torch.FloatTensor([1])
    neg_one = one * -1
    if use_cuda:
        one = one.cuda()
        neg_one = neg_one.cuda()

    # Reset generator gradients
    model_gen.zero_grad()

    # Sample from the generator
    noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
    if use_cuda:
        noise = noise.cuda()
    noise_v = autograd.Variable(noise)
    fake = model_gen(noise_v)

    # Compute generator loss and backprop
    # (negative of empirical mean (w.r.t generator distribution) of discriminator output)
    G = model_dis(fake)
    G = G.mean()
    if compute_grads:
        G.backward(neg_one)
    G_cost = -G

    return G_cost


def np_to_input_var(data, use_cuda):
    data = data[:,np.newaxis,:]
    data = torch.Tensor(data)
    if use_cuda:
        data = data.cuda()
    return autograd.Variable(data)


# Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
def calc_gradient_penalty(model_dis, real_data, fake_data, batch_size, lmbda, use_cuda=True):
    # Compute interpolation factors
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    # Interpolate between real and fake data
    interpolates = alpha * real_data.data + ((1 - alpha) * fake_data.data)
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    # Evaluate discriminator
    disc_interpolates = model_dis(interpolates)

    # Obtain gradients of the discriminator with respect to the inputs
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Compute MSE between 1.0 and the gradient of the norm penalty to encourage
    # discriminator to be a 1-Lipschitz function
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lmbda
    return gradient_penalty

