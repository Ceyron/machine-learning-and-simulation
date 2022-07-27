"""
Variational inference for unknown mean complementing the video
https://www.youtube.com/watch?v=dxwVMeK988Y
"""

import argparse
import torch
import numpy as np
from pyro.infer import Predictive
import pyro.distributions as dist
from pyro.poutine import trace
from torch.distributions import constraints, transforms
from pyro.infer import SVI, Trace_ELBO
import pyro
from pyro.optim import Adam
import seaborn as sns
import matplotlib.pyplot as plt



def plot_elbo(loss):

    plt.figure(figsize=(10, 3))
    plt.plot(np.arange(1, len(loss)+1), loss)
    plt.ylabel('ELBO Loss')
    plt.xlabel('Iterations')
    plt.title(f'iter {len(loss)}, loss: {loss[-1]:.4f}')
    plt.show()


def create_dataset():

    # true params. of dataset
    mu_true = 4.0
    sigma_true = 2.0
    sigma_fix = sigma_true

    N = torch.tensor(100)
    X = dist.Normal(mu_true, sigma_true).sample((N,))

    # priors
    mu_0 = 4.2
    sigma_0 = 0.3

    # params. of true posterior (analytical solution)
    mu_N = (sigma_fix**2 * mu_0 + sigma_0**2 * torch.sum(X)) / (sigma_fix**2 + N * sigma_0**2)
    sigma_N = (sigma_0 * sigma_fix) / (torch.sqrt(sigma_fix**2 + N * sigma_0**2))

    return N, X, mu_N, sigma_N

def generative_model(n_samples, obs=None):
    """
    generative model defines the joint distribution
    """

    mu = pyro.sample('mu', dist.Normal(4.2, 0.3))
    
    with pyro.plate('obs', n_samples):
        X = pyro.sample('X', dist.Normal(loc=mu, scale=2.), obs=obs)
    

def guide(n_samples, obs=None):
    """surrogate posterior"""

    mu_loc = pyro.param('mu_loc', torch.tensor(0.))
    mu_scale = pyro.param(
        'mu_scale', torch.tensor(1.), constraint=constraints.positive)
    mu = pyro.sample('mu', dist.Normal(mu_loc, mu_scale))


def main(args):

    N, X, mu_N, sigma_N = create_dataset()

    graph = pyro.render_model(
        generative_model, (N, X), render_distributions=True, filename='model.png'
        )
    
    pyro.clear_param_store()
    optim = Adam({'lr': args.lr})
    svi = SVI(generative_model, guide, optim, loss=Trace_ELBO())

    elbo_loss = []
    for step in range(args.iter):
        loss = svi.step(N, X)
        elbo_loss.append(loss)

    plot_elbo(elbo_loss)

    svi_mu = pyro.param('mu_loc')
    svi_sigma = pyro.param('mu_scale')

    print('Analytical Solution')
    print('-'*20)
    print(f'mu_N = {mu_N}')
    print(f'sigma_N = {sigma_N}')
    print('\n')
    print('SVI Solution')
    print('-'*20)
    print(f'mu_svi = {svi_mu}')
    print(f'sigma_svi = {svi_sigma}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VI for unknown mean')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--iter', type=int, default=1000)
    args = parser.parse_args()

    main(args)