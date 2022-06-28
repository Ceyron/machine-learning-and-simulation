"""
Variational inference for the exponential distribution
"""

import torch
import torch.distributions as dist
#import pyro.distributions as dist
import pyro
import seaborn as sns
import matplotlib.pyplot as plt


def plot_mixture(probs, x):

    sns.lineplot(x=x, y=probs[:, 0])
    sns.lineplot(x=x, y=probs[:, 1])
    plt.legend(['True Posterior', 'Surrogate Posterior'])
    plt.show()


def kl_divergence(p, q):
    """computes dissimilarity between two distributions"""

    registered = dist.kl.register_kl(p, q)
    print(registered)

    return dist.kl_divergence(p, q)


def true_posterior(mixture_probs, mus, sigmas):
    """mixture distribution with 3 components"""

    return dist.mixture_same_family.MixtureSameFamily(
        mixture_distribution=dist.Categorical(probs=mixture_probs),
        component_distribution=dist.Normal(loc=mus, scale=sigmas))


def surrogate_posterior(loc_approx, scale_approx):
    """surrogate posterior $Q(\theta)$"""

    return dist.normal.Normal(loc=loc_approx, scale=scale_approx)


def main():

    #mu_approx = torch.tensor(float(input('mu_approx = ')))
    #sigma_approx = torch.tensor(float(input('sigma_approx = ')))

    mu_approx = torch.tensor(1.3)
    sigma_approx = torch.tensor(0.75)

    mixture_probs = torch.tensor([0.3, 0.4, 0.3])
    mus = torch.tensor([-1.3, 2.2, 4.0])
    sigmas = torch.tensor([2.3, 1.5, 4.4])
    x = torch.linspace(-5, 5, 100)

    tp = true_posterior(mixture_probs, mus, sigmas)
    prob_values_tp = torch.exp2((tp.log_prob(x))).reshape(-1, 1)

    qp = surrogate_posterior(mu_approx, sigma_approx)
    prob_values_qp = torch.exp2((qp.log_prob(x))).reshape(-1, 1)

    all_prob_values = torch.hstack((prob_values_tp, prob_values_qp))
    plot_mixture(probs=all_prob_values, x=x)

    #loss = kl_divergence(tp, qp)

    #kl = kl_divergence(tp, qp)
    #evidence = torch.min(kl)
    #elbo = -kl + evidence


if __name__ == '__main__':
    main()