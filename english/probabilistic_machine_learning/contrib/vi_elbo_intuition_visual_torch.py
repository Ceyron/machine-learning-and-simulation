"""
Example script in PyTorch to complement the "elbo_intuitive_plot" script
and YouTube video. 
"""

from turtle import done
import numpy as np
import torch
import torch.distributions as dist
import seaborn as sns
import matplotlib.pyplot as plt


def plot_mixture(probs, x, elbo, kl, evidence):

    sns.lineplot(x=x, y=probs[:, 0])
    sns.lineplot(x=x, y=probs[:, 1])
    plt.legend(['True Posterior', 'Surrogate Posterior'])
    plt.title('${} = {} + {}$'.format(elbo, kl, evidence))
    plt.show()


def approximate_kl_divergence(p, q, size):
    """approx. kl-divergence b/c of LLN"""

    sample_size = torch.Size([size])
    sample_set = p.sample(sample_size)
    
    return torch.mean((p.log_prob(sample_set) - q.log_prob(sample_set))).numpy()


def true_posterior(mixture_probs, mus, sigmas):
    """mixture distribution with 3 components"""

    return dist.mixture_same_family.MixtureSameFamily(
        mixture_distribution=dist.Categorical(probs=mixture_probs),
        component_distribution=dist.Normal(loc=mus, scale=sigmas))


def surrogate_posterior(loc_approx, scale_approx):
    """surrogate posterior $Q(\theta)$"""

    return dist.normal.Normal(loc=loc_approx, scale=scale_approx)


def main():

    while True:
        mu_approx = torch.tensor(float(input('mu_approx = ')))
        sigma_approx = torch.tensor(float(input('sigma_approx = ')))

        mixture_probs = torch.tensor([0.3, 0.4, 0.3])
        mus = torch.tensor([-1.3, 2.2, 4.0])
        sigmas = torch.tensor([2.3, 1.5, 4.4])
        x = torch.linspace(-5, 5, 100)

        p = true_posterior(mixture_probs, mus, sigmas)
        prob_values_p = torch.exp2((p.log_prob(x))).reshape(-1, 1)

        q = surrogate_posterior(mu_approx, sigma_approx)
        prob_values_q = torch.exp2((q.log_prob(x))).reshape(-1, 1)

        all_prob_values = torch.hstack((prob_values_p, prob_values_q))

        mean_kl = approximate_kl_divergence(p, q, 100000)
        evidence = torch.mean(torch.log2(prob_values_p))  
        elbo = -mean_kl + evidence
        plot_mixture(all_prob_values, x, elbo, -mean_kl, evidence)

        done = str(input('done: [y/n]: '))

        if done == 'y':
            break
        else:
            pass


if __name__ == '__main__':
    main()