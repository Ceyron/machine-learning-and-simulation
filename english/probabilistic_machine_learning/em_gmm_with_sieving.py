import silence_tensorflow.auto
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from tqdm import tqdm # progress-meter

# def em(dataset, n_classes, n_iterations, random_seed):
#     n_samples = dataset.shape[0]

#     np.random.seed(random_seed)

#     # Initial guesses for the parameters
#     mus = np.random.rand(n_classes)
#     sigmas = np.random.rand(n_classes)
#     class_probs = np.random.dirichlet(np.ones(n_classes))

#     for em_iter in tqdm(range(n_iterations)):
#         # E-Step
#         responsibilities = tfp.distributions.Normal(loc=mus, scale=sigmas).prob(
#             dataset.reshape(-1, 1)
#         ).numpy() * class_probs
        
#         responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True)

#         class_responsibilities = np.sum(responsibilities, axis=0)

#         # M-Step
#         for c in range(n_classes):
#             class_probs[c] = class_responsibilities[c] / n_samples
#             mus[c] = np.sum(responsibilities[:, c] * dataset) / class_responsibilities[c]
#             sigmas[c] = np.sqrt(
#                 np.sum(responsibilities[:, c] * (dataset - mus[c])**2) / class_responsibilities[c]
#             )

#         # Calculate the marginal log likelihood
#         log_likelihood = np.sum(
#             logsumexp(
#                 np.log(class_probs)
#                 +
#                 tfp.distributions.Normal(loc=mus, scale=sigmas).log_prob(
#                     dataset.reshape(-1, 1)
#                 ).numpy()
#                 ,
#                 axis=1
#             )
#             ,
#             axis=0
#         )
#         print(log_likelihood)
    
#     return class_probs, mus, sigmas

def em_with_guesses(
    dataset,
    n_iterations,
    class_probs_initial,
    mus_initial,
    sigmas_initial,
):
    n_classes = class_probs_initial.shape[0]
    n_samples = dataset.shape[0]

    class_probs = class_probs_initial.copy()
    mus = mus_initial.copy()
    sigmas = sigmas_initial.copy()

    log_likelihood_history = []

    for em_iter in tqdm(range(n_iterations)):
        # E-Step
        responsibilities = tfp.distributions.Normal(loc=mus, scale=sigmas).prob(
            dataset.reshape(-1, 1)
        ).numpy() * class_probs
        
        responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True)

        class_responsibilities = np.sum(responsibilities, axis=0)

        # M-Step
        for c in range(n_classes):
            class_probs[c] = class_responsibilities[c] / n_samples
            mus[c] = np.sum(responsibilities[:, c] * dataset) / class_responsibilities[c]
            sigmas[c] = np.sqrt(
                np.sum(responsibilities[:, c] * (dataset - mus[c])**2) / class_responsibilities[c]
            )

        # Calculate the marginal log likelihood
        log_likelihood = np.sum(
            logsumexp(
                np.log(class_probs)
                +
                tfp.distributions.Normal(loc=mus, scale=sigmas).log_prob(
                    dataset.reshape(-1, 1)
                ).numpy()
                ,
                axis=1
            )
            ,
            axis=0
        )
        log_likelihood_history.append(log_likelihood)
    
    return class_probs, mus, sigmas, log_likelihood_history

def em_sieved(
    dataset,
    n_classes,
    n_iterations_pre_sieving,
    n_candidates,
    n_iterations_post_sieving,
    n_chosen_ones,
    random_seed,
):

    # (1) Pre-Sieving

    mus_list = []
    sigmas_list = []
    class_probs_list = []
    log_likelihood_history_list = []

    for candidate_id in range(n_candidates):
        np.random.seed(random_seed + candidate_id)

        mus = np.random.rand(n_classes)
        sigmas = np.random.rand(n_classes)
        class_probs = np.random.dirichlet(np.ones(n_classes))

        class_probs, mus, sigmas, log_likelihood_history = em_with_guesses(
            dataset,
            n_iterations_pre_sieving,
            class_probs,
            mus,
            sigmas,
        )
        mus_list.append(mus)
        sigmas_list.append(sigmas)
        class_probs_list.append(class_probs)
        log_likelihood_history_list.append(log_likelihood_history)
    
    # (2) Sieving, select the best candidates
    log_likelihood_history_array = np.array(log_likelihood_history_list)

    # Sort in descending order
    ordered_candidate_ids = np.argsort( - log_likelihood_history_array[:, -1])
    chosen_ones_ids = ordered_candidate_ids[:n_chosen_ones]

    # (3) Post-Sieving
    mus_chosen_ones_list = []
    sigmas_chosen_ones_list = []
    class_probs_chosen_ones_list = []
    log_likelihood_history_chosen_ones_list = []
    for chosen_one_id in chosen_ones_ids:
        class_probs, mus, sigmas, log_likelihood_history = em_with_guesses(
            dataset,
            n_iterations_post_sieving,
            class_probs_list[chosen_one_id],
            mus_list[chosen_one_id],
            sigmas_list[chosen_one_id],
        )

        mus_chosen_ones_list.append(mus)
        sigmas_chosen_ones_list.append(sigmas)
        class_probs_chosen_ones_list.append(class_probs)
        log_likelihood_history_chosen_ones_list.append(log_likelihood_history)
    
    # (4) Select the very best candidate
    log_likelihood_history_chosen_ones_array = np.array(log_likelihood_history_chosen_ones_list)

    # Sort in descending order
    ordered_chosen_ones_ids = np.argsort( - log_likelihood_history_chosen_ones_array[:, -1])

    best_chosen_one_id = ordered_chosen_ones_ids[0]
    best_mus = mus_chosen_ones_list[best_chosen_one_id]
    best_sigmas = sigmas_chosen_ones_list[best_chosen_one_id]
    best_class_probs = class_probs_chosen_ones_list[best_chosen_one_id]

    return best_class_probs, best_mus, best_sigmas, log_likelihood_history_chosen_ones_array


def main():
    class_probs_true = [0.6, 0.4]
    mus_true = [2.5, 4.8]
    sigmas_true = [0.6, 0.3]
    random_seed = 42 # for reproducability
    n_samples = 1000
    n_iterations_pre_sieving = 5
    n_candidates = 100
    n_iterations_post_sieving = 100
    n_chosen_ones = 5
    n_classes = 2

    # generate the data
    univariate_gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=class_probs_true),
        components_distribution=tfp.distributions.Normal(
            loc=mus_true,
            scale=sigmas_true,
        )
    )

    dataset = univariate_gmm.sample(n_samples, seed=random_seed).numpy()

    class_probs, mus, sigmas, log_likelihood_histories = em_sieved(
        dataset,
        n_classes,
        n_iterations_pre_sieving,
        n_candidates,
        n_iterations_post_sieving,
        n_chosen_ones,
        random_seed,
    ) 

    print(class_probs)
    print(mus)
    print(sigmas)

    plt.figure()
    for log_likelihood_history in log_likelihood_histories:
        plt.plot(log_likelihood_history)
    plt.xlabel("Iteration")
    plt.ylabel("Log Likelihood")
    plt.show()


if __name__ == "__main__":
    main()