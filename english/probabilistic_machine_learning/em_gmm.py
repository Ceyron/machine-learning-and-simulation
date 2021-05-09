import silence_tensorflow.auto
import numpy as np
from tensorboard import data
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm import tqdm # progress-meter

def em(dataset, n_classes, n_iterations, random_seed):
    n_samples = dataset.shape[0]

    np.random.seed(random_seed)

    # Initial guesses for the parameters
    mus = np.random.rand(n_classes)
    sigmas = np.random.rand(n_classes)
    class_probs = np.random.dirichlet(np.ones(n_classes))

    for em_iter in tqdm(range(n_iterations)):
        # E-Step
        # responsibilities = np.zeros((n_samples, n_classes))

        # for i in range(n_samples):
        #     for c in range(n_classes):
        #         responsibilities[i, c] = class_probs[c] *\
        #             tfp.distributions.Normal(loc=mus[c], scale=sigmas[c]).prob(dataset[i])
        ################################################
        ## !!!! That part was wrong in the video !!!!
        ## Thanks to Anuj Shah for pointing it out
        ################################################
        ## -> The correct version will also multiply with the class probabilities
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
    
    return class_probs, mus, sigmas


def main():
    class_probs_true = [0.6, 0.4]
    mus_true = [2.5, 4.8]
    sigmas_true = [0.6, 0.3]
    random_seed = 42 # for reproducability
    n_samples = 1000
    n_iterations = 100
    n_classes = 2

    # generate the data
    univariate_gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=class_probs_true),
        components_distribution=tfp.distributions.Normal(
            loc=mus_true,
            scale=sigmas_true,
        )
    )

    # REMARK: I added the random_seed after the video as I unfortunately missed
    # it
    dataset = univariate_gmm.sample(n_samples, seed=random_seed).numpy()

    # print(dataset)

    class_probs, mus, sigmas = em(dataset, n_classes, n_iterations, random_seed)

    print(class_probs)
    print(mus)
    print(sigmas)

if __name__ == "__main__":
    main()