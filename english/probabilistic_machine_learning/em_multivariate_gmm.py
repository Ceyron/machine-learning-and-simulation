import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import matplotlib.pyplot as plt

def em(dataset, n_clusters, n_iter=100):
    # Infer from the dataset
    n_samples, n_dims = dataset.shape

    # Draw initial guesses
    cluster_probs = tfp.distributions.Dirichlet(tf.ones(n_clusters)).sample(seed=42)
    mus = tfp.distributions.Normal(loc=0.0, scale=3.0).sample((n_clusters, n_dims), seed=42)
    covs = tfp.distributions.WishartTriL(df=3, scale_tril=tf.eye(n_dims)).sample(n_clusters, seed=42)

    for _ in tqdm(range(n_iter)):
        # Batched Cholesky Factorization
        Ls = tf.linalg.cholesky(covs)
        normals = tfp.distributions.MultivariateNormalTriL(
            loc=mus,
            scale_tril=Ls
        )


        ### E-Step

        # (1) resp is of shape (n_samples x n_clusters)
        # batched multivariate normal is of shape (n_clusters x n_dims)
        unnormalized_responsibilities = (
            tf.reshape(cluster_probs, (1, n_clusters)) * normals.prob(tf.reshape(dataset, (n_samples, 1, n_dims)))
        )

        # (2)
        responsibilities = unnormalized_responsibilities / tf.reduce_sum(unnormalized_responsibilities, axis=1, keepdims=True)

        # (3)
        class_responsibilities = tf.reduce_sum(responsibilities, axis=0)

        ### M-Step

        # (1)
        cluster_probs = class_responsibilities / n_samples

        # (2)
        # class_responsibilities is of shape (n_clusters)
        # responsibilities is of shape (n_samples, n_clusters)
        # dataset is of shape (n_samples, n_dims)
        #
        # mus is of shape (n_clusters, n_dims)
        #
        # -> summation has to occur over the samples axis
        mus = tf.reduce_sum(
            tf.reshape(responsibilities, (n_samples, n_clusters, 1)) * tf.reshape(dataset, (n_samples, 1, n_dims)),
            axis=0,
        ) / tf.reshape(class_responsibilities, (n_clusters, 1))
        
        # (3)
        # class_responsibilities is of shape (n_clusters)
        # dataset is of shape (n_samples, n_dims)
        # mus is of shape (n_clusters, n_dims)
        # responsibilities is of shape (n_samples, n_clusters)
        #
        # covs is of shape (n_clusters, n_dims, n_dims)
        
        # (n_clusters, n_samples, n_dims)
        centered_datasets = tf.reshape(dataset, (1, n_samples, n_dims)) - tf.reshape(mus, (n_clusters, 1, n_dims))
        centered_datasets_with_responsibilities = centered_datasets * tf.reshape(tf.transpose(responsibilities), (n_clusters, n_samples, 1))
        
        # Batched Matrix Multiplication
        # (n_clusters, n_dims, n_dims)
        sample_covs = tf.matmul(centered_datasets_with_responsibilities, centered_datasets, transpose_a=True)

        covs = sample_covs / tf.reshape(class_responsibilities, (n_clusters, 1, 1))


        # Ensure positive definiteness by adding a "small amount" to the diagonal
        covs = covs + 1.0e-8 * tf.eye(n_dims, batch_shape=(n_clusters, ))

    
    return cluster_probs, mus, covs


def main():
    N_CLUSTERS = 2
    CLUSTER_PROBS = [0.3, 0.7]
    MUS_TRUE = [
        [5.0, 5.0],
        [-3.0, -2.0],
    ]
    COVS_TRUE = [
        [
            [1.5, 0.5],
            [0.5, 2.0],
        ],
        [
            [1.5, 0.0],
            [0.0, 1.8],
        ]
    ]
    N_SAMPLES = 1000

    # Batched Cholesky factorization of the covariance matrices
    LS_TRUE = tf.linalg.cholesky(COVS_TRUE)

    # The true Gaussian Mixture Model (we want to use for sampling some
    # artificial data)
    cat = tfp.distributions.Categorical(
        probs=CLUSTER_PROBS,
    )
    normals = tfp.distributions.MultivariateNormalTriL(
        loc=MUS_TRUE,
        scale_tril=LS_TRUE,
    )

    gmm_true = tfp.distributions.MixtureSameFamily(
        mixture_distribution=cat,
        components_distribution=normals,
    )

    dataset = gmm_true.sample(N_SAMPLES, seed=42)

    # print(dataset)
    # plt.scatter(dataset.numpy()[:, 0], dataset.numpy()[:, 1])
    # plt.show()

    class_probs_approx, mus_approx, covs_approx = em(dataset, N_CLUSTERS)
    
    print("------")
    print("Class Probabilities")
    print(class_probs_approx)
    print("------")
    print("Mus")
    print(mus_approx)
    print("------")
    print("Covariance Matrices")
    print(covs_approx)


if __name__ == "__main__":
    main()