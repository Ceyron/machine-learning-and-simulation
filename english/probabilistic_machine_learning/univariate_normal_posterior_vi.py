import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

def gauss_gamma(alpha_0, beta_0, mu_0, tau_0):
    tau = yield tfp.distributions.JointDistributionCoroutine.Root(
        tfp.distributions.Gamma(alpha_0, beta_0, name="tau")
    )
    mu = yield tfp.distributions.Normal(
        loc=mu_0, scale=1.0/tf.sqrt(tau * tau_0), name="mu"
    )

def gauss_times_gamma(alpha_0, beta_0, mu_0, tau_0):
    tau = yield tfp.distributions.JointDistributionCoroutine.Root(
        tfp.distributions.Gamma(alpha_0, beta_0, name="tau")
    )
    mu = yield tfp.distributions.JointDistributionCoroutine.Root(
        tfp.distributions.Normal(loc=mu_0, scale=1.0/tf.sqrt(tau_0), name="mu")
    )


def surrogate_posterior_parameter(dataset, alpha_0, beta_0, mu_0, tau_0, max_iter=100):
    N = dataset.shape[0]
    mu_mle = np.mean(dataset)
    variance_mle = np.mean((dataset - mu_mle)**2)

    mu_s = (tau_0 * mu_0 + N * mu_mle) / (tau_0 + N)
    alpha_s = alpha_0 + (N + 1) / 2.0

    beta_s = beta_0
    for _ in range(max_iter):
        tau_s = (tau_0 + N) * alpha_s / beta_s
        beta_s = beta_0 + (
            0.5 * (tau_0 + N) / tau_s
            +
            (tau_0 * N) * (mu_0 - mu_mle)**2 / (2 * (tau_0 + N))
            +
            0.5 * N * variance_mle
        )
    
    return alpha_s, beta_s, mu_s, tau_s

def true_posterior_parameter(dataset, alpha_0, beta_0, mu_0, tau_0):
    N = dataset.shape[0]
    mu_mle = np.mean(dataset)
    variance_mle = np.mean((dataset - mu_mle)**2)

    mu_N = (tau_0 * mu_0 + N * mu_mle) / (tau_0 + N)
    tau_N = tau_0 + N
    alpha_N = alpha_0 + N / 2.0
    beta_N = beta_0 +\
        0.5 * N * variance_mle +\
        (tau_0 * N * (mu_mle - mu_0)**2) / ( 2 * (tau_0 + N))
    
    return alpha_N, beta_N, mu_N, tau_N


def main():
    # Define constants and hyper-parameters
    MU_TRUE = 0.0
    TAU_TRUE = 1.0
    N_SAMPLES = 100
    RANDOM_SEED = 42

    ALPHA_0 = 1.0
    BETA_0 = 1.0
    MU_0 = 0.0
    TAU_0 = 1.0

    # Generate a dataset
    X_true = tfp.distributions.Normal(
        loc=MU_TRUE,
        scale=1.0/tf.sqrt(TAU_TRUE),
    )
    dataset = X_true.sample(N_SAMPLES, seed=RANDOM_SEED).numpy()

    # Surrogate posterior
    alpha_s, beta_s, mu_s, tau_s = surrogate_posterior_parameter(
        dataset,
        ALPHA_0,
        BETA_0,
        MU_0,
        TAU_0
    )
    print("Parameters of the Surrogate Posterior (a Normal times Gamma)")
    print(f"-> alpha_s={alpha_s:1.4f}, beta_s={beta_s:1.4f}, mu_s={mu_s:1.4f}, tau_s={tau_s:1.4f}")

    # True posterior
    alpha_N, beta_N, mu_N, tau_N = true_posterior_parameter(
        dataset,
        ALPHA_0,
        BETA_0,
        MU_0,
        TAU_0,
    )
    print("Parameters of the True Posterior (a Normal-Gamma)")
    print(f"-> alpha_N={alpha_N:1.4f}, beta_N={beta_N:1.4f}, mu_N={mu_N:1.4f}, tau_N={tau_N:1.4f}")

    # Surrogate MAP estimate
    mu_MAP_s = mu_s
    tau_MAP_s = (alpha_s - 1.0) / beta_s
    print("Surrogate MAP estimate")
    print(f"-> mu_MAP_s={mu_MAP_s:1.4f}, tau_MAP_s={tau_MAP_s:1.4f}")

    # True MAP estimate
    mu_MAP_N = mu_N
    tau_MAP_N = (alpha_N - 0.5) / beta_N
    print("True MAP estimate")
    print(f"-> mu_MAP_N={mu_MAP_N:1.4f}, tau_MAP_N={tau_MAP_N:1.4f}")

    # Contour plot
    mu_range = tf.linspace(-1.0, 1.0, 40)
    tau_range = tf.linspace(0.01, 1.5, 40)
    # mu_range = tf.cast(mu_range, tf.float32)
    # tau_range = tf.cast(tau_range, tf.float32)
    mu_mesh, tau_mesh = tf.meshgrid(mu_range, tau_range)
    # Has to adhere to the order in the DGM
    points_2d = (tau_mesh, mu_mesh)

    # Convert to tensorflow objects
    alpha_s, beta_s, mu_s, tau_s = tf.convert_to_tensor((alpha_s, beta_s, mu_s, tau_s))
    alpha_N, beta_N, mu_N, tau_N = tf.convert_to_tensor((alpha_N, beta_N, mu_N, tau_N))
    
    surrogate_posterior = tfp.distributions.JointDistributionCoroutineAutoBatched(
        lambda : gauss_times_gamma(alpha_s, beta_s, mu_s, tau_s)
    )

    true_posterior = tfp.distributions.JointDistributionCoroutineAutoBatched(
        lambda : gauss_gamma(alpha_N, beta_N, mu_N, tau_N)
    )

    log_prob_true_posterior = true_posterior.log_prob(points_2d)
    log_prob_surrogate_posterior = surrogate_posterior.log_prob(points_2d)

    plt.figure()
    plt.contour(mu_mesh, tau_mesh, log_prob_surrogate_posterior, levels=100)
    plt.xlabel("mu")
    plt.ylabel("tau")
    plt.title("Surrogate Posterior")

    plt.figure()
    plt.contour(mu_mesh, tau_mesh, log_prob_true_posterior, levels=100)
    plt.xlabel("mu")
    plt.ylabel("tau")
    plt.title("True Posterior")
    plt.show()




if __name__ == "__main__":
    main()