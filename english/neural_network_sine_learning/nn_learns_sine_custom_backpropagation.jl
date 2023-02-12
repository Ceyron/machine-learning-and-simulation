"""
This script intends to be a hands-on own implementation of a simple MLP trained
by backpropagation (special case of reverse-mode autodiff) to learn the sin
function.

Simplifications:
- No train-val-test split
- No regularization
- No Stochastic Minibatching
- Simple Gradient Descent with constant learning rate
- No training modifications (e.g. batch normalization, dropout, etc.)

---

Given x ∈ ℜ^(1 x N) and y ∈ ℜ^(1 x N) with weight matrices and biases with
correct shape as well as the sigmoid activation function

    σ(x) = 1 / (1 + σ(- x))

The sigmoid has the nice property that

    σ'(x) = σ(x) * (1 - σ(x))

[NOTICE: Julia convention is spatial x batch, DL frameworks in Python use
opposite!]

---

Forward Pass:

    y_1 = W_1 * x
    y_1_plus_bias = y_1 .+ b_1
    y_1_activated = σ.(y_1_plus_bias)

    y_2 = W_2 * y_1_activated
    y_2_plus_bias = y_2 .+ b_2
    y_2_activated = σ.(y_2_plus_bias)

    y_3 = W_3 * y_2_activated
    y_3_plus_bias = y_3 .+ b_3
    y_3_activated = σ.(y_3_plus_bias)

    y_4 = W_4 * y_3_activated
    y_4_plus_bias = y_4 .+ b_4
    y_4_activated = I.(y_4_plus_bias)

    L = mean_over_batch(0.5 * ||y_4_activated .- y||_2^2 )

---

Weight Initialization:

    W_i_kl ∝ 𝒰(-lim, lim)
    lim = sqrt(6 / (fan_in_i + fan_out_i))
    b_i_k = 0.0

---

Backward Pass

    (Run Forward Pass and save all activated layer_states)

    d_L = 1.0
    d_y_4_activated = dL .* (y_4_activated .- y) ./ N

    d_y_4_plus_bias = d_y_4_activated ⊙ 1
     d_b_4 = sum_over_batch(d_y_4_plus_bias)
    d_y_4 = d_y_4_plus_bias
    d_y_3_activated = W_4' * d_y_4
     d_W_4 = d_y_4 * y_3_activated'

    d_y_3_plus_bias = d_y_3_activated ⊙ y_3_activated ⊙ (1 - y_3_activated)
     d_b_3 = sum_over_batch(d_y_3_plus_bias)
    d_y_3 = d_y_3_plus_bias
    d_y_2_activated = W_3' * d_y_3
     d_W_3 = d_y_3 * y_2_activated'
    
    d_y_2_plus_bias = d_y_2_activated ⊙ y_2_activated ⊙ (1 - y_2_activated)
     d_b_2 = sum_over_batch(d_y_2_plus_bias)
    d_y_2 = d_y_2_plus_bias
    d_y_1_activated = W_2' * d_y_2
     d_W_2 = d_y_2 * y_1_activated'
    
    d_y_1_plus_bias = d_y_1_activated ⊙ y_1_activated ⊙ (1 - y_1_activated)
     d_b_1 = sum_over_batch(d_y_1_plus_bias)
    d_y_1 = d_y_1_plus_bias
    [d_x = W_1' * d_y_1]
     d_W_1 = d_y_1 * x'

---

Learning:

    1. Initialize weights
    2. Run forward and backward pass (here for all samples together - no minibatching)
    3. Update each parameter with its gradient, e.g.
        W_1 ← W_1 - η * d_W_1
    4. Repeat until loss is suffiently decreased

---

More details on backward pass:

We involved the following primitive pullback/vJp/backprop/adjoint(¹) rules:

    1. Pullback from loss cotangent to guess cotangent (d_L → d_y_4_activated)

    2. Pullback over broadcasted function application (d_y_i_activated → d_y_i_plus_bias)
     3. Pullback over batch broadcast vector addition (d_y_i_plus_bias → d_b_i)
    4. Pullback over addition (d_y_i_plus_bias → d_y_i)
    5. Pullback over matrix-vector (or here matrix-matrix) multiplication to vector (d_y_i → d_y_(i-1)_activated)
     6. Pullback over matrix-vector (or here matrix-matrix) multiplication to matrix (d_y_i → d_W_i)

The indented pullback operations map back into the parameter space.

If you want more details, on why these (obsurd) rules are the way, they are,
here are more interesting videos:

    1.  
        a. Pullback of L2 Loss: https://youtu.be/TonUAqYCWAY
        b. Pullback of a (mean) aggregator, similar to scalar multiplication: https://youtu.be/ho8v1FpoaEg
    2. Pullback over broadcasted function: https://youtu.be/bLE6xsVSTUs
    3. 
        a. Pullback over scalar addition (holds also for vector and matrix addition): https://youtu.be/SY2ga4ylwVM
        b. Pullback over a scather operation: The reverse will just be the sum the cotangents over the scattered xis
    4. Pullback over scalar addition (holds also for vector and matrix addition):https://youtu.be/SY2ga4ylwVM
    5. 
        a. Pullback over a matrix-vector product: https://youtu.be/lqIhocjJLUc
        b. (More precisely we do a matrix-matrix product to account for batching): https://youtu.be/O5YealZxi68
    6.
        a. Pullback over a matrix-vector product: https://youtu.be/lqIhocjJLUc
        b. (More precisely we do a matrix-matrix product to account for batching): https://youtu.be/O5YealZxi68


(¹): The idea of how NNs obtain their parameter gradients in a backward pass is
a concept found in many disciplines of science. Hence, the history is filled
with multiple re-inventions
(https://en.wikipedia.org/wiki/Backpropagation#History) and namings vary.
"""

using Plots
using Random
using Distributions

SEED = 42
N_SAMPLES = 200
LAYERS = [1, 10, 10, 10, 1]
LEARNING_RATE = 0.1
N_EPOCHS = 30_000

rng = MersenneTwister(SEED)

# Julia (column-major) convention: spatial x batch (instead of batch x spatial
# you commonly see in Python)
x_samples = rand(rng, Uniform(0, 2 * pi), (1, N_SAMPLES))
y_samples = sin.(x_samples) .+ rand(rng, Normal(0.0, 0.3), (1, N_SAMPLES))

# The data we learn on, for now no train-val-test split, we just inspect the fit
# visually
scatter(x_samples[:], y_samples[:], label="data")

sigmoid(x) = 1.0 / (1.0 + exp(-x))

weight_matrices = []
bias_vectors = []
activation_functions = []

# Parameter Initialization
for (fan_in, fan_out) in zip(LAYERS[1:end-1], LAYERS[2:end])
    kernel_matrix_uniform_limit = sqrt(6 / (fan_in + fan_out))

    # Xavier Glorot uniform Initialization
    W = rand(
        rng,
        Uniform(-kernel_matrix_uniform_limit, +kernel_matrix_uniform_limit),
        fan_out,
        fan_in,
    )

    # Zeros bias Initialization
    b = zeros(fan_out)

    push!(weight_matrices, W)
    push!(bias_vectors, b)
    push!(activation_functions, sigmoid)
end

# The last layer is not activated
activation_functions[end] = identity

function network_forward(x, weights, biases, activations)
    # The first layer is the input
    a = x

    for (W, b, activation) in zip(weights, biases, activations)
        z = W * a .+ b
        a = activation.(z)
    end

    return a
end

scatter!(
    x_samples[:],
    network_forward(x_samples, weight_matrices, bias_vectors, activation_functions)[:],
    label="Initial fit",
)

function loss_forward(y, y_ref)
    delta = y .- y_ref
    loss = 0.5 * mean(sum(delta.^2, dims=1), dims=2)
    return loss[1]
end

loss_forward(
    network_forward(x_samples, weight_matrices, bias_vectors, activation_functions),
    y_samples,
)

function loss_backward(y, y_ref)
    delta = y .- y_ref
    N = size(y, 2)
    return delta / N
end

function network_forward_and_backward(x, y_ref, weights, biases, activations, activations_derivatives)
    # The first layer is the input
    a = x

    # Store the intermediate activated states for the backward pass
    layer_values = [a, ]

    for (W, b, activation) in zip(weights, biases, activations)
        z = W * a .+ b
        a = activation.(z)

        push!(layer_values, a)
    end

    y = a

    loss = loss_forward(y, y_ref)


    # Backward pass
    current_cotangent = loss_backward(y, y_ref)

    weights_gradients = []
    bias_gradients = []

    for (W, activation_prime, a_current, a_prev) in zip(
        reverse(weights),
        reverse(activations_derivatives),
        reverse(layer_values[2:end]),
        reverse(layer_values[1:end-1]),
    )
        activated_state_cotangent = current_cotangent
        plus_bias_state_cotangent = activated_state_cotangent .* activation_prime.(a_current)

        bias_grad = sum(plus_bias_state_cotangent, dims=2)

        state_cotangent = plus_bias_state_cotangent

        weight_grad = state_cotangent * a_prev'

        prev_activated_state_cotangent = W' * state_cotangent

        push!(weights_gradients, weight_grad)
        push!(bias_gradients, bias_grad)

        current_cotangent = prev_activated_state_cotangent
    end

    return loss, reverse(weights_gradients), reverse(bias_gradients)
end

sigmoid_prime(x_activated) = x_activated * (1 - x_activated)
identity_prime(x_activated) = 1

activations_derivatives = [[sigmoid_prime for _ in 1:length(LAYERS) - 2]...; identity_prime]

initial_loss, initial_weigh_grads, initial_bias_grads = network_forward_and_backward(
    x_samples,
    y_samples,
    weight_matrices,
    bias_vectors,
    activation_functions,
    activations_derivatives,
)

# Training loop
loss_history = []
for epoch in 1:N_EPOCHS
    loss, weight_grads, bias_grads = network_forward_and_backward(
        x_samples,
        y_samples,
        weight_matrices,
        bias_vectors,
        activation_functions,
        activations_derivatives,
    )

    for (W, W_grad, b, b_grad) in zip(
        weight_matrices,
        weight_grads,
        bias_vectors,
        bias_grads,
    )
        W .-= LEARNING_RATE .* W_grad
        b .-= LEARNING_RATE .* b_grad
    end

    if epoch % 100 == 0
        println("Epoch: $(epoch), loss: $(loss)")
    end

    push!(loss_history, loss)

end

scatter!(
    x_samples[:],
    network_forward(x_samples, weight_matrices, bias_vectors, activation_functions)[:]
)

plot(loss_history, yscale=:log10)