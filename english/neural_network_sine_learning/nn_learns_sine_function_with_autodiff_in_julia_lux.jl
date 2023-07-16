using Lux
using Random
using Zygote
using Optimisers
using Plots
using Distributions
using Statistics

N_SAMPLES = 200
LAYERS = [1, 10, 10, 10, 1]
LEARNING_RATE = 0.1
N_EPOCHS = 30_000

# Our Pseudo-Random Number Generator
rng = Xoshiro(42)

# Draw a toy dataset
x_samples = rand(
    rng,
    Uniform(0.0, 2 * π),
    (1, N_SAMPLES),
)
y_noise = rand(
    rng,
    Normal(0.0, 0.3),
    (1, N_SAMPLES),
)
y_samples = sin.(x_samples) .+ y_noise

scatter(x_samples[:], y_samples[:], label="data")

# Define the model architecture
model = Chain(
    [Dense(fan_in => fan_out, Lux.sigmoid) for (fan_in, fan_out) in zip(LAYERS[1:end-2], LAYERS[2:end-1])]...,
    Dense(LAYERS[end-1] => LAYERS[end], identity),
)

# Initialize the parameters (and also the layer states, only relevant if neural
# network was stateful)
parameters, layer_states = Lux.setup(rng, model)

y_initial_prediction, layer_states = model(x_samples, parameters, layer_states)
scatter!(x_samples[:], y_initial_prediction[:], label="initial prediction")

# The forward function
function loss_fn(p, ls)
    y_prediction, new_ls = model(x_samples, p, ls)
    loss = 0.5 * mean((y_prediction .- y_samples).^2)
    return loss, new_ls
end

# Use plain gradient descent, but swap to ADAM if you want
opt = Descent(LEARNING_RATE)
opt_state = Optimisers.setup(opt, parameters)

# Train loop
loss_history = []
for epoch in 1:N_EPOCHS
    (loss, layer_states,), back = pullback(loss_fn, parameters, layer_states)
    grad, _ = back((1.0, nothing))

    opt_state, parameters = Optimisers.update(opt_state, parameters, grad)

    push!(loss_history, loss)
    if epoch % 100 == 0
        println("Epoch: $epoch, Loss: $loss")
    end
end

plot(loss_history, yscale=:log10)

# Plot final prediction
y_final_prediction, layer_states = model(x_samples, parameters, layer_states)
scatter(x_samples[:], y_samples[:], label="data")
scatter!(x_samples[:], y_final_prediction[:], label="final prediction")