"""
Solves the equations of fluid flow using "Stable Fluids" by Jos Stam with the
FFT to obtain ultra-fast simulations.

Momentum:           ∂u/∂t + (u ⋅ ∇) u = − 1/ρ ∇p + ν ∇²u + f

Incompressibility:  ∇ ⋅ u = 0

u:  Velocity (2d vector)
p:  Pressure
f:  Forcing
ν:  Kinematic Viscosity
ρ:  Density
t:  Time
∇:  Nabla operator (defining nonlinear convection, gradient and divergence)
∇²: Laplace Operator

----

A unit square domain with periodic boundary conditions on each edge
(= simulation on a torus)


          1 +-------------------------------------------------+
            |                                                 |
            |             *                      *            |
            |          *           *    *    *                |
        0.8 |                                                 |
            |                                 *               |
            |     *       *                                   |
            |                      *     *                    |
        0.6 |                                            *    |
            |      *                         <- <- <-         |
            |                             *  <- <- <-         |
            |     -> -> ->               *   <- <- <-         |
            |     -> -> ->    *                *         *    |
        0.4 |     -> -> ->                                    |
            |                                                 |
            |      *            *             *               |
            |           *                             *       |
        0.2 |                       *           *             |
            |                               *                 |
            |  *          *      *                 *       *  |
            |                            *                    |
          0 +-------------------------------------------------+
            0        0.2       0.4       0.6       0.8        1

-> Centered in y but ofset in x there are two horizontal forcings in opposite
   directions

-> Two fluid streams will "collide" with each other

----- 

Solution Strategy:

-> Start with zero velocity everywhere: u = [0, 0]

1. Add forces

    w₁ = u + Δt f

2. Convect by self-advection (set the value at the current
   location to be the value at the position backtraced
   on the streamline.) -> unconditionally stable

    w₂ = w₁(p(x, −Δt))

3. Diffuse and Project in Fourier Domain

    3.1 Forward Transformation into Fourier Domain

        w₂ → w₃
    
    3.2 Diffuse by "low-pass filtering" (convolution
        is multiplication in the Fourier Domain)

        w₄ = exp(− k² ν Δt) w₃
    
    3.3 Compute the (pseudo-) pressure in the Fourier Domain
        by evaluating the divergence in the Fourier Domain

        q = w₄ ⋅ k / ||k||₂
    
    3.4 Correct the velocities such that they are incompressible

        w₅ = w₄ − q k / ||k||₂
    
    3.5 Inverse Transformation back into spatial domain

        w₆ ← w₅

4. Repeat

k = [ kₓ, k_y ] are the spatial frequencies (= wavenumbers)

The Fourier Transformation implicitly prescribes the periodic
Boundary Conditions

------

Changes with respect to the original video (https://youtu.be/F7rWoxeGrko)

1. Use dark theme, set to equal aspect ratio and increase plot window size

2. Change to periodic clamping which respects the periodic boundary conditions,
   then also increase the number of time steps.

"""

using FFTW
using Plots   # Requires ColorSchemes.jl
using ProgressMeter
using Interpolations
using LinearAlgebra

N_POINTS = 250
KINEMATIC_VISCOSITY = 0.0001
TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 300

function backtrace!(
    backtraced_positions,
    original_positions,
    direction,
)
    # Euler Step backwards in time and periodically clamp into [0.0, 1.0]
    backtraced_positions[:] = mod1.(
        original_positions - TIME_STEP_LENGTH * direction,
        1.0,
    )
end

function interpolate_positions!(
    field_interpolated,
    field,
    interval_x,
    interval_y,
    query_points_x,
    query_points_y,
)
    interpolator = LinearInterpolation(
        (interval_x, interval_y),
        field,
    )
    field_interpolated[:] = interpolator.(query_points_x, query_points_y)
end

function main()
    element_length = 1.0 / (N_POINTS - 1)
    x_interval = 0.0:element_length:1.0
    y_interval = 0.0:element_length:1.0

    # Similar to meshgrid in NumPy
    coordinates_x = [x for x in x_interval, y in y_interval]
    coordinates_y = [y for x in x_interval, y in y_interval]

    wavenumbers_1d = fftfreq(N_POINTS) .* N_POINTS

    wavenumbers_x = [k_x for k_x in wavenumbers_1d, k_y in wavenumbers_1d]
    wavenumbers_y = [k_y for k_x in wavenumbers_1d, k_y in wavenumbers_1d]
    wavenumbers_norm = [norm([k_x, k_y]) for k_x in wavenumbers_1d, k_y in wavenumbers_1d]

    decay = exp.(- TIME_STEP_LENGTH .* KINEMATIC_VISCOSITY .* wavenumbers_norm.^2)

    wavenumbers_norm[iszero.(wavenumbers_norm)] .= 1.0
    normalized_wavenumbers_x = wavenumbers_x ./ wavenumbers_norm
    normalized_wavenumbers_y = wavenumbers_y ./ wavenumbers_norm

    # Define the forces
    force_x = 100.0 .* (
        exp.( - 1.0 / (2 * 0.005) * ((coordinates_x .- 0.2).^2 + (coordinates_y .- 0.45).^2))
        -
        exp.( - 1.0 / (2 * 0.005) * ((coordinates_x .- 0.8).^2 + (coordinates_y .- 0.55).^2))
    )

    # Preallocate all arrays
    backtraced_coordinates_x = zero(coordinates_x)
    backtraced_coordinates_y = zero(coordinates_y)

    velocity_x = zero(coordinates_x)
    velocity_y = zero(coordinates_y)

    velocity_x_prev = zero(velocity_x)
    velocity_y_prev = zero(velocity_y)

    velocity_x_fft = zero(velocity_x)
    velocity_y_fft = zero(velocity_y)
    pressure_fft = zero(coordinates_x)

    # Use a dark theme for plotting
    theme(:dark)

    @showprogress "Timestepping ..." for iter in 1:N_TIME_STEPS

        # (1) Apply the forces
        time_current = (iter - 1) * TIME_STEP_LENGTH
        pre_factor = max(1 - time_current, 0.0)
        velocity_x_prev += TIME_STEP_LENGTH * pre_factor * force_x

        # (2) Self-Advection by backtracing and interpolation
        backtrace!(backtraced_coordinates_x, coordinates_x, velocity_x_prev)
        backtrace!(backtraced_coordinates_y, coordinates_y, velocity_y_prev)
        interpolate_positions!(
            velocity_x,
            velocity_x_prev,
            x_interval,
            y_interval,
            backtraced_coordinates_x,
            backtraced_coordinates_y,
        )
        interpolate_positions!(
            velocity_y,
            velocity_y_prev,
            x_interval,
            y_interval,
            backtraced_coordinates_x,
            backtraced_coordinates_y,
        )

        # (3.1) Transform into Fourier Domain
        velocity_x_fft = fft(velocity_x)
        velocity_y_fft = fft(velocity_y)

        # (3.2) Diffuse by low-pass filtering
        velocity_x_fft .*= decay
        velocity_y_fft .*= decay

        # (3.3) Compute Pseudo-Pressure by Divergence in Fourier Domain
        pressure_fft = (
            velocity_x_fft .* normalized_wavenumbers_x
            +
            velocity_y_fft .* normalized_wavenumbers_y
        )

        # (3.4) Project the velocities to be incompressible
        velocity_x_fft -= pressure_fft .* normalized_wavenumbers_x
        velocity_y_fft -= pressure_fft .* normalized_wavenumbers_y

        # (3.5) Transform back into spatial domain
        velocity_x = real(ifft(velocity_x_fft))
        velocity_y = real(ifft(velocity_y_fft))

        # Advance in time
        velocity_x_prev = velocity_x
        velocity_y_prev = velocity_y

        # Visualize
        d_u__d_y = diff(velocity_x, dims=2)[2:end, :]
        d_v__d_x = diff(velocity_y, dims=1)[:, 2:end]
        curl = d_u__d_y - d_v__d_x
        display(heatmap(x_interval, y_interval, curl', c=:diverging_bkr_55_10_c35_n256, aspect_ratio=:equal, size=(680, 650)))
    end
end

main()
