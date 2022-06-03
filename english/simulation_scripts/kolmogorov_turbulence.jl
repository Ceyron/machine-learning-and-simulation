"""
Solves the equations of fluid flow using a "Stable Fluids"-like algorithm
based on the FFT in a Kolmogorov Flow scenario. This setup results in forced
isotropic turbulence.

Momentum:           ∂u/∂t + (u ⋅ ∇) u = − ∇p + ν ∇²u + f

Incompressibility:  ∇ ⋅ u = 0

u:  Velocity (2d vector)
p:  Pressure
f:  Forcing (here Kolmogorov Forcing)
ν:  Kinematic Viscosity
t:  Time
∇:  Nabla operator (defining nonlinear convection, gradient and divergence)
∇²: Laplace Operator

----

Setup:

A rectangular domain (16x9) with periodic boundary conditions in both axes.

            +-------------------------------------------------+
            |                                                 |
            |             *                      *            |
            |          *           *    *    *                |
            |                                                 |
            |                                 *               |
            |     *       *                                   |
            |                      *     *                    |
            |                                            *    |
            |                                                 |
            |  *          *      *                 *       *  |
            |                            *                    |
            +-------------------------------------------------+

-> The fluid is initialized at rest

-> Then, a forcing is applied according to

        ⎡   c * sin(k * y)  ⎤
    f = ⎢                   ⎥
        ⎣         0         ⎦

        with scaling c and wavenumber k

----- 

Solution Strategy:

-> Start with zero velocity everywhere: u = [0, 0]

1. Add forces

    w₁ = u + Δt f

2. Convect by self-advection (set the value at the current
   location to be the value at the position backtraced
   on the streamline.) -> unconditionally stable

    w₂ = w₁(p(x, −Δt))

3. Subtract mean velocity for stabilization

4. Diffuse and Project in Fourier Domain

    4.1 Forward Transformation into Fourier Domain

        w₂ → w₃
    
    4.2 Diffuse by "low-pass filtering" (convolution
        is multiplication in the Fourier Domain)

        w₄ = exp(− k² ν Δt) w₃
    
    4.3 Compute the (pseudo-) pressure in the Fourier Domain
        by evaluating the divergence in the Fourier Domain

        q = w₄ ⋅ k / ||k||₂
    
    4.4 Correct the velocities such that they are incompressible

        w₅ = w₄ − q k / ||k||₂
    
    4.5 Inverse Transformation back into spatial domain

        w₆ ← w₅

5. Subtract mean velocity for stabilization

6. Repeat

k = [ k_x, k_y ] are the spatial frequencies (= wavenumbers)

The Fourier Transformation implicitly prescribes the periodic
Boundary Conditions

-----

To create a video/gif out of the saved images run

    ffmpeg -f image2 -framerate 24 -i kolmogorov_%05d.png kolmogorov_animation.gif

"""

using FFTW
using Plots
using ProgressMeter
using Interpolations
using LinearAlgebra
using Printf
using Statistics

N_POINTS_Y = 360
ASPECT_RATIO = 16/9
KINEMATIC_VISCOSITY = 1.0 / 1000.0
TIME_STEP_LENGTH = 0.001
N_TIME_STEPS = 750

FORCING_WAVENUMBER = 8
FORCING_SCALE = 100.0

function main()
    n_points_x = Int(ASPECT_RATIO * N_POINTS_Y)
    x_extent = n_points_x / N_POINTS_Y - 1.0e-7
    x_interval = range(0.0, x_extent, length=n_points_x)
    y_interval = range(0.0, 1.0, length=N_POINTS_Y)

    coordinates_x = [x for x in x_interval, y in y_interval]
    coordinates_y = [y for x in x_interval, y in y_interval]

    wavenumbers_1d_x = rfftfreq(n_points_x) .* n_points_x
    n_fft_points_x = length(wavenumbers_1d_x)
    wavenumbers_1d_y = fftfreq(N_POINTS_Y) .* N_POINTS_Y
    n_fft_points_y = length(wavenumbers_1d_y)

    wavenumbers_x = [k_x for k_x in wavenumbers_1d_x, k_y in wavenumbers_1d_y]
    wavenumbers_y = [k_y for k_x in wavenumbers_1d_x, k_y in wavenumbers_1d_y]
    wavenumbers_norm = [norm([k_x, k_y]) for k_x in wavenumbers_1d_x, k_y in wavenumbers_1d_y]

    decay = exp.(- TIME_STEP_LENGTH .* KINEMATIC_VISCOSITY .* wavenumbers_norm.^2)
    wavenumbers_norm[iszero.(wavenumbers_norm)] .= 1.0
    normalized_wavenumbers_x = wavenumbers_x ./ wavenumbers_norm
    normalized_wavenumbers_y = wavenumbers_y ./ wavenumbers_norm

    force_x = FORCING_SCALE * sin.(FORCING_WAVENUMBER * pi * coordinates_y)

    # Preallocate arrays
    backtraced_coordinates_x = zeros(Float32, n_points_x, N_POINTS_Y)
    backtraced_coordinates_y = zeros(Float32, n_points_x, N_POINTS_Y)

    velocity_x = zeros(Float32, n_points_x, N_POINTS_Y)
    velocity_y = zeros(Float32, n_points_x, N_POINTS_Y)

    velocity_x_prev = zeros(Float32, n_points_x, N_POINTS_Y)
    velocity_y_prev = zeros(Float32, n_points_x, N_POINTS_Y)

    velocity_x_fft = zeros(Complex{Float32}, n_fft_points_x, n_fft_points_y)
    velocity_y_fft = zeros(Complex{Float32}, n_fft_points_x, n_fft_points_y)
    pressure_fft = zeros(Complex{Float32}, n_fft_points_x, n_fft_points_y)

    d_u_d_y_fft = zeros(Complex{Float32}, n_fft_points_x, n_fft_points_y)
    d_v_d_x_fft = zeros(Complex{Float32}, n_fft_points_x, n_fft_points_y)
    curl_fft = zeros(Complex{Float32}, n_fft_points_x, n_fft_points_y)
    curl = zeros(Float32, n_points_x, N_POINTS_Y)

    interpolator = interpolate(
        (x_interval, y_interval),
        velocity_x,
        Gridded(Linear()),
    )

    theme(:dark)

    @showprogress "Timestepping ..." for iter in 1:N_TIME_STEPS
        # (1) Apply the forces
        velocity_x_prev .+= force_x * TIME_STEP_LENGTH

        # (2) Self-Advection by backtracing and interpolation
        backtraced_coordinates_x .= mod1.(
            coordinates_x - TIME_STEP_LENGTH * velocity_x_prev,
            x_extent,
        ) 
        backtraced_coordinates_y .= mod1.(
            coordinates_y - TIME_STEP_LENGTH * velocity_y_prev,
            1.0,
        )
        interpolator.coefs .= velocity_x_prev
        velocity_x .= interpolator.(backtraced_coordinates_x, backtraced_coordinates_y)
        interpolator.coefs .= velocity_y_prev
        velocity_y .= interpolator.(backtraced_coordinates_x, backtraced_coordinates_y)

        # (3) Stabilize by subtracting the mean velocities
        velocity_x .-= mean(vec(velocity_x))
        velocity_y .-= mean(vec(velocity_y))

        # (4.1) Transform into Fourier domain
        velocity_x_fft = rfft(velocity_x)
        velocity_y_fft = rfft(velocity_y)

        # (4.2) Diffuse by low-pass filtering
        velocity_x_fft .*= decay
        velocity_y_fft .*= decay

        # (4.3) Compute the Pseudo-Pressure by Divergence in Fourier Domain
        pressure_fft = (
            velocity_x_fft .* normalized_wavenumbers_x
            +
            velocity_y_fft .* normalized_wavenumbers_y
        )

        # (4.4) Project the velocities to be incompressible
        velocity_x_fft -= pressure_fft .* normalized_wavenumbers_x
        velocity_y_fft -= pressure_fft .* normalized_wavenumbers_y

        # (4.5) Transform back into spatial domain
        velocity_x = irfft(velocity_x_fft, n_points_x)
        velocity_y = irfft(velocity_y_fft, n_points_x)

        # (5) Stabilize by subtracting the mean velocities
        velocity_x .-= mean(vec(velocity_x))
        velocity_y .-= mean(vec(velocity_y))

        # (6) Advance in time
        velocity_x_prev = velocity_x
        velocity_y_prev = velocity_y

        # Visualize
        d_u_d_y_fft = im .* wavenumbers_y .* velocity_x_fft
        d_v_d_x_fft = im .* wavenumbers_x .* velocity_y_fft
        curl_fft = d_v_d_x_fft - d_u_d_y_fft
        curl = irfft(curl_fft, n_points_x)

        curl = sign.(curl) .* sqrt.(abs.(curl) ./ quantile(vec(curl), 0.8))

        heatmap(
            x_interval,
            y_interval,
            curl',
            c = :seaborn_icefire_gradient,
            size = (1920, 1080),
            clim = (-5.0, 5.0),
            axis = false,
            showaxis = false,
            legend = :none,
            ticks = false,
            margin = 0.0Plots.mm,
            annotations = (
                0.06,
                1.0 - 0.06,
                Plots.text(
                    "$(@sprintf("iter: %05d", iter))",
                    pointsize = 40,
                    color = :white,
                    halign = :left,
                )
            )
        )
        savefig("kolmogorov_$(@sprintf("%05d", iter)).png")

    end

end

main()

