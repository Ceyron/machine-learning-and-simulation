"""
Solves the incompressible Navier-Stokes equations in three dimensions using a
pseudo-spectral approach involving the FFT. Simulates the 3D Taylor-Green vortex
from vortex stretching to fully turbulent to dissipation. The approach follows
the "SpectralDNS in Python" paper by Mortensen and Langtangen:
https://arxiv.org/pdf/1602.03638v1.pdf


Momentum:           ∂u/∂t + (u ⋅ ∇) u = − 1/ρ ∇p + ν ∇²u

Incompressibility:  ∇ ⋅ u = 0


u :  Velocity (3d vector)
p :  Pressure
ν :  Kinematic Viscosity 
ρ :  Density (here=1) 
t :  Time
∇ :  Nabla operator (defining nonlinear convection, gradient and divergence)
∇²:  Laplace Operator

------

Scenario:

The flow is simulated on a Ω = (0, 2π)³ domain (in 3D). The initial condition is
according to

    u(t=0, x, y, z) =   sin(x)cos(y)cos(z)

    v(t=0, x, y, z) = − cos(x)sin(y)cos(z)

    w(t=0, x, y, z) = 0

No forcing is applied.

------

Expected Outcome:

(Following page 18 of
https://doc.nektar.info/tutorials/latest/incns/taylor-green-vortex/incns-taylor-green-vortex.pdf)

1. Laminar Vortex sheets

2. Vortex stretching

3. Rearrangement

4. Breakdown

5. Fully turbulent

6. Dissipation & Decay

------

Solution Strategy:

u  : The velocity field (3D vector) in spatial domain
λ  : The velocity field (3D vector) in Fourier domain

ω  : The vorticity field (3D vector) in spatial domain
η  : The vorticity field (3D vector) in Fourier domain

m  : The product of vorticity and velocity (3D vector) in spatial domain
ϕ  : The product of vorticity and velocity (3D vector) in Fourier domain

k  : The wavenumber vector (3D vector)

ψ  : The pressure in Fourier domain (1D scalar)

b  : The rhs of the ODE system in Fourier domain (3D vector)

i  : The imaginary unit (1D scalar)


0. Initialize the velocity vectors according to the IC and transform them into
   Fourier Domain

1. Compute the curl in Fourier Domain by means of a cross
   product with the wavenumber vector and imaginary unit

    η = ℱ(ω) = ℱ(∇ × u) = i k × λ

2. Transform the vorticity back to spatial domain

    ω = ℱ⁻¹(η)

3. Compute the "convection" by means of a cross product

    m = u × ω

4. Transform the "convection" back into Fourier domain

    ϕ = ℱ(m)

5. Perform a dealising on the convection in order to suppress
   unresolved wave numbers

6. Compute the (pseudo) "pressure" in Fourier domain

    ψ = (k ⋅ ϕ) / ||k||₂

7. Compute the rhs to the ODE system

    b = ϕ - ν ||k||₂² λ - (k ψ) / ||k||₂

8. Advance the velocity in Fourier Domain by means of an
   Explicit Euler time step
   
   λ ← λ + Δt b

9. Transform the newly obtained velocity back into spatial 
   domain

   u = ℱ⁻¹(λ)

10. [Optional] visualize the vorticity magnitude in spatial
   domain interactively

11. Repeat from (1.)


In total, it takes us three (three-dimensional) Fourier Transforms
per time iteration:
- Transformation of the curl to spatial domain
- Transformation of the "convection" to Fourier Domain
- Transformation of the velocity to spatial domain

It is called pseudo-spectral because some operations are performed
in Fourier Domain and some in the spatial domain.

-------

Stability Consideration

Choose the time step size carefully.

-----

Additional Hints:

Check out the Python Repo:
https://github.com/spectralDNS/spectralDNS

In particular this file:
https://github.com/spectralDNS/spectralDNS/blob/master/spectralDNS3D_short.py

"""

using FFTW
using GLMakie
using Statistics
using ProgressMeter

N_POINTS_P_AXIS = 50
KINEMATIC_VISCOSITY = 1.0 / 1_600
TIME_STEP_LENGTH = 0.02
N_TIME_STEPS = 1_700
PLOT_EVERY = 3

function cross_product!(
    res_x::Array{T, 3},
    res_y::Array{T, 3},
    res_z::Array{T, 3},
    a_x::Array{T, 3},
    a_y::Array{T, 3},
    a_z::Array{T, 3},
    b_x::Array{T, 3},
    b_y::Array{T, 3},
    b_z::Array{T, 3},
) where T<:Union{Float64, Complex{Float64}}
    res_x .= (
        a_y .* b_z
        .-
        a_z .* b_y
    )

    res_y .= (
        a_z .* b_x
        .-
        a_x .* b_z
    )

    res_z .= (
        a_x .* b_y
        .-
        a_y .* b_x
    )
end

function main()
    x_range = range(0.0, 2*pi, N_POINTS_P_AXIS+1)[1:end-1]
    y_range = range(0.0, 2*pi, N_POINTS_P_AXIS+1)[1:end-1]
    z_range = range(0.0, 2*pi, N_POINTS_P_AXIS+1)[1:end-1]

    coordinates_x = [x for x in x_range, y in y_range, z in z_range]
    coordinates_y = [y for x in x_range, y in y_range, z in z_range]
    coordinates_z = [z for x in x_range, y in y_range, z in z_range]

    wavenumbers_1d_x = fftfreq(N_POINTS_P_AXIS) .* N_POINTS_P_AXIS
    wavenumbers_1d_y = fftfreq(N_POINTS_P_AXIS) .* N_POINTS_P_AXIS
    wavenumbers_1d_z = fftfreq(N_POINTS_P_AXIS) .* N_POINTS_P_AXIS

    wavenumbers_x = [k_x for k_x in wavenumbers_1d_x, k_y in wavenumbers_1d_y, k_z in wavenumbers_1d_z]
    wavenumbers_y = [k_y for k_x in wavenumbers_1d_x, k_y in wavenumbers_1d_y, k_z in wavenumbers_1d_z]
    wavenumbers_z = [k_z for k_x in wavenumbers_1d_x, k_y in wavenumbers_1d_y, k_z in wavenumbers_1d_z]

    wavenumbers_norm = sqrt.(
        wavenumbers_x.^2
        +
        wavenumbers_y.^2
        +
        wavenumbers_z.^2
    )

    wavenumbers_norm[iszero.(wavenumbers_norm)] .= 1.0
    normalized_wavenumbers_x = wavenumbers_x ./ wavenumbers_norm
    normalized_wavenumbers_y = wavenumbers_y ./ wavenumbers_norm
    normalized_wavenumbers_z = wavenumbers_z ./ wavenumbers_norm

    # Taylor-Green vortex initial condition
    velocity_x =   sin.(coordinates_x) .* cos.(coordinates_y) .* cos.(coordinates_z)
    velocity_y = - cos.(coordinates_x) .* sin.(coordinates_y) .* cos.(coordinates_z)
    velocity_z =   zeros((N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))

    # Pre-Plan the FFT for additional performance
    fft_operator = plan_fft(velocity_x; flags=FFTW.MEASURE)
    
    velocity_x =   sin.(coordinates_x) .* cos.(coordinates_y) .* cos.(coordinates_z)

    # Pre-Allocatearrays
    velocity_x_fft = fft_operator * velocity_x
    velocity_y_fft = fft_operator * velocity_y
    velocity_z_fft = fft_operator * velocity_z

    curl_x = zeros(Float64, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))
    curl_y = zeros(Float64, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))
    curl_z = zeros(Float64, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))

    curl_x_squared = zeros(Float64, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))
    curl_y_squared = zeros(Float64, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))
    curl_z_squared = zeros(Float64, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))

    curl_magnitude = zeros(Float64, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))

    curl_x_fft = zeros(Complex{Float64}, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))
    curl_y_fft = zeros(Complex{Float64}, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))
    curl_z_fft = zeros(Complex{Float64}, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))

    convection_x = zeros(Float64, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))
    convection_y = zeros(Float64, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))
    convection_z = zeros(Float64, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))
    
    convection_x_fft = zeros(Complex{Float64}, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))
    convection_y_fft = zeros(Complex{Float64}, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))
    convection_z_fft = zeros(Complex{Float64}, (N_POINTS_P_AXIS, N_POINTS_P_AXIS, N_POINTS_P_AXIS))

    # De-Aliasing is necessary to stabilize the simulation
    k_max_dealias = 2.0/3.0 * (N_POINTS_P_AXIS//2 + 1)
    dealias = (
        (abs.(wavenumbers_x) .< k_max_dealias)
        .*
        (abs.(wavenumbers_y) .< k_max_dealias)
        .*
        (abs.(wavenumbers_z) .< k_max_dealias)
    )
    
    # Prepare the visualization
    makie_plot_figure_window = Figure()
    display(makie_plot_figure_window)

    makie_plot_axis = Axis3(
        makie_plot_figure_window[1, 1],
        perspectiveness = 0.3,
    )

    makie_plot_3d_contour = contour!(
        makie_plot_axis,
        x_range,
        y_range,
        z_range,
        curl_magnitude,
        alpha = 0.05,
        levels = range(0.2, 1.0, length=4),
    )
    println("Hit enter when plot is ready")
    readline()

    @showprogress "Simulating & Animating ..." for t in 1:N_TIME_STEPS
        # (1) Compute the Curl in Fourier Domain
        cross_product!(
            curl_x_fft,
            curl_y_fft,
            curl_z_fft,
            im .* wavenumbers_x,
            im .* wavenumbers_y,
            im .* wavenumbers_z,
            velocity_x_fft,
            velocity_y_fft,
            velocity_z_fft,
        )

        # (2) Transform curl to spatial domain
        curl_x .= real(fft_operator \ curl_x_fft)
        curl_y .= real(fft_operator \ curl_y_fft)
        curl_z .= real(fft_operator \ curl_z_fft)

        # (3) Compute "Convection" in spatial domain
        cross_product!(
            convection_x,
            convection_y,
            convection_z,
            velocity_x,
            velocity_y,
            velocity_z,
            curl_x,
            curl_y,
            curl_z,
        )

        # (4) Transform "Convection" to Fourier Domain
        convection_x_fft .= fft_operator * convection_x
        convection_y_fft .= fft_operator * convection_y
        convection_z_fft .= fft_operator * convection_z

        # (5) Dealiasing on the higher wavenumbers
        convection_x_fft .*= dealias
        convection_y_fft .*= dealias
        convection_z_fft .*= dealias

        # (6) Compute the Pseudo-Pressure by a Divergence in Fourier Domain
        pressure_fft = (
            normalized_wavenumbers_x .* convection_x_fft
            +
            normalized_wavenumbers_y .* convection_y_fft
            +
            normalized_wavenumbers_z .* convection_z_fft
        )

        # (7) Assemble the rhs to the ODE system in Fourier Domain
        rhs_x_fft = (
            convection_x_fft
            -
            KINEMATIC_VISCOSITY * wavenumbers_norm.^2 .* velocity_x_fft
            -
            normalized_wavenumbers_x .* pressure_fft
        )
        rhs_y_fft = (
            convection_y_fft
            -
            KINEMATIC_VISCOSITY * wavenumbers_norm.^2 .* velocity_y_fft
            -
            normalized_wavenumbers_y .* pressure_fft
        )
        rhs_z_fft = (
            convection_z_fft
            -
            KINEMATIC_VISCOSITY * wavenumbers_norm.^2 .* velocity_z_fft
            -
            normalized_wavenumbers_z .* pressure_fft
        )

        # (8) Euler Step Update
        velocity_x_fft .+= rhs_x_fft * TIME_STEP_LENGTH
        velocity_y_fft .+= rhs_y_fft * TIME_STEP_LENGTH
        velocity_z_fft .+= rhs_z_fft * TIME_STEP_LENGTH

        # (9) Transform the velocities back to spatial domain
        velocity_x .= real(fft_operator \ velocity_x_fft)
        velocity_y .= real(fft_operator \ velocity_y_fft)
        velocity_z .= real(fft_operator \ velocity_z_fft)

        # VISUALIZATION
        if t % PLOT_EVERY == 0
            curl_x_squared .= curl_x.^2
            curl_y_squared .= curl_y.^2
            curl_z_squared .= curl_z.^2
            curl_magnitude .= sqrt.(
                curl_x_squared
                +
                curl_y_squared
                +
                curl_z_squared
            )

            delete!(makie_plot_3d_contour.parent, makie_plot_3d_contour)

            makie_plot_3d_contour = contour!(
                makie_plot_axis,
                x_range,
                y_range,
                z_range,
                curl_magnitude,
                alpha = 0.01,
                levels = range(1.0, 10.0, length=5),
            )

            sleep(0.02)

        end

    end

end

main()
