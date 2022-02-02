"""
Solves the equation of fluid motion in a pressure driven pipe flow with periodic
boundary conditions on both ends. If started with an initial uniform profile,
the Hagen-Poiseuille parabula profile will develop overe time.

Momentum:           ∂u/∂t + (u ⋅ ∇) u = − 1/ρ ∇p + ν ∇²u + f

Incompressibility:  ∇ ⋅ u = 0


u:  Velocity (2d vector)
p:  Pressure
f:  Forcing (here =0)
ν:  Kinematic Viscosity
ρ:  Density
t:  Time
∇:  Nabla operator (defining nonlinear convection, gradient and divergence)
∇²: Laplace Operator

-------

Scenario

                  constant pressure gradient
                    <<<<<<<<<<------------

                        wall: u=0, v=0
   p    +-----------------------------------------------+   p
   e    |  -->      -->       -->        -->      -->   |   e
   r    |                                               |   r
   i    |  -->      -->       -->        -->      -->   |   i
   o    |                                               |   o
   d    |  -->      -->       -->        -->      -->   |   d
   i    +-----------------------------------------------+   i
   c                    wall: u=0, v=0                      c

-> A rectangular domain (think of a slice from a pipe with
   circular crossection alongside the longitudal axis)
-> The left and right edge are connected by periodicity,
   representing an infinitely long domain in x axis
-> Top and bottom edge represent wall boundary conditions
-> A constant pressure gradient (in x direction) acts on the
   entire domain

--------

Expected Outcome

After some time the parabula flow profile will develop due
to a boundary layer developing by the viscous effects of
the fluid. This does not affect the total mass flux which
will conserve continuity. [NOTE: This is not fully correct.
The attained Hagen-Poisseuille profile is indeed independent
of the initial profile and just depends on the strength of
the pressure gradient.]

        +-----------------------------------------------+
        |   ->       ->       ->       ->       ->      |
        |   --->     --->     --->     --->     --->    |
        |   ---->    ---->    ---->    ---->    ---->   |
        |   --->     --->     --->     --->     --->    |
        |   ->       ->       ->       ->       ->      |
        +-----------------------------------------------+

-------

Solution strategy:

We do not have to consider the velocity in y
direction (aka v-velocity) since it will be constant 0 throughout
the computational domain. (and therefore also its derivatives)

Discretization of the u-momentum equation

    ∂u/∂t + u ∂u/∂x + v ∂u/∂y = - ∂p/∂x + ν ∇²u
                |     |             |       |
                |     ↓             ↓       |
                |    = 0        constant    |
                |                           |
                ↓                           ↓
        central differences         five-point stencil


0. Instantiate the u-solution field with ones except for
   the top and bottom boundary

1. Compute convection by periodic central difference

    u ∂u/∂x ≈ u[i, j] ⋅ (u[i, (j+1)%N] − u[i, (j−1)%N]) / (2 dx)

2. Compute diffusion by periodic five-point stencil

    ν ∇²u ≈ ν (
        + u[i, (j+1)%N]
        + u[(i+1)%N, j]
        + u[i, (j−1)%N]
        + u[(i−1)%N, j]
        − 4 ⋅ u[i, j]
        ) / (dx²)

3. Advance to next step by explicit Euler step

    u ← u + dt ⋅ (− ∂p/∂x + ν ∇²u − u ∂u/∂x)

4. Enfore the wall boundary condition by setting the u velocity
   at the top and bottom boundary to zero

5. Repeat from (1.) until a steady state is reached.


No pressure correction equation has to be solved since the
pressure gradient is prescribed constant throughout the domain.

------

Notes on stability:

1. We are using an explicit diffusion treatment (FTCS) which
   has the stability condition:

   (ν dt) / (dx²) ≤ 1/2

2. We are using a central difference approximation for the
   convection term which is only stable if the diffusive
   transport is dominant (i.e., do not select the kinematic
   viscosity too low).

A staggered grid is not necessary, since our pressure gradient
is constant. Hence, we do not have to solve a pressure poisson
equation.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N_POINTS = 11
KINEMATIC_VISCOSITY = 0.01
TIME_STEP_LENGTH = 0.2
N_TIME_STEPS = 100

PRESSURE_GRADIENT = np.array([-1.0, 0.0])

def main():
    element_length = 1.0 / (N_POINTS - 1)

    x_range = np.linspace(0.0, 1.0, N_POINTS)
    y_range = np.linspace(0.0, 1.0, N_POINTS)

    coordinates_x, coordinates_y = np.meshgrid(x_range, y_range)

    def central_difference_x_periodic(field):
        diff = (
            (
                np.roll(field, shift=1, axis=1)
                -
                np.roll(field, shift=-1, axis=1)
            ) / (
                2 * element_length
            )
        )
        return diff
    
    def laplace_periodic(field):
        diff = (
            (
                np.roll(field, shift=1, axis=1)
                +
                np.roll(field, shift=1, axis=0)
                +
                np.roll(field, shift=-1, axis=1)
                +
                np.roll(field, shift=-1, axis=0)
                -
                4 * field
            ) / (
                element_length**2
            )
        )
        return diff
    
    # Define the initial condition
    velocity_x_prev = np.ones((N_POINTS, N_POINTS))
    velocity_x_prev[0, :] = 0.0
    velocity_x_prev[-1, :] = 0.0

    plt.style.use("dark_background")
    
    for iter in tqdm(range(N_TIME_STEPS)):
        convection_x = (
            velocity_x_prev
            *
            central_difference_x_periodic(velocity_x_prev)
        )
        diffusion_x = (
            KINEMATIC_VISCOSITY
            *
            laplace_periodic(velocity_x_prev)
        )
        velocity_x_next = (
            velocity_x_prev
            +
            TIME_STEP_LENGTH
            *
            (
                -
                PRESSURE_GRADIENT[0]
                +
                diffusion_x
                -
                convection_x
            )
        )

        velocity_x_next[0, :] = 0.0
        velocity_x_next[-1, :] = 0.0

        # Advance in time
        velocity_x_prev = velocity_x_next

        # Visualization
        plt.contourf(coordinates_x, coordinates_y, velocity_x_next, levels=50)
        plt.colorbar()
        plt.quiver(coordinates_x, coordinates_y, velocity_x_next, np.zeros_like(velocity_x_next))
        plt.xlabel("Position alongside the pipe")
        plt.ylabel("Position perpendicular to the pipe axis")

        plt.twiny()
        plt.plot(velocity_x_next[:, 1], coordinates_y[:, 1], color="white")
        plt.xlabel("Flow Velocity")

        plt.draw()
        plt.pause(0.05)
        plt.clf()
    
    plt.show()


if __name__ == "__main__":
    main()
