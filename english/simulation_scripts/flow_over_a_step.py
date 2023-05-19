"""
Solves the equation of interior fluid motion over a step.
This will simulate the inflow behavior into the pipe and the
boundary layer developing over time and space as well as the vortex
that is being created right after the step. The system of equations
is solved using a Staggered Grid, Finite Differences (almost Finite
Volume), explicit Euler time-stepping and a P2 pressure correction
scheme (very similar to the SIMPLE algorithm).


Momentum:           ∂u/∂t + (u ⋅ ∇) u = − 1/ρ ∇p + ν ∇²u + f

Incompressibility:  ∇ ⋅ u = 0


u:  Velocity (2d vector)
p:  Pressure
f:  Forcing (here =0)
ν:  Kinematic Viscosity
ρ:  Density (here =1)
t:  Time
∇:  Nabla operator (defining nonlinear convection, gradient and divergence)
∇²: Laplace Operator

--------

Scenario


                        wall: u=0, v=0
        +-----------------------------------------------+
 inflow |  -->      -->       -->        -->      -->   |
 u = 1  |                                               | outflow
 v = 0  |  -->      -->       -->        -->      -->   | ∂u/∂x = 0
        +--------------+                                | ∂v/∂x = 0
        |//// step ////|                                |
        +-----------------------------------------------+
                        wall: u=0, v=0                       

-> A rectangular domain
-> Top and bottom edge represent wall boundary conditions
-> In the bottom-left a step modifies the flow
-> The left edge above the step is an inflow with uniform profile
   for the u velocity
-> The entire right edge is an outflow out of the domain
-> The edges around the step are also wall boundary conditions
-> The flow is initialized with u=1 in the part of the domain, which
   is above an imaginary line over the step
-> The bottom right of the domain is initialized as zero in both
   velocity directions

--------

Expected Outcome

        +-----------------------------------------------+
        | --->     ->         ->  🔄      ->      ->     |
        | --->     ---->    ⋱                          |
        | --->     ->         --->       --->    --->   |
        +--------------+    ⋱                          |
        |//// step ////| 🔄   ->         ->      ->     |
        +-----------------------------------------------+


Above the step, the flow is developing the Hagen-Poiseuille
profile, which is characteristic for pipe flow.

After the step, the flow adjusts to be over the entire 
crosssection of the pipe and develops the parabolic profile
over the wider diameter.

Towards the end of the domain, the flow is fully developed
and adheres to the outflow boundary condition.

The 🔄 indicates spot, in which vortices develop depending
on the investigated Reynolds Number.

-------

The staggered grid with ghost cells

(see https://youtu.be/rV8tD2nQfkk &
https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/pipe_flow_with_inlet_and_outlet_python.py
for more details)

        |       |       |       |       |       |       |    
    •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
    ↑ - ❖---↑---❖---↑---❖---↑---❖---↑---❖---↑---❖---↑---❖ - ↑
        |       |       |       |       |       |       |    
    •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
        |       |       |       |       |       |       |    
    ↑ - ❖ - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - + - ↑ - ❖ - ↑
        |       |       |       |       |       |       |    
    •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
        |       |       |       |       |       |       |    
    ↑ - ❖ - ↑ - ❖ - ↑ - ❖ - ↑ - ❖ - ↑ - ❖ - ↑ - + - ↑ - ❖ - ↑
        |       |       |       |       |       |       |    
    •   →   •   →   •   →   •   →   • - →   •   →   •   →   •
                                        |       |       |    
    ↑ - +   ↑   +   ↑   +   ↑   +   ↑ - ❖ - ↑ - + - ↑ - ❖ - ↑
                                        |       |       |    
    •   →   •   →   •   →   •   →   •   →   •   →   •   →   •
                                        |       |       |    
    ↑ - 0   ↑   +   ↑   +   ↑   +   ↑ - ❖---↑---❖---↑---❖ - ↑
                                        |       |       |    
    •   →   •   →   •   →   •   →   •   →   •   →   •   →   •


"❖" denotes grid vertices that are on the boundary. Everything
outside of it, is called a ghost node. We need it to enforce the
boundary condition, especially for the step. We keep the degrees
of freedom inside the step for simplicity.

* u_velocities use (N_y + 1) by N_x nodes
* v_velocities use N_y by (N_x + 1) nodes
* pressure use (N_y + 1) by (N_x + 1) nodes

IMPORTANT: When taking derivatives make sure in which staggered
grid you are thinking.

-----

Solution Strategy:

Usage of a P2 pressure correction scheme (very similar to the SIMPLE
algorithm)

0. Initialization

    0.1 Initialize the u velocity 

    0.2 Initialize the v velocity 

    0.3 Initialize the p (=pressure) uniformly with zeros

1. Update the u velocities (+ Boundary Conditions)

    u ← u + dt ⋅ (− ∂p/∂x + ν ∇²u − ∂u²/∂x − v ∂u/∂y)

2. Update the v velocities (+ Boundary Conditions)

    v ← v + dt ⋅ (− ∂p/∂y + ν ∇²v − u ∂v/∂x − ∂v²/∂y)

3. Compute the divergence of the tentative velocity components

    d = ∂u/∂x + ∂v/∂y

4. Solve a Poisson problem for the pressure correction q
   (this problem has homogeneous Neumann BC everywhere except
   for the right edge of the domain (the outlet))

    solve   ∇²q = d / dt   for  q

5. Update the pressure

    p ← p + q

6. Update the velocities to be incompressible

    u ← u − dt ⋅ ∂q/∂x

    v ← v − dt ⋅ ∂q/∂y

7. Repeat time loop until steady-state is reached


For visualizations the velocities have to mapped to the
original vertex-centered grid.

The flow might require a correction at the outlet to ensure
continuity over the entire domain.

The density is assumed to be 1.0

-----

Notes on stability:

1. We are using an explicit diffusion treatment (FTCS) which
   has the stability condition:

   (ν dt) / (dx²) ≤ 1/2

2. We are using a central difference approximation for the
   convection term which is only stable if the diffusive
   transport is dominant (i.e., do not select the kinematic
   viscosity too low).

3. The Pressure Poisson (correction) problem is solved using
   Jacobi smoothing. This is sufficient for this simple
   application, but due to the fixed number of iterations
   does not ensure the residual is sufficiently small. That 
   could introduce local compressibility.
"""

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

N_POINTS_Y = 15
ASPECT_RATIO = 20
KINEMATIC_VISCOSITY = 0.01
TIME_STEP_LENGTH = 0.001
N_TIME_STEPS = 6000
PLOT_EVERY = 100

STEP_HEIGHT_POINTS = 7
STEP_WIDTH_POINTS = 60

N_PRESSURE_POISSON_ITERATIONS = 50

def main():
    cell_length = 1.0 / (N_POINTS_Y - 1)

    n_points_x = (N_POINTS_Y - 1) * ASPECT_RATIO + 1

    x_range = np.linspace(0.0, 1.0 * ASPECT_RATIO, n_points_x)
    y_range = np.linspace(0.0, 1.0, N_POINTS_Y)

    coordinates_x, coordinates_y = np.meshgrid(x_range, y_range)

    # Initial condition
    velocity_x_prev = np.ones((N_POINTS_Y + 1, n_points_x))
    velocity_x_prev[:(STEP_HEIGHT_POINTS + 1), :] = 0.0

    # Top Edge
    velocity_x_prev[-1, :] = - velocity_x_prev[-2, :]

    # Top Edge of the step
    velocity_x_prev[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS] =\
        - velocity_x_prev[(STEP_HEIGHT_POINTS + 1), 1:STEP_WIDTH_POINTS]
    
    # Right Edge of the step
    velocity_x_prev[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = 0.0

    # Bottom Edge of the domain
    velocity_x_prev[0, (STEP_WIDTH_POINTS + 1):-1] =\
        - velocity_x_prev[1, (STEP_WIDTH_POINTS + 1):-1]
    
    # Values inside of the step
    velocity_x_prev[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

    velocity_y_prev = np.zeros((N_POINTS_Y, n_points_x+1))

    pressure_prev = np.zeros((N_POINTS_Y+1, n_points_x+1))

    # Pre-Allocate some arrays
    velocity_x_tent = np.zeros_like(velocity_x_prev)
    velocity_x_next = np.zeros_like(velocity_x_prev)

    velocity_y_tent = np.zeros_like(velocity_y_prev)
    velocity_y_next = np.zeros_like(velocity_y_prev)

    plt.style.use("dark_background")
    plt.figure(figsize=(15, 6))

    for iter in tqdm(range(N_TIME_STEPS)):
        # Update interior of u velocity
        diffusion_x = KINEMATIC_VISCOSITY * (
            (
                +
                velocity_x_prev[1:-1, 2:  ]
                +
                velocity_x_prev[2:  , 1:-1]
                +
                velocity_x_prev[1:-1,  :-2]
                +
                velocity_x_prev[ :-2, 1:-1]
                - 4 *
                velocity_x_prev[1:-1, 1:-1]
            ) / (
                cell_length**2
            )
        )
        convection_x = (
            (
                velocity_x_prev[1:-1, 2:  ]**2
                -
                velocity_x_prev[1:-1,  :-2]**2
            ) / (
                2 * cell_length
            )
            +
            (
                velocity_y_prev[1:  , 1:-2]
                +
                velocity_y_prev[1:  , 2:-1]
                +
                velocity_y_prev[ :-1, 1:-2]
                +
                velocity_y_prev[ :-1, 2:-1]
            ) / 4
            *
            (
                velocity_x_prev[2:  , 1:-1]
                -
                velocity_x_prev[ :-2, 1:-1]
            ) / (
                2 * cell_length
            )
        )
        pressure_gradient_x = (
            (
                pressure_prev[1:-1, 2:-1]
                -
                pressure_prev[1:-1, 1:-2]
            ) / (
                cell_length
            )
        )

        velocity_x_tent[1:-1, 1:-1] = (
            velocity_x_prev[1:-1, 1:-1]
            +
            TIME_STEP_LENGTH
            *
            (
                -
                pressure_gradient_x
                +
                diffusion_x
                -
                convection_x
            )
        )

        # Apply BC

        # Inflow
        velocity_x_tent[(STEP_HEIGHT_POINTS + 1):-1, 0] = 1.0

        # Outflow
        inflow_mass_rate_tent = np.sum(velocity_x_tent[(STEP_HEIGHT_POINTS + 1):-1, 0])
        outflow_mass_rate_tent = np.sum(velocity_x_tent[1:-1, -2])
        velocity_x_tent[1:-1, -1] =\
            velocity_x_tent[1:-1, -2] * inflow_mass_rate_tent / outflow_mass_rate_tent
        
        # Top edge of the step
        velocity_x_tent[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS] =\
            - velocity_x_tent[STEP_HEIGHT_POINTS + 1, 1:STEP_WIDTH_POINTS]
        
        # Right edge of the step
        velocity_x_tent[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = 0.0

        # Bottom edge of the domain
        velocity_x_tent[0, (STEP_WIDTH_POINTS + 1):-1] =\
            - velocity_x_tent[1, (STEP_WIDTH_POINTS + 1):-1]
        
        # Top edge of the domain
        velocity_x_tent[-1, :] = - velocity_x_tent[-2, :]

        # Set all u-velocities to zero inside the step
        velocity_x_tent[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

        # Update interior of v velocity
        diffusion_y = KINEMATIC_VISCOSITY * (
            (
                +
                velocity_y_prev[1:-1, 2:  ]
                +
                velocity_y_prev[2:  , 1:-1]
                +
                velocity_y_prev[1:-1,  :-2]
                +
                velocity_y_prev[ :-2, 1:-1]
                -
                4 * velocity_y_prev[1:-1, 1:-1]
            ) / (
                cell_length**2
            )
        )
        convection_y = (
            (
                velocity_x_prev[2:-1, 1:  ]
                +
                velocity_x_prev[2:-1,  :-1]
                +
                velocity_x_prev[1:-2, 1:  ]
                +
                velocity_x_prev[1:-2,  :-1]
            ) / 4
            *
            (
                velocity_y_prev[1:-1, 2:  ]
                -
                velocity_y_prev[1:-1,  :-2]
            ) / (
                2 * cell_length
            )
            +
            (
                velocity_y_prev[2:  , 1:-1]**2
                -
                velocity_y_prev[ :-2, 1:-1]**2
            ) / (
                2 * cell_length
            )
        )
        pressure_gradient_y = (
            (
                pressure_prev[2:-1, 1:-1]
                -
                pressure_prev[1:-2, 1:-1]
            ) / (
                cell_length
            )
        )

        velocity_y_tent[1:-1, 1:-1] = (
            velocity_y_prev[1:-1, 1:-1]
            +
            TIME_STEP_LENGTH
            *
            (
                -
                pressure_gradient_y
                +
                diffusion_y
                -
                convection_y
            )
        )

        # Apply BC

        # Inflow
        velocity_y_tent[(STEP_HEIGHT_POINTS + 1):-1, 0] =\
            - velocity_y_tent[(STEP_HEIGHT_POINTS + 1):-1, 1]
        
        # Outflow
        velocity_y_tent[1:-1, -1] = velocity_y_tent[1:-1, -2]

        # Top edge of the step
        velocity_y_tent[STEP_HEIGHT_POINTS, 1:(STEP_WIDTH_POINTS + 1)] = 0.0

        # Right edge of the step
        velocity_y_tent[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] =\
            - velocity_y_tent[1:(STEP_HEIGHT_POINTS + 1), (STEP_WIDTH_POINTS + 1)]
        
        # Bottom edge of the domain
        velocity_y_tent[0, (STEP_WIDTH_POINTS + 1):] = 0.0

        # Top edge of the domain
        velocity_y_tent[-1, :] = 0.0

        # Set all v-velocities to zero inside of the edge
        velocity_y_tent[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

        # Compute the divergence as it will be the rhs of the pressure poisson
        # problem
        divergence = (
            (
                velocity_x_tent[1:-1, 1:  ]
                -
                velocity_x_tent[1:-1,  :-1]
            ) / (
                cell_length
            )
            +
            (
                velocity_y_tent[1:  , 1:-1]
                -
                velocity_y_tent[ :-1, 1:-1]
            ) / (
                cell_length
            )
        )
        pressure_poisson_rhs = divergence / TIME_STEP_LENGTH

        # Solve the pressure correction poisson problem
        pressure_correction_prev = np.zeros_like(pressure_prev)
        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            pressure_correction_next = np.zeros_like(pressure_correction_prev)
            pressure_correction_next[1:-1, 1:-1] = 1/4 * (
                +
                pressure_correction_prev[1:-1, 2:  ]
                +
                pressure_correction_prev[2:  , 1:-1]
                +
                pressure_correction_prev[1:-1,  :-2]
                +
                pressure_correction_prev[ :-2, 1:-1]
                -
                cell_length**2
                *
                pressure_poisson_rhs
            )

            # Apply pressure BC: Homogeneous Neumann everywhere except for the
            # right where is a homogeneous Dirichlet

            # Inflow
            pressure_correction_next[(STEP_HEIGHT_POINTS + 1):-1, 0] =\
                pressure_correction_next[(STEP_HEIGHT_POINTS + 1):-1, 1]
            
            # Outflow
            pressure_correction_next[1:-1, -1] =\
                - pressure_correction_next[1:-1, -2]
            
            # Top edge of the step
            pressure_correction_next[STEP_HEIGHT_POINTS, 1:(STEP_WIDTH_POINTS + 1)] =\
                pressure_correction_next[(STEP_HEIGHT_POINTS + 1), 1:(STEP_WIDTH_POINTS + 1)]
            
            # Right edge of the step
            pressure_correction_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] =\
                pressure_correction_next[1:(STEP_HEIGHT_POINTS + 1), (STEP_WIDTH_POINTS + 1)]
            
            # Bottom edge of the domain
            pressure_correction_next[0, (STEP_WIDTH_POINTS + 1):-1] =\
                pressure_correction_next[1, (STEP_WIDTH_POINTS + 1):-1]
            
            # Top edge of the domain
            pressure_correction_next[-1, :] = pressure_correction_next[-2, :]

            # Set all pressure (correction) values inside the step to zero
            pressure_correction_next[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0

            # Advance in smoothing
            pressure_correction_prev = pressure_correction_next
        
        # Update the pressure
        pressure_next = pressure_prev + pressure_correction_next

        # Correct the velocities to be incompressible
        pressure_correction_gradient_x = (
            (
                pressure_correction_next[1:-1, 2:-1]
                -
                pressure_correction_next[1:-1, 1:-2]
            ) / (
                cell_length
            )
        )

        velocity_x_next[1:-1, 1:-1] = (
            velocity_x_tent[1:-1, 1:-1]
            -
            TIME_STEP_LENGTH
            *
            pressure_correction_gradient_x
        )

        pressure_correction_gradient_y = (
            (
                pressure_correction_next[2:-1, 1:-1]
                -
                pressure_correction_next[1:-2, 1:-1]
            ) / (
                cell_length
            )
        )

        velocity_y_next[1:-1, 1:-1] = (
            velocity_y_tent[1:-1, 1:-1]
            -
            TIME_STEP_LENGTH
            *
            pressure_correction_gradient_y
        )

        # Again enforce BC
        
        # Inflow
        velocity_x_next[(STEP_HEIGHT_POINTS + 1):-1, 0] = 1.0

        # Outflow
        inflow_mass_rate_next = np.sum(velocity_x_next[(STEP_HEIGHT_POINTS + 1):-1, 0])
        outflow_mass_rate_next = np.sum(velocity_x_next[1:-1, -2])
        velocity_x_next[1:-1, -1] =\
            velocity_x_next[1:-1, -2] * inflow_mass_rate_next / outflow_mass_rate_next
        
        # Top edge of the step
        velocity_x_next[STEP_HEIGHT_POINTS, 1:STEP_WIDTH_POINTS] =\
            - velocity_x_next[STEP_HEIGHT_POINTS + 1, 1:STEP_WIDTH_POINTS]
        
        # Right edge of the step
        velocity_x_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] = 0.0

        # Bottom edge of the domain
        velocity_x_next[0, (STEP_WIDTH_POINTS + 1):-1] =\
            - velocity_x_next[1, (STEP_WIDTH_POINTS + 1):-1]
        
        # Top edge of the domain
        velocity_x_next[-1, :] = - velocity_x_next[-2, :]

        # Set all u-velocities to zero inside the step
        velocity_x_next[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0
        
        # Inflow
        velocity_y_next[(STEP_HEIGHT_POINTS + 1):-1, 0] =\
            - velocity_y_next[(STEP_HEIGHT_POINTS + 1):-1, 1]
        
        # Outflow
        velocity_y_next[1:-1, -1] = velocity_y_next[1:-1, -2]

        # Top edge of the step
        velocity_y_next[STEP_HEIGHT_POINTS, 1:(STEP_WIDTH_POINTS + 1)] = 0.0

        # Right edge of the step
        velocity_y_next[1:(STEP_HEIGHT_POINTS + 1), STEP_WIDTH_POINTS] =\
            - velocity_y_next[1:(STEP_HEIGHT_POINTS + 1), (STEP_WIDTH_POINTS + 1)]
        
        # Bottom edge of the domain
        velocity_y_next[0, (STEP_WIDTH_POINTS + 1):] = 0.0

        # Top edge of the domain
        velocity_y_next[-1, :] = 0.0

        # Set all v-velocities to zero inside of the edge
        velocity_y_next[:STEP_HEIGHT_POINTS, :STEP_WIDTH_POINTS] = 0.0


        # Advance in time
        velocity_x_prev = velocity_x_next
        velocity_y_prev = velocity_y_next
        pressure_prev = pressure_next

        # inflow_mass_rate_next = np.sum(velocity_x_next[1:-1, 0])
        # outflow_mass_rate_next = np.sum(velocity_x_next[1:-1, -1])
        # print(f"Inflow: {inflow_mass_rate_next}")
        # print(f"Outflow: {outflow_mass_rate_next}")
        # print()

        # Visualization
        if iter % PLOT_EVERY == 0:
            velocity_x_vertex_centered = (
                (
                    velocity_x_next[1:  , :]
                    +
                    velocity_x_next[ :-1, :]
                ) / 2
            )
            velocity_y_vertex_centered = (
                (
                    velocity_y_next[:, 1:  ]
                    +
                    velocity_y_next[:,  :-1]
                ) / 2
            )

            velocity_x_vertex_centered[:(STEP_HEIGHT_POINTS + 1),:(STEP_WIDTH_POINTS + 1)] = 0.0
            velocity_y_vertex_centered[:(STEP_HEIGHT_POINTS + 1),:(STEP_WIDTH_POINTS + 1)] = 0.0

            plt.contourf(
                coordinates_x,
                coordinates_y,
                velocity_x_vertex_centered,
                levels=10,
                cmap=cmr.amber,
                vmin=-1.5,
                vmax=1.5,
            )
            plt.colorbar()

            plt.quiver(
                coordinates_x[:, ::6],
                coordinates_y[:, ::6],
                velocity_x_vertex_centered[:, ::6],
                velocity_y_vertex_centered[:, ::6],
                alpha=0.4,
            )

            plt.plot(
                5 * cell_length + velocity_x_vertex_centered[:, 5],
                coordinates_y[:, 5], 
                color="black",
                linewidth=3,
            )
            plt.plot(
                40 * cell_length + velocity_x_vertex_centered[:, 40],
                coordinates_y[:, 40], 
                color="black",
                linewidth=3,
            )
            plt.plot(
                80 * cell_length + velocity_x_vertex_centered[:, 80],
                coordinates_y[:, 80], 
                color="black",
                linewidth=3,
            )
            plt.plot(
                180 * cell_length + velocity_x_vertex_centered[:, 180],
                coordinates_y[:, 180], 
                color="black",
                linewidth=3,
            )

            plt.draw()
            plt.pause(0.15)
            plt.clf()
    
    plt.figure()
    plt.streamplot(
        coordinates_x,
        coordinates_y,
        velocity_x_vertex_centered,
        velocity_y_vertex_centered,
    )

    plt.show()

if __name__ == "__main__":
    main()
