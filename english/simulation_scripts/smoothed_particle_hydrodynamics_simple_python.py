"""
Simulates the free-surface flow of "pouring a beer" using Smoothed Particle
Hydrodynamics (SPH). Uses the super-simple approch of Matthias Müller:
https://matthias-research.github.io/pages/publications/sca03.pdf

Simulates the Navier-Stokes Momentum in Lagrangian Form

    ρ Du/Dt = − ∇p + μ ∇²u + g

u : velocity
ρ : density
p : pressure
μ : dynamics viscosity
g : gravity

Du/Dt : Lagrangian temporal derivative (=Material Derivative)
∇p    : Pressure Gradient
∇²u   : Velocity Laplacian (=collection of second derivatives)

IMPORTANT: The simulated fluid is not incompressible.

-------

Scenario

        +-------------------------+
        |      /  /  /            |
        |     /  /  /             |
        |    ↙  ↙  ↙              |
        |                         |
        |                         |
        |                         |
        |                         |
        |                         |
        |                         |
        |                         |
        |                         |
        |                         |
        |                         |
        |                         |
        +-------------------------+

-> A vertical Rectangular domain with all walls

-> New Particles enter the domain slightly below the top with a velocity
directed to the bottom left (like pouring in the beer)

----------

Solution Strategy:

Discretize the fluid by N particles (i=0, 1, ..., N-1) that smooth
their properties radially with some smoothing kernels. Then the Momentum
Equation discretizes to

Duᵢ/Dt = Pᵢ + Vᵢ + G

with

uᵢ : The velocity of the i-th smoothed particle
Pᵢ : The pressure forces acting on the i-th smoothed particle
Vᵢ : The viscosity forces acting on the i-th smoothed particle
G  : The gravity forces (acting equally on all particles)

This yields a set of N ODEs each for the two velocity components (in case
of 2D) of the particles. These can be solved using a (simpletic) integrator
to advance the position of the particles.


Let xᵢ be the 2D position of each smoothed particle.

Let L be the smoothing length of each smoothed particle.

Let M be the mass of each smoothed particle.

------

Algorithm

(for details on the chosen smoothing kernels, see the paper mentioned above)

1. Compute the rhs for each particle Fᵢ

    1.1 Compute the distances between all particle positions

        dᵢⱼ = || xᵢ − xⱼ ||₂

    1.2 Compute the density at each particle's position

        ρᵢ = (315 M) / (64 π L⁹) ∑ⱼ (L² − dᵢⱼ²)³
    
    1.3 Compute the pressure at each particle's position (κ is the isentropic
        exponent, ρ₀ is a base density)

        pᵢ = κ * (ρ − ρ₀)

    1.4 Compute the pressure force of each particle

        Pᵢ = (− (45 M) / (π L⁶)) ∑ⱼ − (xⱼ − xᵢ) / (dᵢⱼ) (pⱼ + pᵢ) / (2 ρⱼ) (L − dᵢⱼ)²
    
    1.5 Compute the viscosity force of each particle

        Vᵢ = (45 μ M) / (π L⁶) ∑ⱼ (uⱼ − uᵢ) / (ρⱼ) (L − dᵢⱼ)
    
    1.6 Add up the RHS

        Fᵢ = Pᵢ + Vᵢ + G

2. Integrate the Ordinary Differential Equation  "ρ Duᵢ/Dt = Fᵢ" with a
   Δt timestep

    2.1 Update the particles' velocities

        uᵢ ← uᵢ + Δt Fᵢ / ρᵢ
    
    2.2 Update the particles' positions

        xᵢ ← xᵢ + Δt uᵢ

3. Enforce the wall Boundary Conditions. If a particle leaves the
   domain then:

    3.1 Set its position to the Boundary

    3.2 Inverse its velocity component perpendicular to the wall

    3.3 Multiply the velocity component perpendicular to the
        wall with a damping factor

-------

Computational Considerations.

1. The steps on computing density, pressure force and viscosity force
   involve the computation of the various smoothing kernels. Those
   always consist of a constant part that is due to the normalization
   which can be precomputed.

2. When applying summations in the distance calculations and when
   applying the smothing kernels in density, pressure force and
   viscosity force calculations only OTHER PARTICLES IN THE
   SMOOTHING LENGTH OF THE CONSIDERED PARTICLE ARE RELEVANT. Hence,
   we can use efficient neighbor computation routines.

-------

Take care that the ODE integration can become instable when using too
large time steps.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from tqdm import tqdm

MAX_PARTICLES = 125
DOMAIN_WIDTH = 40
DOMAIN_HEIGHT = 80

PARTICLE_MASS = 1
ISOTROPIC_EXPONENT = 20
BASE_DENSITY = 1
SMOOTHING_LENGTH = 5
DYNAMIC_VISCOSITY = 0.5
DAMPING_COEFFICIENT = - 0.9
CONSTANT_FORCE = np.array([[0.0, -0.1]])

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 2_500
ADD_PARTICLES_EVERY = 50

FIGURE_SIZE = (4, 6)
PLOT_EVERY = 6
SCATTER_DOT_SIZE = 2_000

DOMAIN_X_LIM = np.array([
    SMOOTHING_LENGTH,
    DOMAIN_WIDTH - SMOOTHING_LENGTH,
])
DOMAIN_Y_LIM = np.array([
    SMOOTHING_LENGTH,
    DOMAIN_HEIGHT - SMOOTHING_LENGTH,
])

NORMALIZATION_DENSITY = (
    (
        315 * PARTICLE_MASS
    ) / (
        64 * np.pi * SMOOTHING_LENGTH**9
    )
)
NORMALIZATION_PRESSURE_FORCE = (
    -
    (
        45 * PARTICLE_MASS
    ) / (
        np.pi * SMOOTHING_LENGTH**6
    )
)
NORMALIZATION_VISCOUS_FORCE = (
    (
        45 * DYNAMIC_VISCOSITY * PARTICLE_MASS
    ) / (
        np.pi * SMOOTHING_LENGTH**6
    )
)

def main():
    n_particles = 1

    positions = np.zeros((n_particles, 2))
    velocities = np.zeros_like(positions)
    forces = np.zeros_like(positions)

    plt.style.use("dark_background")
    plt.figure(figsize=FIGURE_SIZE, dpi=160)

    for iter in tqdm(range(N_TIME_STEPS)):
        if iter % ADD_PARTICLES_EVERY == 0 and n_particles < MAX_PARTICLES:
            new_positions = np.array([
                [10 + np.random.rand(), DOMAIN_Y_LIM[1]],
                [15 + np.random.rand(), DOMAIN_Y_LIM[1]],
                [20 + np.random.rand(), DOMAIN_Y_LIM[1]],
            ])
            
            new_velocities = np.array([
                [-3.0, -15.0],
                [-3.0, -15.0],
                [-3.0, -15.0],
            ])

            n_particles += 3

            positions = np.concatenate((positions, new_positions), axis=0)
            velocities = np.concatenate((velocities, new_velocities), axis=0)
        
        neighbor_ids, distances = neighbors.KDTree(
            positions,
        ).query_radius(
            positions,
            SMOOTHING_LENGTH,
            return_distance=True,
            sort_results=True,
        )

        densities = np.zeros(n_particles)

        for i in range(n_particles):
            for j_in_list, j in enumerate(neighbor_ids[i]):
                densities[i] += NORMALIZATION_DENSITY * (
                    SMOOTHING_LENGTH**2
                    -
                    distances[i][j_in_list]**2
                )**3
        
        pressures = ISOTROPIC_EXPONENT * (densities - BASE_DENSITY)

        forces = np.zeros_like(positions)

        # Drop the element itself
        neighbor_ids = [ np.delete(x, 0) for x in neighbor_ids]
        distances = [ np.delete(x, 0) for x in distances]

        for i in range(n_particles):
            for j_in_list, j in enumerate(neighbor_ids[i]):
                # Pressure force
                forces[i] += NORMALIZATION_PRESSURE_FORCE * (
                    -
                    (
                        positions[j]
                        -
                        positions[i]
                    ) / distances[i][j_in_list]
                    *
                    (
                        pressures[j]
                        +
                        pressures[i]
                    ) / (2 * densities[j])
                    *
                    (
                        SMOOTHING_LENGTH
                        -
                        distances[i][j_in_list]
                    )**2
                )

                # Viscous force
                forces[i] += NORMALIZATION_VISCOUS_FORCE * (
                    (
                        velocities[j]
                        -
                        velocities[i]
                    ) / densities[j]
                    *
                    (
                        SMOOTHING_LENGTH
                        -
                        distances[i][j_in_list]
                    )
                )
        
        # Force due to gravity
        forces += CONSTANT_FORCE

        ### There is an error in the video, in that the gravity is divided by the density which
        ### wrongly scales it. Uncommenting the below, and commenting the call above will fix
        ### that but, also slightly changes the visual result of the simulation.
        # forces += CONSTANT_FORCE * densities[:, np.newaxis]
        

        # Euler Step
        velocities = velocities + TIME_STEP_LENGTH * forces / densities[:, np.newaxis]
        positions = positions + TIME_STEP_LENGTH * velocities

        # Enfore Boundary Conditions
        out_of_left_boundary = positions[:, 0] < DOMAIN_X_LIM[0]
        out_of_right_boundary = positions[:, 0] > DOMAIN_X_LIM[1]
        out_of_bottom_boundary = positions[:, 1] < DOMAIN_Y_LIM[0]
        out_of_top_boundary = positions[:, 1] > DOMAIN_Y_LIM[1]

        velocities[out_of_left_boundary, 0]     *= DAMPING_COEFFICIENT
        positions [out_of_left_boundary, 0]      = DOMAIN_X_LIM[0]

        velocities[out_of_right_boundary, 0]    *= DAMPING_COEFFICIENT
        positions [out_of_right_boundary, 0]     = DOMAIN_X_LIM[1]

        velocities[out_of_bottom_boundary, 1]   *= DAMPING_COEFFICIENT
        positions [out_of_bottom_boundary, 1]    = DOMAIN_Y_LIM[0]

        velocities[out_of_top_boundary, 1]      *= DAMPING_COEFFICIENT
        positions [out_of_top_boundary, 1]       = DOMAIN_Y_LIM[1]

        if iter % PLOT_EVERY == 0:
            plt.scatter(
                positions[:, 0],
                positions[:, 1],
                s=SCATTER_DOT_SIZE,
                c=positions[:, 1],
                cmap="Wistia_r",
            )
            plt.ylim(DOMAIN_Y_LIM)
            plt.xlim(DOMAIN_X_LIM)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
            plt.draw()
            plt.pause(0.0001)
            plt.clf()


if __name__ == "__main__":
    main()
