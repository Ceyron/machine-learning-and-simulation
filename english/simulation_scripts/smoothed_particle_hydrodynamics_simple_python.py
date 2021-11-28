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
TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 2_500
DAMPING_COEFFICIENT = - 0.9
CONSTANT_FORCE = np.array([[0.0, -0.1]])

FIGURE_SIZE = (4, 6)
PLOT_EVERY_N_STEPS = 6
SCATTER_DOT_SIZE = 2000

DOMAIN_X_LIM = np.array([
    SMOOTHING_LENGTH,
    DOMAIN_WIDTH - SMOOTHING_LENGTH
])
DOMAIN_Y_LIM = np.array([
    SMOOTHING_LENGTH,
    DOMAIN_HEIGHT - SMOOTHING_LENGTH
])

NORMALIZATION_POLY_6 = (
    (
        PARTICLE_MASS * 315
    ) / (
        64 * np.pi * SMOOTHING_LENGTH**9
    )
)
NORMALIZATION_SPIKY = (
    -
    ( PARTICLE_MASS * 45 )
    /
    ( np.pi * SMOOTHING_LENGTH**6 )
)
NORMALIZATION_LAPLCE = (
    ( DYNAMIC_VISCOSITY * PARTICLE_MASS * 45 )
    /
    ( np.pi * SMOOTHING_LENGTH**6 )
)


def main():
    n_particles = 1
    
    positions = np.zeros((n_particles, 2))
    velocities = np.zeros_like(positions)
    forces = np.zeros_like(positions)
    
    plt.style.use("dark_background")
    plt.figure(figsize=FIGURE_SIZE, dpi=160)

    for iter in tqdm(range(N_TIME_STEPS)):
        if iter % 50 == 0 and n_particles < MAX_PARTICLES:
            new_pos = np.array([
                [10 + np.random.rand(), DOMAIN_Y_LIM[1]],
                [15 + np.random.rand(), DOMAIN_Y_LIM[1]],
                [20 + np.random.rand(), DOMAIN_Y_LIM[1]],
            ])
            new_vel = np.array([
                [-3.0, -15.0],
                [-3.0, -15.0],
                [-3.0, -15.0],
            ])
            n_particles += 3
            positions = np.concatenate((positions, new_pos), axis=0)
            velocities = np.concatenate((velocities, new_vel), axis=0)

        neighbor_ids, distances_graph = neighbors.BallTree(
            positions
        ).query_radius(
            positions,
            SMOOTHING_LENGTH,
            return_distance=True,
            sort_results=True
        )

        densities = np.zeros(n_particles)
        for i in range(n_particles):
            densities[i] = NORMALIZATION_POLY_6 * np.sum(
                (
                    SMOOTHING_LENGTH**2
                    -
                    distances_graph[i]**2
                )**3
            )

        pressure = ISOTROPIC_EXPONENT * (densities - BASE_DENSITY)
        
        forces = np.zeros_like(positions)

        # Drop the element itself
        neighbor_ids = [ np.delete(x, 0) for x in neighbor_ids]
        distances = [ np.delete(x, 0) for x in distances_graph]

        for i in range(n_particles):
            forces[i] += np.sum(
                - 
                (
                    positions[neighbor_ids[i]]
                    -
                    positions[i][np.newaxis, :]
                )
                /
                distances[i][:, np.newaxis]
                *
                (
                    pressure[i]
                    +
                    pressure[neighbor_ids[i]][:, np.newaxis]
                ) / (2 * densities[neighbor_ids[i]][:, np.newaxis])
                *
                NORMALIZATION_SPIKY
                *
                (
                    (
                        SMOOTHING_LENGTH
                        -
                        distances[i][:, np.newaxis]
                    )**2
                ),
                axis=0,
            )
            forces[i] += np.sum(
                DYNAMIC_VISCOSITY
                *
                PARTICLE_MASS
                *
                (
                    velocities[neighbor_ids[i]]
                    -
                    velocities[i][np.newaxis, :]
                ) / densities[neighbor_ids[i]][:, np.newaxis]
                *
                NORMALIZATION_LAPLCE
                *
                (
                    SMOOTHING_LENGTH
                    -
                    distances[i][:, np.newaxis]
                ),
                axis=0
            )
        
        forces += CONSTANT_FORCE

        # Euler Step
        velocities += TIME_STEP_LENGTH * forces / densities[:, np.newaxis]
        positions += TIME_STEP_LENGTH * velocities

        # Apply Boundaries
        OUT_OF_LEFT_BOUNDARY = positions[:, 0] < DOMAIN_X_LIM[0]
        OUT_OF_RIGHT_BOUNDARY = positions[:, 0] > DOMAIN_X_LIM[1]
        OUT_OF_BOTTOM_BOUNDARY = positions[:, 1] < DOMAIN_Y_LIM[0]
        OUT_OF_TOP_BOUNDARY = positions[:, 1] > DOMAIN_Y_LIM[1]


        velocities[OUT_OF_LEFT_BOUNDARY, 0]   *= DAMPING_COEFFICIENT
        positions [OUT_OF_LEFT_BOUNDARY, 0]    = DOMAIN_X_LIM[0]
        
        velocities[OUT_OF_RIGHT_BOUNDARY, 0]  *= DAMPING_COEFFICIENT
        positions [OUT_OF_RIGHT_BOUNDARY, 0]   = DOMAIN_X_LIM[1]

        velocities[OUT_OF_BOTTOM_BOUNDARY, 1] *= DAMPING_COEFFICIENT
        positions [OUT_OF_BOTTOM_BOUNDARY, 1]  = DOMAIN_Y_LIM[0]

        velocities[OUT_OF_TOP_BOUNDARY, 1]    *= DAMPING_COEFFICIENT
        positions [OUT_OF_TOP_BOUNDARY, 1]     = DOMAIN_Y_LIM[1]


        if iter % PLOT_EVERY_N_STEPS == 0:
            plt.scatter(positions[:, 0], positions[:, 1], s=SCATTER_DOT_SIZE, c=positions[:, 1], cmap="Wistia_r")
            plt.ylim(DOMAIN_Y_LIM)
            plt.xlim(DOMAIN_X_LIM)
            plt.xticks([],[])
            plt.yticks([],[])
            plt.tight_layout()
            plt.draw()
            plt.pause(0.0001)
            plt.clf()
    
    plt.show()



if __name__ == "__main__":
    main()