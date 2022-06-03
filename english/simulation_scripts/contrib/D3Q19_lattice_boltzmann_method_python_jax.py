# -*- coding: utf-8 -*-
"""
Adjustment from the 2D version from Machine Learning & Simulation code and video:
    https://www.youtube.com/watch?v=ZUXmO4hu-20&list=LL&index=1&ab_channel=MachineLearning%26Simulation
    https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/lattice_boltzmann_method_python_jax.py

by Bart Davids. Originally made in Google Colab:
https://colab.research.google.com/drive/1F3EH9_2N3lkEpgQXOScR3lcQ6oqCARPk?usp=sharing

Additional notes and figures for clarification can be found there.

"""

# Import dependancies
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

if __name__ == '__main__':
    
    # Enable 64bit JAX
    jax.config.update("jax_enable_x64", True)
    
    # Radius of the cylinder
    radius = 5.5
    
    # Dimensions of domain
    ny = 50
    nz = 60
    nx = 300
    
    KINEMATIC_VISCOSITY = 0.0025                
    HORIZONTAL_INFLOW_VELOCITY = 0.04           
    
    reynolds_number = (HORIZONTAL_INFLOW_VELOCITY * radius) / KINEMATIC_VISCOSITY                            
    
    RELAXATION_OMEGA = (1.0 / (3.0 * KINEMATIC_VISCOSITY + 0.5))
    
    PLOT_EVERY_N_STEPS = 100
    SKIP_FIRS_N_ITERATIONS = 5000  
    N_ITERATIONS = 20000
    print('Reynolds number:', reynolds_number)
    
    # Define a mesh for the obstacle mask
    x = jnp.arange(nx)
    y = jnp.arange(ny)
    z = jnp.arange(nz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    
    cylinder = jnp.sqrt((X - nx//5)**2 + (Y - ny//2)**2)
    obstacle_mask = cylinder < radius
    
    # Show topview of the cylinder:
    plt.imshow(obstacle_mask[:, :, nz//2].T)
    plt.show()
    
    # Front view:
    plt.imshow(obstacle_mask[nx//5, :, :].T)
    plt.show()
    
    # Side View:
    plt.imshow(obstacle_mask[:, ny//2, :].T)
    plt.show()
    
    def get_density(discrete_velocities):
        density = jnp.sum(discrete_velocities, axis=-1)
        return density
    
    def get_macroscopic_velocities(discrete_velocities, density):
        return jnp.einsum("NMLQ,dQ->NMLd", discrete_velocities, LATTICE_VELOCITIES) / density[..., jnp.newaxis]
    
    def get_equilibrium_discrete_velocities(macroscopic_velocities, density):
        projected_discrete_velocities = jnp.einsum("dQ,NMLd->NMLQ", LATTICE_VELOCITIES, macroscopic_velocities)
        macroscopic_velocity_magnitude = jnp.linalg.norm(macroscopic_velocities, axis=-1, ord=2)
        equilibrium_discrete_velocities = (density[..., jnp.newaxis] * LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, jnp.newaxis, :] *
            (1 + 3 * projected_discrete_velocities + 9/2 * projected_discrete_velocities**2 -
            3/2 * macroscopic_velocity_magnitude[..., jnp.newaxis]**2))    
        return equilibrium_discrete_velocities
    
    N_DISCRETE_VELOCITIES = 19
    
    # 3D lattice velocities and numbering used as in: 
    # https://www.researchgate.net/publication/290158292_An_introduction_to_Lattice-Boltzmann_methods
    LATTICE_INDICES =          jnp.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18])
    LATICE_VELOCITIES_X =      jnp.array([ 0, 1, 0,-1, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1, 0, 0, 0, 0])
    LATICE_VELOCITIES_Y =      jnp.array([ 0, 0, 1, 0,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1,-1, 1])
    LATICE_VELOCITIES_Z =      jnp.array([ 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1])
    
    OPPOSITE_LATTICE_INDICES = jnp.array([ 0, 3, 4, 1, 2, 6, 5, 9,10, 7, 8,13,14,11,12,17,18,15,16])
    
    LATTICE_VELOCITIES = jnp.array([LATICE_VELOCITIES_X,
                                    LATICE_VELOCITIES_Y,
                                    LATICE_VELOCITIES_Z])
    
    LATTICE_WEIGHTS = jnp.array([# rest particle
                                 1/3, 
                                 
                                 # face-connected neighbors
                                 1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
                                 
                                 # edge-connected neighbors
                                 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36])
    
    # Velocity directions/planes
    RIGHT_VELOCITIES = jnp.array([1, 7, 10, 11, 14])             # LATICE_VELOCITIES_X = 1
    LEFT_VELOCITIES = jnp.array([3, 8, 9, 12, 13])               # LATICE_VELOCITIES_X =-1
    YZ_VELOCITIES = jnp.array([0, 2, 4, 5, 6, 15, 16, 17, 18])   # LATICE_VELOCITIES_X = 0
    
    
    VELOCITY_PROFILE = jnp.zeros((nx, ny, nz, 3))
    VELOCITY_PROFILE = VELOCITY_PROFILE.at[:, :, :, 0].set(HORIZONTAL_INFLOW_VELOCITY)
    discrete_velocities_prev = get_equilibrium_discrete_velocities(VELOCITY_PROFILE, 
                                                                   jnp.ones((nx, ny, nz)))
    
    @jax.jit
    def update(discrete_velocities_prev):
        # (1) Prescribe the outflow BC on the right boundary. Flow can go out, but not back in.
        discrete_velocities_prev = discrete_velocities_prev.at[-1, :, :, LEFT_VELOCITIES].set(discrete_velocities_prev[-2, :, :, LEFT_VELOCITIES])
    
        # (2) Determine macroscopic velocities
        density_prev = get_density(discrete_velocities_prev)
        macroscopic_velocities_prev = get_macroscopic_velocities(
            discrete_velocities_prev,
            density_prev)
    
        # (3) Prescribe Inflow Dirichlet BC using Zou/He scheme in 3D: 
        # https://arxiv.org/pdf/0811.4593.pdf
        # https://terpconnect.umd.edu/~aydilek/papers/LB.pdf
        macroscopic_velocities_prev = macroscopic_velocities_prev.at[0, 1:-1, 1:-1, :].set(VELOCITY_PROFILE[0, 1:-1, 1:-1, :])
        lateral_densities = get_density(jnp.transpose(discrete_velocities_prev[0, :, :, YZ_VELOCITIES], axes = (1, 2, 0)))
        left_densities = get_density(jnp.transpose(discrete_velocities_prev[0, :, :, LEFT_VELOCITIES], axes = (1, 2, 0)))
        density_prev = density_prev.at[0, :, :].set((lateral_densities + 2 * left_densities) / 
                                                    (1 - macroscopic_velocities_prev[0, :, :, 0]))
    
        # (4) Compute discrete Equilibria velocities
        equilibrium_discrete_velocities = get_equilibrium_discrete_velocities(
            macroscopic_velocities_prev,
            density_prev)
    
        # (3) Belongs to the Zou/He scheme
        discrete_velocities_prev =\
              discrete_velocities_prev.at[0, :, :, RIGHT_VELOCITIES].set(
                  equilibrium_discrete_velocities[0, :, :, RIGHT_VELOCITIES])
        
        # (5) Collide according to BGK
        discrete_velocities_post_collision = (discrete_velocities_prev - RELAXATION_OMEGA *
              (discrete_velocities_prev - equilibrium_discrete_velocities))
        
        # (6) Bounce-Back Boundary Conditions to enfore the no-slip 
        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_post_collision = discrete_velocities_post_collision.at[obstacle_mask, LATTICE_INDICES[i]].set(
                                                          discrete_velocities_prev[obstacle_mask, OPPOSITE_LATTICE_INDICES[i]])
        
       
        # (7) Stream alongside lattice velocities
        discrete_velocities_streamed = discrete_velocities_post_collision
        for i in range(N_DISCRETE_VELOCITIES):
            discrete_velocities_streamed = discrete_velocities_streamed.at[:, :, :, i].set(
                      jnp.roll(
                          jnp.roll(
                              jnp.roll(
                                discrete_velocities_post_collision[:, :, :, i], LATTICE_VELOCITIES[0, i], axis = 0),
                          	  LATTICE_VELOCITIES[1, i], axis = 1),
                          LATTICE_VELOCITIES[2, i], axis = 2))
    
        return discrete_velocities_streamed
        
    def run(discrete_velocities_prev):   
        for i in tqdm(range(N_ITERATIONS)):
            discrete_velocities_next = update(discrete_velocities_prev)
            discrete_velocities_prev = discrete_velocities_next
            
            if i % PLOT_EVERY_N_STEPS == 0 and i > SKIP_FIRS_N_ITERATIONS - PLOT_EVERY_N_STEPS:
                density = get_density(discrete_velocities_next)
                macroscopic_velocities = get_macroscopic_velocities(
                    discrete_velocities_next,
                    density)
                print('\n', jnp.max(macroscopic_velocities))
                velocity_magnitude = jnp.linalg.norm(
                    macroscopic_velocities,
                    axis=-1,
                    ord=2)
                fig = plt.figure(figsize = (15, 3))
                cont = plt.contourf(X[:, :, nz//2], Y[:, :,  nz//2], jnp.flip(velocity_magnitude[:, :,  nz//2], axis = 1), alpha=0.8, cmap=cmr.iceburn)  
                plt.axis('scaled')
                plt.axis('off')
                plt.show()
            
        return 
    
    run(discrete_velocities_prev)

