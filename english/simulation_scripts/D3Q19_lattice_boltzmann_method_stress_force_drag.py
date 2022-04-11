# -*- coding: utf-8 -*-
"""LBM 3D drag.ipynb

# The Lattice Boltzmann Method - stress and force

This notebook uses the Lattice Boltzmann (LBM) Method for numerical simulation of fluid flow to calculate the flow around a sphere. Written in python and jax. This notebook will focus on the stress tensor and force acting on the sphere.
The stress tensor will be determined [using Chapman-Enskog expensions](https://arxiv.org/pdf/0812.3242.pdf), force will be determined using the Momentum Exchange Method and the drag coefficient will be determined using Stokes Law. In the end, only the drag coefficient will be presented, since it is non-dimensional and unit conversion will be left for a future notebook.

The code is adjusted from the the code presented by Machine Learning & Simulation (MLS) in 2D:
- On Youtube: 
    https://www.youtube.com/watch?v=ZUXmO4hu-20&list=LL&index=1&ab_channel=MachineLearning%26Simulation) 
- and Github:
    https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/lattice_boltzmann_method_python_jax.py

Expanded to 3D:
- In Google Colab:
    https://colab.research.google.com/drive/1F3EH9_2N3lkEpgQXOScR3lcQ6oqCARPk?usp=sharing
- and on Github:
    https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/D3Q19_lattice_bolzmann_method_python_jax.p


It is recommended to watch that video first and go through the notebook in 3D, because a lot of explanation of this method, the setup and syntax mentioned in that video and code will be skipped here.

This code was originally written in google colab:
    https://colab.research.google.com/drive/1oryCdOPXapOWxGSgCDNkvSUQ_MahfoRX?usp=sharing
More explanation (with some figures as well) is given there.

"""

# Import packages
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib import cm
from tqdm import tqdm

# Enable 64bit
jax.config.update("jax_enable_x64", True)

# Define functions
def get_strain_rate_tensor_FD(macroscopic_velocities):
  disalligned_gradients = jnp.array(
      [jnp.gradient(macroscopic_velocities[..., i]) for i in range(3)])
  
  gradients = jnp.einsum('ij... -> ...ij', 
                         disalligned_gradients)
  return - (gradients + 
            jnp.einsum('...ij -> ...ji', 
                       gradients)
            )/2

def get_cαcβ(alpha, beta):
    c_alpha = LATTICE_VELOCITIES[alpha, :]
    c_beta  = LATTICE_VELOCITIES[beta,  :]
    return c_alpha * c_beta

def get_non_equilibrium_discrete_velocities(discrete_velocities, macroscopic_velocities, density):
  equilibrium_discrete_velocities = get_equilibrium_discrete_velocities(
                                      macroscopic_velocities,
                                      density)
  return discrete_velocities - equilibrium_discrete_velocities

def get_stress_tensor(discrete_velocities, macroscopic_velocities, density):
  non_equilibrium_discrete_velocities = get_non_equilibrium_discrete_velocities(
      discrete_velocities, macroscopic_velocities, density)
  
  non_equilibrium_stress_tensor = ((1 - RELAXATION_OMEGA / 2) * 
                                   jnp.sum(cαcβ[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...] * 
                                           non_equilibrium_discrete_velocities[:, :, :, jnp.newaxis, jnp.newaxis, :],
                                           axis = -1))
  return non_equilibrium_stress_tensor

def get_strain_rate_tensor_LB(discrete_velocities, macroscopic_velocities, density):
  stress_tensor = get_stress_tensor(discrete_velocities, macroscopic_velocities, density)
  strain_rate_tensor = (stress_tensor /
                        (2 * 
                         density[..., jnp.newaxis, jnp.newaxis] * 
                         KINEMATIC_VISCOSITY_L)
                        )
  return strain_rate_tensor

def get_density(discrete_velocities):
    density = jnp.sum(discrete_velocities, axis=-1)
    return density

def get_macroscopic_velocities(discrete_velocities, density):
    return jnp.einsum("...Q,dQ->...d", discrete_velocities, LATTICE_VELOCITIES) / density[..., jnp.newaxis]

def get_equilibrium_discrete_velocities(macroscopic_velocities, density):
    projected_discrete_velocities = jnp.einsum("dQ,...d->...Q", LATTICE_VELOCITIES, macroscopic_velocities)
    macroscopic_velocity_magnitude = jnp.linalg.norm(macroscopic_velocities, axis=-1, ord=2)
    equilibrium_discrete_velocities = (density[..., jnp.newaxis] * LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, jnp.newaxis, :] *
        (1 + 3 * projected_discrete_velocities + 9/2 * projected_discrete_velocities**2 -
        3/2 * macroscopic_velocity_magnitude[..., jnp.newaxis]**2))    
    return equilibrium_discrete_velocities

# Dimensions of domain in number of grid cells
ny = 50
nz = 50
nx = 300
radius_L = 5

# Setup the flow regime
KINEMATIC_VISCOSITY_L        = 0.002
HORIZONTAL_INFLOW_VELOCITY_L = 0.02

# Setup simulation iterations and frequency of plots
NUMBER_OF_ITERATIONS = 5000
PLOT_EVERY_N_STEP = 25
SKIP_FIRST_N = 1000

# Determin relevant coefficients and the relaxation time (or inversely, relaxation omega)
reynolds_number_L = (HORIZONTAL_INFLOW_VELOCITY_L * 2 * radius_L) / KINEMATIC_VISCOSITY_L
speed_of_sound_L = 1/jnp.sqrt(3)
mach_number_L = HORIZONTAL_INFLOW_VELOCITY_L / speed_of_sound_L**2
RELAXATION_OMEGA = (1.0 / (KINEMATIC_VISCOSITY_L/(speed_of_sound_L**2) + 0.5))

print('Reynolds number: ', reynolds_number_L)
print('Mach number:             ', mach_number_L)
print('Relaxation time:         ', 1/RELAXATION_OMEGA)

# Define a mesh
x = jnp.arange(nx)
y = jnp.arange(ny)
z = jnp.arange(nz)
X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

# Construct the sphere
sphere = jnp.sqrt((X - x[nx//5])**2 + (Y - y[ny//2])**2 + (Z - z[nz//2])**2)
OBSTACLE_MASK = sphere < radius_L

plt.imshow(OBSTACLE_MASK[:, :, nz//2].T)
plt.show()
plt.imshow(OBSTACLE_MASK[nx//5, :, :].T)
plt.show()
plt.imshow(OBSTACLE_MASK[:, ny//2, :].T)
plt.show()

# Setup the discrete velocities
N_DISCRETE_VELOCITIES = 19

LATTICE_INDICES =     jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18])
LATICE_VELOCITIES_X = jnp.array([0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0])
LATICE_VELOCITIES_Y = jnp.array([0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1])
LATICE_VELOCITIES_Z = jnp.array([0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1])

LATTICE_VELOCITIES = jnp.array([LATICE_VELOCITIES_X,
                                LATICE_VELOCITIES_Y,
                                LATICE_VELOCITIES_Z])

LATTICE_WEIGHTS = jnp.array([# rest particle
                             1/3, 
                             
                             # face-connected neighbors
                             1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
                             
                             # edge-connected neighbors
                             1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36])

OPPOSITE_LATTICE_INDICES = jnp.array(
    [jnp.where(
        (LATTICE_VELOCITIES.T == -LATTICE_VELOCITIES[:, i])
        .all(axis = 1))[0] 
     for i in range(N_DISCRETE_VELOCITIES)]).T[0]

RIGHT_VELOCITIES = jnp.where(LATICE_VELOCITIES_X == 1)[0]   # [ 1,  7,  9, 11, 13]
LEFT_VELOCITIES =  jnp.where(LATICE_VELOCITIES_X ==-1)[0]   # [ 2,  8, 10, 12, 14]
YZ_VELOCITIES =    jnp.where(LATICE_VELOCITIES_X == 0)[0]   # [ 0,  3,  4,  5,  6, 15, 16, 17, 18]

# For determining stress
alpha, beta = jnp.meshgrid(jnp.arange(3), jnp.arange(3))
cαcβ = get_cαcβ(alpha, beta)

# Determining force
MOMENTUM_EXCHANGE_MASK_IN = jnp.zeros((nx, ny, nz, 19)) > 0
momentum_exchange_mask_in_per_iter = jnp.zeros((nx, ny, nz, 19)) > 0
MOMENTUM_EXCHANGE_MASK_OUT = jnp.zeros((nx, ny, nz, 19)) > 0
momentum_exchange_mask_out_per_iter = jnp.zeros((nx, ny, nz, 19)) > 0
for i, (x, y, z) in enumerate(LATTICE_VELOCITIES.T):
  # Determine the momentum going into the object:
  location_in = jnp.logical_and(
            jnp.roll(
                jnp.roll(
                      jnp.roll(jnp.logical_not(OBSTACLE_MASK), 
                               x, axis = 0),
                         y, axis = 1),
                     z, axis = 2), 
            OBSTACLE_MASK)
  MOMENTUM_EXCHANGE_MASK_IN = MOMENTUM_EXCHANGE_MASK_IN.at[location_in, i].set(True)

  # Determine the momentum going out of the object:
  location_out = jnp.logical_and(
            jnp.roll(
                jnp.roll(
                      jnp.roll(OBSTACLE_MASK, 
                               -x, axis = 0),
                         -y, axis = 1),
                     -z, axis = 2), 
            jnp.logical_not(OBSTACLE_MASK))
  MOMENTUM_EXCHANGE_MASK_OUT = MOMENTUM_EXCHANGE_MASK_OUT.at[location_out, OPPOSITE_LATTICE_INDICES[i]].set(True)

VELOCITY_PROFILE = jnp.zeros((nx, ny, nz, 3))
VELOCITY_PROFILE = VELOCITY_PROFILE.at[:, :, :, 0].set(HORIZONTAL_INFLOW_VELOCITY_L)
discrete_velocities_prev = get_equilibrium_discrete_velocities(VELOCITY_PROFILE, 
                                                               jnp.ones((nx, ny, nz)))

@jax.jit
def get_force(discrete_velocities):
  return jnp.sum(
                 (LATTICE_VELOCITIES.T[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...] *  
                  discrete_velocities[..., jnp.newaxis])[MOMENTUM_EXCHANGE_MASK_IN] + 
                 (LATTICE_VELOCITIES.T[OPPOSITE_LATTICE_INDICES][jnp.newaxis, jnp.newaxis, jnp.newaxis, ...] *  
                  discrete_velocities[..., jnp.newaxis])[MOMENTUM_EXCHANGE_MASK_OUT], 
                 axis = 0)

@jax.jit
def update(discrete_velocities_prev):
    # (1) Prescribe the outflow BC on the right boundary. Flow can go out, but not back in.
    discrete_velocities_prev = discrete_velocities_prev.at[-1, ..., LEFT_VELOCITIES].set(discrete_velocities_prev[-2, ..., LEFT_VELOCITIES])

    # (2) Determine macroscopic velocities
    density_prev = get_density(discrete_velocities_prev)
    macroscopic_velocities_prev = get_macroscopic_velocities(
        discrete_velocities_prev,
        density_prev)

    # (3) Prescribe Inflow Dirichlet BC using Zou/He scheme in 3D: 
    # https://arxiv.org/pdf/0811.4593.pdf
    # https://terpconnect.umd.edu/~aydilek/papers/LB.pdf
    macroscopic_velocities_prev = macroscopic_velocities_prev.at[0, ..., :].set(VELOCITY_PROFILE[0, ..., :])
    lateral_densities = get_density(jnp.einsum('i...->...i', discrete_velocities_prev[0, ..., YZ_VELOCITIES]))
    left_densities = get_density(jnp.einsum('i...->...i', discrete_velocities_prev[0, ..., LEFT_VELOCITIES]))
    density_prev = density_prev.at[0, ...].set((lateral_densities + 2 * left_densities) / 
                                                (1 - macroscopic_velocities_prev[0, ..., 0]))

    # (4) Compute discrete Equilibria velocities
    equilibrium_discrete_velocities = get_equilibrium_discrete_velocities(
       macroscopic_velocities_prev,
       density_prev)

    # (3) Belongs to the Zou/He scheme
    discrete_velocities_prev =\
          discrete_velocities_prev.at[0, ..., RIGHT_VELOCITIES].set(
              equilibrium_discrete_velocities[0, ..., RIGHT_VELOCITIES])
    
    # (5) Collide according to BGK
    discrete_velocities_post_collision = (discrete_velocities_prev - RELAXATION_OMEGA *
          (discrete_velocities_prev - equilibrium_discrete_velocities))
    
    # (6) Bounce-Back Boundary Conditions to enfore the no-slip 
    for i in range(N_DISCRETE_VELOCITIES):
        discrete_velocities_post_collision = discrete_velocities_post_collision.at[OBSTACLE_MASK, LATTICE_INDICES[i]].set(
                                                      discrete_velocities_prev[OBSTACLE_MASK, OPPOSITE_LATTICE_INDICES[i]])
   
    # (7) Stream alongside lattice velocities
    discrete_velocities_streamed = discrete_velocities_post_collision
    for i in range(N_DISCRETE_VELOCITIES):
        discrete_velocities_streamed_i = discrete_velocities_post_collision[..., i]
        for axis in range(LATTICE_VELOCITIES.shape[0]):
              discrete_velocities_streamed_i = jnp.roll(discrete_velocities_streamed_i, LATTICE_VELOCITIES[axis, i], axis = axis)
        discrete_velocities_streamed = discrete_velocities_streamed.at[..., i].set(discrete_velocities_streamed_i)

    return discrete_velocities_streamed

def run(discrete_velocities_prev, axis1 = 0, axis2 = 0):   
    C_d = []
    for i in tqdm(range(NUMBER_OF_ITERATIONS)):
        discrete_velocities_next = update(discrete_velocities_prev)
        discrete_velocities_prev = discrete_velocities_next

        horizontal_force = get_force(discrete_velocities_next)[0] 
        drag_coefficient = drag_coefficient = 2*abs(horizontal_force)/(1 * (jnp.pi * radius_L**2) * (HORIZONTAL_INFLOW_VELOCITY_L**2))
        C_d.append(drag_coefficient)
        
        if i % PLOT_EVERY_N_STEP == 0 and i > SKIP_FIRST_N - PLOT_EVERY_N_STEP:
            density = get_density(discrete_velocities_next)
            macroscopic_velocities = get_macroscopic_velocities(
                discrete_velocities_next,
                density)
            velocity_magnitude = jnp.linalg.norm(
                macroscopic_velocities,
                axis=-1,
                ord=2)
            
            shear_rate_tensor = get_strain_rate_tensor_FD(macroscopic_velocities)
            strain_rate_FD = shear_rate_tensor[..., axis1, axis2]

            shear_stress = get_strain_rate_tensor_LB(discrete_velocities_next, macroscopic_velocities, density)
            strain_rate_LB = shear_stress[..., axis1, axis2]

            fig = plt.figure(figsize = (15, 3))
            cont = plt.contourf(X[:, :, nz//2], Y[:, :,  nz//2], 
                                jnp.flip(velocity_magnitude[:, :,  nz//2], axis = 1), 
                                alpha=0.8, cmap=cmr.amber)  
            plt.axis('scaled')
            plt.axis('off')
            plt.show()

            fig = plt.figure(figsize = (15, 3))
            cont = plt.contourf(X[:, :, nz//2], Y[:, :,  nz//2], 
                                jnp.flip(strain_rate_FD[..., nz//2], axis = 1), 
                                levels = 50, alpha=0.8, cmap=cm.seismic)  
            plt.axis('scaled')
            plt.axis('off')
            plt.show()

            fig = plt.figure(figsize = (15, 3))
            cont = plt.contourf(X[:, :, nz//2], Y[:, :,  nz//2], 
                                jnp.flip(strain_rate_LB[..., nz//2], axis = 1), 
                                levels = 50, alpha=0.8, cmap=cm.seismic)  
            
            plt.axis('scaled')
            plt.axis('off')
            plt.show()

            
            fig = plt.figure(figsize = (15, 3))
            plt.plot(C_d[SKIP_FIRST_N:], 'k')
            plt.grid()
            plt.show()
    return discrete_velocities_next
discrete_velocities = run(discrete_velocities_prev, axis1 = 0, axis2 = 1)

