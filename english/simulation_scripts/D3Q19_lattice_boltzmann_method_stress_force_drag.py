"""

# The Lattice Boltzmann Method - stress and force

This notebook uses the Lattice Boltzmann (LBM) Method for numerical simulation 
of fluid flow to calculate the flow around a sphere. Written in python and jax. 
This code will focus on the stress tensor and force acting on the sphere.
The stress tensor will be determined using Chapman-Enskog expensions 
(:https://arxiv.org/pdf/0812.3242.pdf), force will be determined using the 
Momentum Exchange Method and the drag coefficient will be determined as a non-dimensonal
indicator to check on the physical representation of the LBM. Unit conversion 
will be left for a future work.

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


It is recommended to watch that video first and go through the notebook in 3D, 
because a lot of explanation of this method, the setup and syntax mentioned in 
that video and code will be skipped here.

This code was originally written in google colab:
    https://colab.research.google.com/drive/1oryCdOPXapOWxGSgCDNkvSUQ_MahfoRX?usp=sharing

### The stress tensor ###

The stress tensor is the complete representation of the stresses on a point inside 
the fluid (or any material). 
The representation is most easily visualized on a grid cell:

      _____________
     /            /|
    /     ↑ ↗    / |
   /      · →   /  |
  /            /   |
 /____________/  ↑ ↗
 |            |  · →
 |      ↑ ↗   |    /
 |      · →   |   / 
 |            |  /
 |            | /
 |____________|/

Where on each grid face the three arrows represents the forces along that grid 
face along the different axes. For each point in our grid the stress tensor σ is:  
         _              _
        | σ_xx σ_xy σ_xz |
    σ = | σ_yx σ_yy σ_yz |
        |_σ_zx σ_zy σ_zz_|
        
On the row of the tensor are the faces of the above grid cell and on the columns 
the vector of the stress acting on that face. The unit of each value in the tensor 
is N/m² or kg/ms².

The stress tensor is closely related to the strain rate tensor  S  (by  2𝜈S=σ, 
where  𝜈  is the kinematic viscosity). S is defined by the gradients of the 
macroscopic velocities (∇v):  

    S = (∇v + ∇vᵀ)/2

where ᵀ is the transpose operator. The strain rate tensor can be solved using a 
finite differences scheme and the LBM discrete velocities and lattice velocities.
The two methods will be compared for validation.

### Stress tensor in LBM ###

In the LBM, the stress can be seen as the momentum interchanged with 

    σₐᵦ = (1−1/(2τ)) * ∑ᵢ cᵢₐ * cᵢᵦ * fⁿᵉᵢ

Where alpha and beta are the relevant axes (x, y or z), supersctip fⁿᵉ is the
non-equilibrium discrete velocities defined by: fⁿᵉ = f - fᵉ. τ is the relaxation
time (1/Ω). cₐ are the lattice velocities over the axis of the rows of the 
stress tensor and  cᵦ the lattice velocities over the aoxis of the column of the 
stress tensor. Subscript i indicates each lattice velocity index.

Now that we have the sress tensot, we can extract the stresses acting on a point
in the fluid.

### Force in LBM ###

But what if we want to know the forces acting on an object? We could solve it 
using the the stress tensor (F =σ⋅n * dA, where n is the normal vector pointing 
in the direction of your object), but there is a LBM friendly way to achieve it 
as well.

Here we determine the forces acting on the object using the momentum exchange method. 
Momentum is the mass of an object times its velocity. Force and momentum are related 
by a change in time, as in: force is momentum over time. The momentum in Lattice 
Boltzmann space is  fᵢcᵢ, which are the lattice velocities of each lattice index 
multiplied by their discrete counterpart. When we see the populations  fᵢ  of 
our lattices as particles, we can determine the momentum of those particles 
hitting the object and transfering their momentum. To determine this exchange of 
momentum, we need to determine which populations would cross the object boundary 
(both from the fluid into the object and the other way around).

When we know which lattice velocities are going to hit the object (indiciated with i
and which once are pointing back out of the object (indicated with î, of which 
there is no unicode subscript available), we can calculate their change in momentum 
using:
    
    ΔP = ∑ᵢ(fᵢcᵢ−fîcî)

And the force:
    
    F = ΔP/Δt 
    
Since lattice time is 1 per step we can directly get the force:

    F = ∑ᵢcᵢ(fᵢ + fî)

### Drag coefficient ###

For spheres, the drag coefficient is determines using Stokes Law:
    
    c_drag = 2F / ρ u²A

Where F is the force counter to the direction the sphere is moving in, ρ is the
density of the fluid and A is the projected area of the sphere on the plain
perpendicular to the direction of force.
   
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
  # Get the strain rate tensor using a finite difference scheme 
  _gradients = jnp.gradient(macroscopic_velocities, axis = (0, 1, 2))
  gradients = jnp.einsum('d...D-> ...dD', jnp.array(_gradients))
  return - (gradients + 
            jnp.einsum('...ij -> ...ji', 
                       gradients)
            )/2

# Stress tensor functions
def get_cαcβ(alpha, beta):
    # Precalculate  cᵢₐ * cᵢᵦ for calculating the stress tensor
    c_alpha = LATTICE_VELOCITIES[alpha, :]
    c_beta  = LATTICE_VELOCITIES[beta,  :]
    return c_alpha * c_beta

def get_non_equilibrium_discrete_velocities(discrete_velocities, macroscopic_velocities, density):
  # fⁿᵉ = f - fᵉ
  equilibrium_discrete_velocities = get_equilibrium_discrete_velocities(
                                      macroscopic_velocities,
                                      density)
  return discrete_velocities - equilibrium_discrete_velocities

def get_stress_tensor(discrete_velocities, macroscopic_velocities, density):
  # σₐᵦ = (1−1/(2τ)) * ∑ᵢ cᵢₐ * cᵢᵦ * fⁿᵉᵢ
  non_equilibrium_discrete_velocities = get_non_equilibrium_discrete_velocities(
      discrete_velocities, macroscopic_velocities, density)
  
  non_equilibrium_stress_tensor = ((1 - RELAXATION_OMEGA / 2) * 
                                   jnp.sum(CACB[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...] * 
                                           non_equilibrium_discrete_velocities[:, :, :, jnp.newaxis, jnp.newaxis, :],
                                           axis = -1))
  return non_equilibrium_stress_tensor

@jax.jit
def get_strain_rate_tensor_LB(discrete_velocities, macroscopic_velocities, density):
  # Get the strain rate tensor using the LBM 
  stress_tensor = get_stress_tensor(discrete_velocities, macroscopic_velocities, density)
  strain_rate_tensor = (stress_tensor /
                        (2 * 
                         density[..., jnp.newaxis, jnp.newaxis] * 
                         KINEMATIC_VISCOSITY_L)
                        )
  return strain_rate_tensor

@jax.jit
def get_force(discrete_velocities):
  return jnp.sum(
                 (LATTICE_VELOCITIES.T[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...] *  
                  discrete_velocities[..., jnp.newaxis])[MOMENTUM_EXCHANGE_MASK_IN] + 
                 (LATTICE_VELOCITIES.T[OPPOSITE_LATTICE_INDICES][jnp.newaxis, jnp.newaxis, jnp.newaxis, ...] *  
                  discrete_velocities[..., jnp.newaxis])[MOMENTUM_EXCHANGE_MASK_OUT], 
                 axis = 0)

# Implementing the LBM functions
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

# Dimensions of domain in number of grid cells. Subscript L to stress lattice dimensions.
ny = 50
nz = 50
nx = 300
RADIUS_L = 5

# Setup the flow regime
KINEMATIC_VISCOSITY_L        = 0.002
HORIZONTAL_INFLOW_VELOCITY_L = 0.02

# Setup simulation iterations and frequency of plots
NUMBER_OF_ITERATIONS = 5000
PLOT_EVERY_N_STEP = 25
SKIP_FIRST_N = 1000

# Determin relevant coefficients and the relaxation time (or inversely, relaxation omega)
reynolds_number_L = ((HORIZONTAL_INFLOW_VELOCITY_L * 
                      2 * 
                      RADIUS_L) / 
                     KINEMATIC_VISCOSITY_L)

speed_of_sound_L = 1 / jnp.sqrt(3)

mach_number_L = (HORIZONTAL_INFLOW_VELOCITY_L / 
                 speed_of_sound_L**2)

RELAXATION_OMEGA = (1.0 / 
                    (KINEMATIC_VISCOSITY_L / 
                     speed_of_sound_L**2 + 
                     0.5
                     )
                    )

print(f'Reynolds number:  {reynolds_number_L: g}')
print(f'Mach number:      {mach_number_L: g}')
print(f'Relaxation time:  {1.0 /RELAXATION_OMEGA: g}')

# Define a mesh
x = jnp.arange(nx)
y = jnp.arange(ny)
z = jnp.arange(nz)
X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

# Construct the sphere
sphere = jnp.sqrt((X - x[nx//5])**2 + 
                  (Y - y[ny//2])**2 + 
                  (Z - z[nz//2])**2)

OBSTACLE_MASK = sphere < RADIUS_L

# Show from all sides
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
CACB = get_cαcβ(alpha, beta)

# Force
"""
For determining forece, we also need to determine for each point in the grid
which lattice velocities are going into a grid, and which ones are going out of
the sphere.
"""
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

# Set up the velocity profile
VELOCITY_PROFILE = jnp.zeros((nx, ny, nz, 3))
VELOCITY_PROFILE = VELOCITY_PROFILE.at[:, :, :, 0].set(HORIZONTAL_INFLOW_VELOCITY_L)

if __name__ == '__main__':
    
    discrete_velocities_prev = get_equilibrium_discrete_velocities(VELOCITY_PROFILE, 
                                                                   jnp.ones((nx, ny, nz)))
    
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
    
            # Force and drag
            horizontal_force = get_force(discrete_velocities_next)[0] 
            drag_coefficient = drag_coefficient = 2*abs(horizontal_force)/(1 * (jnp.pi * RADIUS_L**2) * (HORIZONTAL_INFLOW_VELOCITY_L**2))
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
                
                # Get the strain rate tensor using the two methods
                strain_rate_tensor_FD = get_strain_rate_tensor_FD(macroscopic_velocities)
                strain_rate_FD = strain_rate_tensor_FD[..., axis1, axis2]
    
                strain_rate_tensor_LB = get_strain_rate_tensor_LB(discrete_velocities_next, macroscopic_velocities, density)
                strain_rate_LB = strain_rate_tensor_LB[..., axis1, axis2]
    
                fig, axs = plt.subplots(4, 1, figsize = (15, 12))
                axs[0].contourf(X[:, :, nz//2], Y[:, :,  nz//2], 
                                velocity_magnitude[:, :,  nz//2], 
                                alpha=0.8, 
                                cmap=cmr.amber)  
                axs[0].set_aspect('equal', adjustable='box')
                axs[0].axis('off')
    
                # Compare the two methods for determining the strain rate tensor
                axs[1].contourf(X[:, :, nz//2], Y[:, :,  nz//2], 
                                strain_rate_FD[..., nz//2], 
                                levels = 50, 
                                alpha = 0.8, 
                                cmap = cm.seismic)  
                axs[1].set_aspect('equal', adjustable='box')
                axs[1].axis('off')
    
                axs[2].contourf(X[:, :, nz//2], Y[:, :,  nz//2], 
                                strain_rate_LB[..., nz//2], 
                                levels = 50, 
                                alpha = 0.8, 
                                cmap = cm.seismic)  
                
                axs[2].set_aspect('equal', adjustable='box')
                axs[2].axis('off')
                
                # Plot the drag coefficient for so far
                axs[3].plot(C_d[SKIP_FIRST_N:], 'k')
                axs[3].grid()
                axs[3].set_xlabel('Number of iterations')
                axs[3].set_ylabel('Drag coefficient')
                fig.tight_layout()
                plt.draw()
        return discrete_velocities_next
    
    
    discrete_velocities = run(discrete_velocities_prev, axis1 = 0, axis2 = 1)
    
