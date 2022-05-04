"""

# The Lattice Boltzmann Method - unit conversion

This code uses the Lattice Boltzmann (LBM) Method for numerical simulation of 
fluid flow to calculate the flow around a sphere. Written in python and jax. This 
code will focus on converting relevant variables from physical to lattice units 
and back.

The code is adjusted from the code presented by Machine Learning & Simulation (MLS) in 2D:
- On Youtube: 
    https://www.youtube.com/watch?v=ZUXmO4hu-20&list=LL&index=1&ab_channel=MachineLearning%26Simulation) 
- and Github:
    https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/lattice_boltzmann_method_python_jax.py

Expanded to 3D:
- In Google Colab:
    https://colab.research.google.com/drive/1F3EH9_2N3lkEpgQXOScR3lcQ6oqCARPk?usp=sharing
- and on Github:
    https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/D3Q19_lattice_bolzmann_method_python_jax.py

Maybe try it out with stress and force:
- In Google Colab:
    https://colab.research.google.com/drive/1oryCdOPXapOWxGSgCDNkvSUQ_MahfoRX?usp=sharing
- and on Github:   
    https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/simulation_scripts/D3Q19_lattice_boltzmann_method_stress_force_drag.py
    
It is recommended to watch that video first and go through the notebook in 3D, 
because a lot of explanation of this method, the setup and syntax mentioned in 
that video and code will be skipped here. The force as determined in the relevant
notebook/code will be used in the end of this code to convert force from lattice units
to physical units.

This code was originally written in google colab:
https://colab.research.google.com/drive/1OkpFHdGmCEmfEq1a_FgKsiRgc-gh6g2A?usp=sharing
"""

# Import packages
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

# Enable 64bit
jax.config.update("jax_enable_x64", True)

# Dimensions of domain in number of grid cells
NY = 50
NZ = 50
NX = 300

# Radius of the sphere in physical units (_P), meters
radius_P = 1 # m

# Dimensions of domain in physical units
xmin_P = ymin_P = zmin_P = 0
xmax_P = 60
ymax_P = zmax_P = 10

# The speed of sound in physical and lattice units (_L)
speed_of_sound_P  = 1500              # m/s
speed_of_sound_L  = 1/(jnp.sqrt(3))  

# The density in physical units
density_P = 1400 # kg/m³ 

# Physical parameters in physical units
KINEMATIC_VISCOSITY_P        = 0.1         # in m2/s
HORIZONTAL_INFLOW_VELOCITY_P = 10          # in m/s

# Lattice Boltzmann Lattice
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

# Plotting parameters
SECONDS_OF_SIMULATION = 3
PLOT_EVERY_N_STEP = 250
SKIP_FIRST_N = 1000

# Get conversion factors for length: Δxₚ
ΔX_P = (xmax_P - xmin_P)/NX
ΔY_P = (ymax_P - ymin_P)/NY
ΔZ_P = (zmax_P - zmin_P)/NZ

# Get conversion factor for time Δtₚ

# By determining the relevant factor based on Δxₚ and the speed of sound L/P ratio
# ΔT_P              = (speed_of_sound_L / 
#                      speed_of_sound_P * 
#                      ΔX_P # in s per lattice time step 

# or setting Δtₚ, which artificially raises the Mach number
ΔT_P              = 1.6e-4

# Get conversion factor mass Δmₚ
ΔM_P = density_P * (ΔX_P ** 3)

# Define functions based on conversion factors based on units of variable
def convert_to_lattice_units(value, length = 0, time = 0, mass = 0):
  return value * (ΔX_P ** -length) * (ΔT_P ** -time) * (ΔM_P ** -mass)

def convert_to_physical_units(value, length = 0, time = 0, mass = 0):
  return value * (ΔX_P ** length) * (ΔT_P ** time) * (ΔM_P ** mass)

# Convert physical radius to lattice units
RADIUS_L = convert_to_lattice_units(radius_P, length = 1)

KINEMATIC_VISCOSITY_L        = convert_to_lattice_units(
    KINEMATIC_VISCOSITY_P, 
    length = 2, 
    time = -1)
HORIZONTAL_INFLOW_VELOCITY_L = convert_to_lattice_units(
    HORIZONTAL_INFLOW_VELOCITY_P,
    length = 1,
    time = -1)

# Dimensionless constants
reynolds_number_L = (HORIZONTAL_INFLOW_VELOCITY_L * 2 * RADIUS_L) / KINEMATIC_VISCOSITY_L
reynolds_number_P = (HORIZONTAL_INFLOW_VELOCITY_P * 2 * radius_P) / KINEMATIC_VISCOSITY_P

# Mach number and relaxation time to determine stability and physical representation
mach_number_L = HORIZONTAL_INFLOW_VELOCITY_L / speed_of_sound_L
RELAXATION_OMEGA = (1.0 / 
                    (KINEMATIC_VISCOSITY_L /
                     (speed_of_sound_L**2) + 
                     0.5)
                    )

iterations_per_second = 1/ΔT_P
NUMBER_OF_ITERATIONS = round(iterations_per_second * SECONDS_OF_SIMULATION)

print('Number of iterations:      {NUMBER_OF_ITERATIONS}')
print(f'Lattice Reynolds number:  {reynolds_number_L: g}')
print(f'Physical Reynolds number: {reynolds_number_P: g}')
print(f'Mach number:              {mach_number_L: g}')
print(f'Relaxation time:          {1.0 /RELAXATION_OMEGA: g}')

# Now that we can convert from lattice coordinates and back we are done!
# But it is more fun to apply it...

@jax.jit
def get_density(discrete_velocities):
    density = jnp.sum(discrete_velocities, axis=-1)
    return density

@jax.jit
def get_macroscopic_velocities(discrete_velocities, density):
    return jnp.einsum("...Q,dQ->...d", discrete_velocities, LATTICE_VELOCITIES) / density[..., jnp.newaxis]

@jax.jit
def get_equilibrium_discrete_velocities(macroscopic_velocities, density):
    projected_discrete_velocities = jnp.einsum("dQ,...d->...Q", LATTICE_VELOCITIES, macroscopic_velocities)
    macroscopic_velocity_magnitude = jnp.linalg.norm(macroscopic_velocities, axis=-1, ord=2)
    equilibrium_discrete_velocities = (density[..., jnp.newaxis] * LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, jnp.newaxis, :] *
        (1 + 3 * projected_discrete_velocities + 9/2 * projected_discrete_velocities**2 -
        3/2 * macroscopic_velocity_magnitude[..., jnp.newaxis]**2))    
    return equilibrium_discrete_velocities

if __name__ == '__main__':
    # Define a mesh
    x = jnp.arange(NX)
    y = jnp.arange(NY)
    z = jnp.arange(NZ)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    # Mask for sphere
    sphere = jnp.sqrt((X - NX//5)**2 + (Y - y[NY//2])**2 + (Z - z[NZ//2])**2)
    sphere_mask = sphere < RADIUS_L
    OBSTACLE_MASK = sphere_mask

    # Show views of the sphere (uncomment when needed)
    # plt.imshow(OBSTACLE_MASK[:, :, NZ//2].T)
    # plt.show()
    # plt.imshow(OBSTACLE_MASK[NX//5, :, :].T)
    # plt.show()
    # plt.imshow(OBSTACLE_MASK[:, NY//2, :].T)
    # plt.show()
    
    OPPOSITE_LATTICE_INDICES = jnp.array(
    [jnp.where(
        (LATTICE_VELOCITIES.T == -LATTICE_VELOCITIES[:, i])
        .all(axis = 1))[0] 
     for i in range(N_DISCRETE_VELOCITIES)]).T[0]
    
    RIGHT_VELOCITIES = jnp.where(LATICE_VELOCITIES_X == 1)[0]   # [ 1,  7,  9, 11, 13]
    LEFT_VELOCITIES =  jnp.where(LATICE_VELOCITIES_X ==-1)[0]   # [ 2,  8, 10, 12, 14]
    YZ_VELOCITIES =    jnp.where(LATICE_VELOCITIES_X == 0)[0]   # [ 0,  3,  4,  5,  6, 15, 16, 17, 18]

    VELOCITY_PROFILE = jnp.zeros((NX, NY, NZ, 3))
    VELOCITY_PROFILE = VELOCITY_PROFILE.at[:, :, :, 0].set(HORIZONTAL_INFLOW_VELOCITY_L)
    discrete_velocities_prev = get_equilibrium_discrete_velocities(VELOCITY_PROFILE, 
                                                                  jnp.ones((NX, NY, NZ)))

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
            discrete_velocities_streamed = discrete_velocities_streamed.at[..., i].set(
                jnp.roll(discrete_velocities_post_collision[..., i], 
                LATTICE_VELOCITIES[:, i], axis = (0, 1, 2)))

        return discrete_velocities_streamed
        
    def run(discrete_velocities_prev): 
        maximum_velocity_L = []  
        plt.figure(figsize=(15, 6))
        for i in tqdm(range(NUMBER_OF_ITERATIONS)):
            discrete_velocities_next = update(discrete_velocities_prev)
            discrete_velocities_prev = discrete_velocities_next
            
            density_L = get_density(discrete_velocities_next)
            macroscopic_velocities_L = get_macroscopic_velocities(
                discrete_velocities_next,
                density_L)
            velocity_magnitude_L = jnp.linalg.norm(
                macroscopic_velocities_L,
                axis=-1,
                ord=2)
            maximum_velocity_L.append(jnp.max(velocity_magnitude_L))

            if i % PLOT_EVERY_N_STEP == 0 and i > SKIP_FIRST_N - PLOT_EVERY_N_STEP:   

                plt.subplot(2, 1, 1)
                plt.contourf(X[:, :, NZ//2], Y[:, :,  NZ//2], 
                             velocity_magnitude_L[:, :,  NZ//2], 
                             alpha=0.8, cmap=cmr.amber)  
                plt.axis('scaled')
                plt.axis('off')
                plt.show()

                time = jnp.linspace(0, 
                                    convert_to_physical_units(
                                        i, 
                                        time = 1),
                                    i+1)
                
                # print(len(maximum_velocity_L))
                maximum_velocity_P = convert_to_physical_units(
                    jnp.array(maximum_velocity_L),
                    length = 1,
                    time = -1)
                
                # print(len(maximum_velocity_P))
                plt.subplot(2, 1, 2)
                plt.plot(time, maximum_velocity_P, c = 'k')
                plt.xlabel('time (s)')
                plt.ylabel('maximum velocity (m/s)')
                plt.grid()
                
                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)
                plt.clf()
        return discrete_velocities_next

    discrete_velocities = run(discrete_velocities_prev)

    # The momentum Exchange method for determining the forces acting on the
    # object (See relevant notebook)
    
    MOMENTUM_EXCHANGE_MASK_IN  = jnp.zeros((NX, NY, NZ, 19)) > 0
    MOMENTUM_EXCHANGE_MASK_OUT = jnp.zeros((NX, NY, NZ, 19)) > 0
    
    for i, (x, y, z) in enumerate(LATTICE_VELOCITIES.T):
        # Determine the momentum going into the object:
        location_in = jnp.logical_and(
                    jnp.roll(
                        jnp.logical_not(OBSTACLE_MASK),
                        (x, y, z), 
                        axis = (0, 1, 2)), 
                    OBSTACLE_MASK)
          
        MOMENTUM_EXCHANGE_MASK_IN = MOMENTUM_EXCHANGE_MASK_IN.at[location_in, i].set(True)
          
        # Determine the momentum going out of the object:
        location_out = jnp.logical_and(
                    jnp.roll(
                        OBSTACLE_MASK,
                        (-x, -y, -z), 
                        axis = (0, 1, 2)),
                    jnp.logical_not(OBSTACLE_MASK))
        
        MOMENTUM_EXCHANGE_MASK_OUT = MOMENTUM_EXCHANGE_MASK_OUT.at[location_out, OPPOSITE_LATTICE_INDICES[i]].set(True)
    
    force_L =  jnp.sum(
                       (LATTICE_VELOCITIES.T[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...] *  
                        discrete_velocities[..., jnp.newaxis])[MOMENTUM_EXCHANGE_MASK_IN] + 
                       (LATTICE_VELOCITIES.T[OPPOSITE_LATTICE_INDICES][jnp.newaxis, jnp.newaxis, jnp.newaxis, ...] *  
                        discrete_velocities[..., jnp.newaxis])[MOMENTUM_EXCHANGE_MASK_OUT], 
                       axis = 0)
    
    # And to convert the force in lattice units to the force in physical units (kg¹m¹s⁻²)
    force_P = convert_to_physical_units(force_L,
                                    mass = 1,
                                    length = 1,
                                    time = -2)

    print(f'Force over the horizontal axis: {force_P[0]: ,g} kg⋅m/s²')