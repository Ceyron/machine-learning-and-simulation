"""
Solves the incompressible Navier Stokes equations in a lid-driven cavity
scenario using Finite Elements and Chorin's Projection. We will employ
the FEniCS Python Package

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

----

Lid-Driven Cavity Scenario:


                            ------>>>>> u_top

          1 +-------------------------------------------------+
            |                                                 |
            |             *                      *            |
            |          *           *    *    *                |
        0.8 |                                                 |
            |                                 *               |
            |     *       *                                   |
            |                      *     *                    |
        0.6 |                                            *    |
u = 0       |      *                             *            |   u = 0
v = 0       |                             *                   |   v = 0
            |                     *                           |
            |           *                *         *          |
        0.4 |                                                 |
            |                                                 |
            |      *            *             *               |
            |           *                             *       |
        0.2 |                       *           *             |
            |                               *                 |
            |  *          *      *                 *       *  |
            |                            *                    |
          0 +-------------------------------------------------+
            0        0.2       0.4       0.6       0.8        1

                                    u = 0
                                    v = 0

* Velocity components have zero initial condition.
* Homogeneous Dirichlet Boundary Conditions everywhere except for horizontal
  velocity at top. It is driven by an external flow.

-----

Denote by:

u   : The velocity field (from the previous iteration)
u*  : The velocity field after a tentative momentum step
u** : The velocity field after incompressibility correction (=after projection)
p*  : The pressure field at the next time step
Δt  : The time step size

v   : The test function to the function space of u
q   : The test function to the function space of p

Solution strategy:

1. Solve the weak form to the momentum equation without the pressure gradient.
   Treat the convection explicitly and the diffusion implicitly. Use an Euler
   time step. Solve for u*

   0 = 1/Δt <(u* - u), v> + <(∇u)u, v> + ν <∇u*, ∇v>

2. Solve the weak form to the pressure poisson equation with the divergence
   of the tentative velocity field as the right-hand side. Solve for p*

   <∇p*, ∇q> = - 1/Δt <∇⋅u*, q>

3. Solve the weak form to the incompressibility correction. Solve for u**

   <u**, v> = <u*, v> - 1/Δt <∇p*, v>


In order to guarantee stability of the spatial discretization, we will
use so-called Taylor-Hood elements -> The order of Ansatz/Shape functions
for the pressure function space has to be one order smaller than the 
velocity function space.

-----

Note on stability:

We use an explicit treatment of the convection/advection. Hence, choose
the time step length with care.
"""

import fenics as fe
from tqdm import tqdm # optional
import matplotlib.pyplot as plt

N_POINTS_P_AXIS = 41

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 100
KINEMATIC_VISCOSITY = 0.01 # -> Re = 100

def main():
    mesh = fe.UnitSquareMesh(N_POINTS_P_AXIS, N_POINTS_P_AXIS)

    # Taylor-Hood Elements. The order of the function space for the pressure has
    # to be one order lower than for the velocity
    velocity_function_space = fe.VectorFunctionSpace(mesh, "Lagrange", 2)
    pressure_function_space = fe.FunctionSpace(mesh, "Lagrange", 1)

    u_trial = fe.TrialFunction(velocity_function_space)
    p_trial = fe.TrialFunction(pressure_function_space)
    v_test = fe.TestFunction(velocity_function_space)
    q_test = fe.TestFunction(pressure_function_space)

    # Define the Boundary Condition
    stationary_wall_boundary_condition = fe.DirichletBC(
        velocity_function_space,
        (0.0, 0.0),
        """
        on_boundary && (x[0] < DOLFIN_EPS || x[1] < DOLFIN_EPS || x[0] > (1.0 - DOLFIN_EPS))
        """
    )
    moving_wall_boundary_condition = fe.DirichletBC(
        velocity_function_space,
        (1.0, 0.0),
        """
        on_boundary && (x[1] > (1.0 - DOLFIN_EPS))
        """
    )
    velocity_boundary_conditions = [stationary_wall_boundary_condition, moving_wall_boundary_condition]

    # Define the solution fields involved
    u_prev = fe.Function(velocity_function_space)
    u_tent = fe.Function(velocity_function_space)
    u_next = fe.Function(velocity_function_space)
    p_next = fe.Function(pressure_function_space)

    # Weak form of the momentum equation
    momentum_weak_form_residuum = (
        1.0 / TIME_STEP_LENGTH * fe.inner(u_trial - u_prev, v_test) * fe.dx
        +
        fe.inner(fe.grad(u_prev) * u_prev, v_test) * fe.dx
        +
        KINEMATIC_VISCOSITY * fe.inner(fe.grad(u_trial), fe.grad(v_test)) * fe.dx
    )
    momentum_weak_form_lhs = fe.lhs(momentum_weak_form_residuum)
    momentum_weak_form_rhs = fe.rhs(momentum_weak_form_residuum)

    # Weak form of the pressure poisson problem
    pressure_poisson_weak_form_lhs = fe.inner(fe.grad(p_trial), fe.grad(q_test)) * fe.dx
    pressure_poisson_weak_form_rhs = - 1.0 / TIME_STEP_LENGTH * fe.div(u_tent) * q_test * fe.dx

    # Weak form of the velocity update equation
    velocity_update_weak_form_lhs = fe.inner(u_trial, v_test) * fe.dx
    velocity_update_weak_form_rhs = (
        fe.inner(u_tent, v_test) * fe.dx
        -
        TIME_STEP_LENGTH * fe.inner(fe.grad(p_next), v_test) * fe.dx
    )

    # Pre-Compute the system matrices (because they do not greatly change)
    momentum_assembled_system_matrix = fe.assemble(momentum_weak_form_lhs)
    pressure_poisson_assembled_system_matrix = fe.assemble(pressure_poisson_weak_form_lhs)
    velocity_update_assembled_system_matrix = fe.assemble(velocity_update_weak_form_lhs)

    for t in tqdm(range(N_TIME_STEPS)):
        # (1) Solve for tentative velocity
        momentum_assembled_rhs = fe.assemble(momentum_weak_form_rhs)
        [bc.apply(momentum_assembled_system_matrix, momentum_assembled_rhs) for bc in velocity_boundary_conditions]
        fe.solve(
            momentum_assembled_system_matrix,
            u_tent.vector(),
            momentum_assembled_rhs,
            "gmres",
            "ilu",
        )

        # (2) Solve for the pressure
        pressure_poisson_assembled_rhs = fe.assemble(pressure_poisson_weak_form_rhs)
        fe.solve(
            pressure_poisson_assembled_system_matrix,
            p_next.vector(),
            pressure_poisson_assembled_rhs,
            "gmres",
            "amg",
        )

        # (3) Correct the velocities to be incompressible
        [bc.apply(momentum_assembled_system_matrix, momentum_assembled_rhs) for bc in velocity_boundary_conditions]
        velocity_update_assembled_rhs = fe.assemble(velocity_update_weak_form_rhs)
        fe.solve(
            velocity_update_assembled_system_matrix,
            u_next.vector(),
            velocity_update_assembled_rhs,
            "gmres",
            "ilu",
        )

        # Advance in time
        u_prev.assign(u_next)

        # Visualize interactively
        fe.plot(u_next)
        plt.draw()
        plt.pause(0.02)
        plt.clf()


if __name__ == "__main__":
    main()
