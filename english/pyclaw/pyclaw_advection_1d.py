"""
Solves the 1D advection equation with a Riemann Solver and the Finite Volume
Method using the PyClaw Package.

    ∂q/∂t + u ∂q/∂x = 0

q : The conserved quantity e.g. a density

u : The advection speed

------

Scenario: A rectangular initial condition is transported to the right

         ┌────────────────────────────────────────┐   
    1.00 │⠀⡇⠀⠀⠀⠀⠀⠀⡖⠒⠒⠒⠒⠒⠒⠒⢲⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ y1
         │⠀⡇⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀ ⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│   
         │⠀⡇⠀⠀⠀⠀⠀⠀⢸ ⠀⠀⠀⠀⠀ ⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│   
         │⠀⡇⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀ ⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│   
         │⠀⡇⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀ ⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│   
         │⠀⡇⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀ ⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│   
         │⠀⡇⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⢸ ⠀⠀⠀⠀u⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│   
         │⠀⡇⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⢸ ⠀⠀-----> ⠀⠀⠀⠀⠀⠀⠀│   
         │⠀⡇⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⢸ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│   
         │⠀⡇⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⢸ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│   
         │⠀⡇⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⢸ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│   
         │⠀⡇⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⢸ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│   
         │⠀⡇⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⢸ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│   
         │⠀⡇⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⢸  ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│   
    0.00 │⠤⡧⠤⠤⠤⠤⠤⠤⠼         ⠧⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│   
         └────────────────────────────────────────┘   
         ⠀ 0.00⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀1.00⠀   

-> periodic Boundary conditions on the left and right end

------

Solution strategy

1. Instantiate a ClawPack Solver with an attached Riemann Solver according to
   the PDE being solved including the Boundary Conditions.

2. Define the Finite Volume Mesh.

3. Instantiate fields on the Mesh (that save the conserved quantity "q").

4. Prescribe the initial condition.

5. Set Problem-Specific Parameters (in our case the advection speed "u").

6. Instantiate a controller that takes care of the time integration and attach
   solution and solver to it.

7. Run the simulation and visualize the results.
"""
from clawpack import pyclaw
from clawpack import riemann
import numpy as np

def main():
    # (1) Define the Finite Voluem solver to be used with a Riemann Solver from
    # the library
    solver = pyclaw.ClawSolver1D(riemann.advection_1D)
    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic

    # (2) Define the mesh
    x_dimension = pyclaw.Dimension(0.0, 1.0, 100)
    domain = pyclaw.Domain(x_dimension)

    # (3) Instantiate a solution field on the Mesh
    solution = pyclaw.Solution(solver.num_eqn, domain,)

    # (4) Prescribe an initial state
    state = solution.state
    cell_center_coordinates = state.grid.p_centers[0]
    state.q[0, :] = np.where(
        (cell_center_coordinates > 0.2)
        &
        (cell_center_coordinates < 0.4),
        1.0,
        0.0,
    )

    # (5) Assign problem-specific parameters ("u" refers to the advection speed)
    state.problem_data["u"] = 1.0

    # (6) The controller takes care of the time integration
    controller = pyclaw.Controller()
    controller.solution = solution
    controller.solver = solver
    controller.tfinal = 1.0

    # (7) Run and visualize
    controller.run()

    pyclaw.plot.interactive_plot()


if __name__ == "__main__":
    main()
