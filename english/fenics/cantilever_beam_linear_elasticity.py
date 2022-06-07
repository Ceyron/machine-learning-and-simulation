"""
Solves the equations for linear elasticity in structural mechanics using the
Finite Element Method with FEniCS in Python. The primary unknown is the
displacement u which is a 3D vector field in 3D space.


Cauchy Momentum Equation:           − ∇⋅σ = f

Constitutive Stress-Strain:         σ = λ tr(ε) I₃ + 2 μ ε

Displacement-Strain:                ε = 1/2 (∇u + (∇u)ᵀ)


σ  : Cauchy Stress (3x3 matrix)
f  : Forcing right hand side (3D vector)
λ  : Lambda Lame parameter (scalar)
μ  : Mu Lame parameter (scalar)
ε  : Engineering Strain (3x3 matrix)
I₃ : 3x3 Identity tensor (=matrix)
u  : Displacement (3D vector)

∇⋅ : The divergence operator (here contracts matrix to vector)
tr : The trace operator (sum of elements on main diagonal)
∇  : The gradient operator (here expands vector to matrix)
ᵀ  : The transpose operatore 

-------

Scenario:

A cantilever beam is clamped at one end

               .+------------------------+
             .' |                      .'|
            +---+--------------------+'  |      ↓ gravity
   clamped  |   |                    |   |
            |  ,+--------------------+---+
            |.'                      | .'
            +------------------------+'

It is subject to the load due to its own weight and will
deflect accordingly. Under an assumpation of small
deformation the material follows linear elasticity.

------

Solution strategy.:


Define by "v" a test function from the vector function space
on u.

Weak Form:

    <σ(u), ∇v> = <f, v> + <T, v>

with T being the traction vector to prescribe Neumann BC (here =0)


Alternative Weak Form (more commonly used):

    <σ(u), ε(v)> = <f, v> + <T, v>

(valid because σ(u) will always be symmetric and the inner product
of a symmetric matrix with a non-symmetric matrix vanishes)

------

Once the displacement vector field u is obtained, we can compute the
von Mises stress (a scalar stress measure) by

1. Evaluating the deviatoric stress tensor

    s = σ − 1/3 tr(σ) I₃

2. Computing the von Mises stress

    σ_M = √(3/2 s : s)

"""

import fenics as fe

CANTILEVER_LENGTH = 1.0
CANTILEVER_WIDTH = 0.2

N_POINTS_LENGTH = 10
N_POINTS_WIDTH = 3

LAME_MU = 1.0
LAME_LAMBDA = 1.25
DENSITY = 1.0
ACCELERATION_DUE_TO_GRAVITY = 0.016

def main():
    # Mesh and Vector Function Space
    mesh = fe.BoxMesh(
        fe.Point(0.0, 0.0, 0.0),
        fe.Point(CANTILEVER_LENGTH, CANTILEVER_WIDTH, CANTILEVER_WIDTH),
        N_POINTS_LENGTH,
        N_POINTS_WIDTH,
        N_POINTS_WIDTH,
    )
    lagrange_vector_space_first_order = fe.VectorFunctionSpace(
        mesh,
        "Lagrange",
        1,
    )
    
    # Boundary Conditions
    def clamped_boundary(x, on_boundary):
        return on_boundary and x[0] < fe.DOLFIN_EPS

    dirichlet_clamped_boundary = fe.DirichletBC(
        lagrange_vector_space_first_order,
        fe.Constant((0.0, 0.0, 0.0)),
        clamped_boundary,
    )

    # Define strain and stress
    def epsilon(u):
        engineering_strain = 0.5 * (fe.nabla_grad(u) + fe.nabla_grad(u).T)
        return engineering_strain
    
    def sigma(u):
        cauchy_stress = (
            LAME_LAMBDA * fe.tr(epsilon(u)) * fe.Identity(3)
            +
            2 * LAME_MU * epsilon(u)
        )
        return cauchy_stress
    
    # Define weak form
    u_trial = fe.TrialFunction(lagrange_vector_space_first_order)
    v_test = fe.TestFunction(lagrange_vector_space_first_order)
    forcing = fe.Constant((0.0, 0.0, - DENSITY * ACCELERATION_DUE_TO_GRAVITY))
    traction = fe.Constant((0.0, 0.0, 0.0))

    weak_form_lhs = fe.inner(sigma(u_trial), epsilon(v_test)) * fe.dx  # Crucial to use inner and not dot
    weak_form_rhs = (
        fe.dot(forcing, v_test) * fe.dx
        +
        fe.dot(traction, v_test) * fe.ds
    )

    # Compute solution
    u_solution = fe.Function(lagrange_vector_space_first_order)
    fe.solve(
        weak_form_lhs == weak_form_rhs,
        u_solution,
        dirichlet_clamped_boundary,
    )

    # Compute the von Mises stress
    deviatoric_stress_tensor = (
        sigma(u_solution)
        -
        1/3 * fe.tr(sigma(u_solution)) * fe.Identity(3)
    )
    von_Mises_stress = fe.sqrt(3/2 * fe.inner(deviatoric_stress_tensor, deviatoric_stress_tensor))

    lagrange_scalar_space_first_order = fe.FunctionSpace(
        mesh,
        "Lagrange",
        1,
    )
    von_Mises_stress = fe.project(von_Mises_stress, lagrange_scalar_space_first_order)

    # Write out fields for visualization with Paraview
    u_solution.rename("Displacement Vector", "")
    von_Mises_stress.rename("von Mises stress", "")

    beam_deflection_file = fe.XDMFFile("beam_deflection.xdmf")
    beam_deflection_file.parameters["flush_output"] = True
    beam_deflection_file.parameters["functions_share_mesh"] = True
    beam_deflection_file.write(u_solution, 0.0)
    beam_deflection_file.write(von_Mises_stress, 0.0)


if __name__ == "__main__":
    main()
