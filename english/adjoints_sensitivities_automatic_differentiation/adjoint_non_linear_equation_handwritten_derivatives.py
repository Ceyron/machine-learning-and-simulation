"""
Showcases different sensitivity methods for gradient propagation over the
solution to non-linear equations.

Assume the residual function

    r(u, θ) = u² - sin(θ)

is solved for u by a root finding method. (For simplicity, we assume
this will always work fine).

The resulting solution u* associated with θ* enters a loss functional

    J(u) = u²

We now seek gradient (here just derivative) information

    dJ/dθ

e.g. for usage in gradient-based optimization.

For that assume we have a (hypothetical) function f that maps from θ to u,
i.e. it solves the root finding problem.

---

Finite Differences

    dJ/dθ ≈ (J(f(θ* + ε)) - J(f(θ*)) / ε

---

Forward sensitivities

    (1) Solve classical problem

        u* = f(θ*)

    (2) Obtain solution sensitivities (all evaluated at u* and θ*)

        du/dθ = - (∂r/∂u)^(-1) * ∂r/∂θ
    
    (3) Obtain loss sensitivities (all evaluated at u* and θ*)

        dJ/dθ = ∂J/∂u * du/dθ

---

Adjoint Sensitivities

    (1) Solve classical problem

        u* = f(θ*)

    (2) Obtain adjoint variable (all evaluated at u* and θ*)

        λ = (∂r/∂u)^(-1) * ∂J/∂u

    (3) Obtain loss sensitivities (all evaluated at u* and θ*)
    
        dJ/dθ = λ * ∂r/∂θ

"""

import numpy as np
from scipy import optimize

def main():
    def residual(u, theta):
        return u**2 - np.sin(theta)
    
    def del_residual__del_u(u, theta):
        return 2*u
    
    def del_residual__del_theta(u, theta):
        return - np.cos(theta)
    
    def obtain_root(theta):
        u_0 = 1.0 # A wild guess. Usually, the initial guess is crucial.
        root_finding_result = optimize.root_scalar(
            f=lambda u: residual(u, theta=theta),
            x0=u_0,
            fprime=lambda u: del_residual__del_u(u, theta=theta),
        )

        assert root_finding_result.converged

        return root_finding_result.root
    
    def loss_functional(u):
        return u**2
    
    def del_J__del_u(u):
        return 2*u
    
    
    evaluation_point = 1.0  # One particular theta value
    u_at_evaluation_point = obtain_root(evaluation_point)
    J_at_evaluation_point = loss_functional(u_at_evaluation_point)

    print(f"u_at_evaluation_point={u_at_evaluation_point:1.6f}")
    print(f"J_at_evaluation_point={J_at_evaluation_point:1.6f}")

    ### Finite Differences
    epsilon = 1.0e-6
    d_J__d_theta__finite_differences = (
        loss_functional(obtain_root(evaluation_point + epsilon))
        -
        loss_functional(obtain_root(evaluation_point))
    ) / epsilon

    print(f"{d_J__d_theta__finite_differences:1.16f}: Finite Differences derivative")

    ### Forward Sensitivities
    u_classical = obtain_root(evaluation_point)
    solution_sensitivity = (
        -
        1.0 / del_residual__del_u(u_classical, theta=evaluation_point)
        *
        del_residual__del_theta(u_classical, theta=evaluation_point)
    )
    d_J__d_theta__forward = (
        del_J__del_u(u_classical)
        *
        solution_sensitivity
    )
    print(f"{d_J__d_theta__forward:1.16f}: Forward derivative")

    ### Adjoint Sensitivities
    u_classical = obtain_root(evaluation_point)
    adjoint_variable = (
        1.0 / del_residual__del_u(u_classical, theta=evaluation_point)
        *
        del_J__del_u(u_classical)
    )
    d_J__d_theta__adjoint = (
        -
        adjoint_variable
        *
        del_residual__del_theta(u_classical, theta=evaluation_point)
    )
    print(f"{d_J__d_theta__adjoint:1.16f}: Adjoint derivative")


if __name__ == "__main__":
    main()