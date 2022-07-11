"""
Showcases an example of how to propagate derivative information over the root
finding process to a nonlinear system of equations.

For the (arbitrarily selected) residual function

    r = [
        x₀ + θ₀(x₀ - x₁)³ + θ₁

        θ₂(x₀ - x₁ - θ₃)³
    ]

that is parameterized by the vector θ = [θ₀, θ₁, θ₂, θ₃]ᵀ.

Invoking a root-finding algorithm yields (given the convergence of the
algorithm) a concrete solution vector x* = [x_0*, x_1*]. This solution
vector varies based on the value of the parameter vector θ.

The found solution vector enters a simple loss functional

    J = 0.5 * ||x||^2

We are now interested in gradient information

    ∂J/∂θ = [ dJ/dθ₀, dJ/dθ₁, dJ/dθ₂, dJ/dθ₃ ]ᵀ

For the subsequent gradient computations as well as the actual root
finding process, we need additional derivative information (denominator
layout)

    ∂r/∂x = [
        1 + 3θ₀(x₀ - x₁)²       -3θ₀(x₀ - x₁)²

        3θ₂(x₀ - x₁ - θ₃)²      -3θ₂(x₀ - x₁ - θ₃)²
    ]


    ∂r/∂θ = [
        (x₀ - x₁)³          1             0                     0

        0                   0             (x₀ - x₁ - θ₃)³       -3 θ₂ (x₀ - x₁ - θ₃)²
    ]


    ∂J/∂x = x

---

Finite Difference sensitivities

For each entry of the loss sensitivity vector dJ/dθ, we have to augment
the parameter vector θ and compute a classical solution. For example, for
the entry dJ/dθ_0

    θ_aug = [θ_0 + h, θ_1, θ_2, θ_3]

Then we compute the corresponding solution vector x*_aug by invoking
the root-finding algorithm.

Finally, we compute the approximation to the loss sensitivities by

    dJ/dθ_0 ≈ (J(x*_aug) - J(x*)) / h

---

Forward sensitivities

We first solve the original problem for x* using the concrete parameter
vector θ*. This is necessary because we can then evaluate all functions.

    (1) Solve (a batched) linear system for the solution sensitivities

        ∂r/∂x dx/dθ = - ∂r/∂θ

        Note that ∂r/∂x is 2x2 matrix, and ∂r/∂θ is a 2x4 matrix. Hence,
        we have to solve 4 linear systems. One for each parameter.
    
    (2) Evaluate the loss sensitivity vector dJ/dθ

        dJ/dθ = ∂J/∂x dx/dθ

---

Adjoint Sensitivities

Again, first solve x* using the concrete parameter vector θ*.

    (1) Solve one linear system for the adjoint variable λ

        (∂r/∂x)ᵀ λ = (∂J/∂x)ᵀ
    
    (2) Evaluate the loss sensitivities

        dJ/dθ = - λᵀ ∂r/∂θ
"""

import time

import numpy as np
from scipy import optimize

def main():
    def residual(x, theta):
        """
        r = [
            x₀ + θ₀(x₀ - x₁)³ + θ₁

            θ₂(x₀ - x₁ - θ₃)³
        ]
        """
        residual_value = np.array([
            x[0] + theta[0] * (x[0] - x[1])**3 + theta[1],
            theta[2] * (x[0] - x[1] - theta[3])**3
        ])

        return residual_value
    
    def del_r__del_x(x, theta):
        """
        ∂r/∂x = [
            1 + 3θ₀(x₀ - x₁)²       -3θ₀(x₀ - x₁)²

            3θ₂(x₀ - x₁ - θ₃)²      -3θ₂(x₀ - x₁ - θ₃)²
        ]
        """
        del_r__del_x__value = np.array([
            [1 + 3*theta[0]*(x[0] - x[1])**2, -3*theta[0] * (x[0] - x[1])**2],
            [3 * theta[2] * (x[0] - x[1] - theta[3])**2, -3*theta[2]*(x[0] - x[1] - theta[3])**2],
        ])

        return del_r__del_x__value
    
    def del_r__del_theta(x, theta):
        """
        ∂r/∂θ = [
            (x₀ - x₁)³          1             0                     0

            0                   0             (x₀ - x₁ - θ₃)³       -3 θ₂ (x₀ - x₁ - θ₃)²
        ]
        """
        del_r__del_theta__value = np.array([
            [(x[0] - x[1])**3, 1.0, 0.0, 0.0],
            [0.0, 0.0, (x[0] - x[1] - theta[3])**3, -3*theta[2]*(x[0] - x[1] - theta[3])**2],
        ])

        return del_r__del_theta__value
    
    def obtain_root(theta):
        initial_guess = np.array([0.0, 0.0])

        root_finding_result = optimize.root(
            fun=residual,
            x0=initial_guess,
            jac=del_r__del_x,
            args=(theta, ),
        )

        assert root_finding_result.success

        return root_finding_result.x
    
    theta_evaluation_point = np.array([1.0, -2.0, 1.0, 1.2])
    print(obtain_root(theta_evaluation_point))
    print(np.linalg.norm(residual(obtain_root(theta_evaluation_point), theta_evaluation_point)))

    def loss_functional(x):
        return 0.5 * np.linalg.norm(x)**2
    
    def del_J__del_x(x):
        return x
    
    print(loss_functional(obtain_root(theta_evaluation_point)))

    ### Finite Differences
    time_finite_differences = time.time_ns()

    step_length = 1.0e-6

    n_parameters = len(theta_evaluation_point)
    d_J__d_theta__finite_differences = np.zeros(n_parameters)

    loss_classical = loss_functional(obtain_root(theta_evaluation_point))

    for i in range(n_parameters):
        theta_augmented = theta_evaluation_point.copy()
        theta_augmented[i] += step_length

        d_J__d_theta__finite_differences[i] = (
            loss_functional(obtain_root(theta_augmented))
            -
            loss_classical
        ) / step_length

    time_finite_differences = time.time_ns() - time_finite_differences
    
    ### Forward Sensitivities
    time_forward_sensitivities = time.time_ns()

    x_classical = obtain_root(theta_evaluation_point)
    solution_sensitivities = np.linalg.solve(
        del_r__del_x(x_classical, theta_evaluation_point),
        - del_r__del_theta(x_classical, theta_evaluation_point)
    )
    d_J__d_theta__forward_sensitivities = del_J__del_x(x_classical).T @ solution_sensitivities

    time_forward_sensitivities = time.time_ns() - time_forward_sensitivities

    ### Adjoint Sensitivities
    time_adjoint_sensitivities = time.time_ns()

    x_classical = obtain_root(theta_evaluation_point)
    adjoint_variable = np.linalg.solve(
        del_r__del_x(x_classical, theta_evaluation_point).T,
        del_J__del_x(x_classical).T
    )
    d_J__d_theta__adjoint_sensitivities = - adjoint_variable.T @ del_r__del_theta(x_classical, theta_evaluation_point)

    time_adjoint_sensitivities = time.time_ns() - time_adjoint_sensitivities


    print()
    print("Finite Differences")
    print(time_finite_differences)
    print(d_J__d_theta__finite_differences)

    print()
    print("Forward Sensitivities")
    print(time_forward_sensitivities)
    print(d_J__d_theta__forward_sensitivities)

    print()
    print("Adjoint Sensitivities")
    print(time_adjoint_sensitivities)
    print(d_J__d_theta__adjoint_sensitivities)





if __name__ == "__main__":
    main()
