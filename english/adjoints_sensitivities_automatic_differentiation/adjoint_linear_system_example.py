"""
Containing an example for obtaining the gradients through the implicit relation
given by a linear system.

Ax = b

with
J = 0.5 * (x_r - x)^T(x_r - x)

whereas x_r is a reference solution.
"""

import numpy as np
import time

if __name__ == "__main__":

    A = np.array([
        [10, 2, 1],
        [2, 5, 1],
        [1, 1, 3]
    ])

    ##### Creating a reference solution

    b_true = np.array([5, 4, 3])
    x_ref = np.linalg.solve(A, b_true)

    #### [A] Solve the classical system
    b_guess = np.ones(3)

    x = np.linalg.solve(A, b_guess)

    # Evaluate the loss function
    J = 0.5 * (x - x_ref).T @ (x - x_ref)

    #### [B] Obtaining gradients

    ### [3] Adjoint Sensitivities

    time_adjoint = time.time_ns()

    del_J__del_theta = np.zeros((1, 3))
    del_J__del_x = (x - x_ref).T
    d_b__d_theta = np.eye(3)

    # Solve adjoint system
    adjoint_variable = np.linalg.solve(A.T, del_J__del_x.T)

    # Plug in
    d_J__d_theta__adjoint = del_J__del_theta + adjoint_variable.T @ d_b__d_theta

    time_adjoint = time.time_ns() - time_adjoint

    ### [2] Forward Sensitivities

    time_forward = time.time_ns()

    del_J__del_theta = np.zeros((1, 3))
    del_J__del_x = (x - x_ref).T
    d_b__d_theta = np.eye(3)

    # Solve forward system
    d_x__d_theta = np.linalg.solve(A, d_b__d_theta)

    # Plug in
    d_J__d_theta__forward = del_J__del_theta + del_J__del_x @ d_x__d_theta

    time_forward = time.time_ns() - time_forward

    ### [1] Finite differences

    time_finite_differences = time.time_ns()

    eps = 1.0e-6

    d_J__d_theta__finite_difference = np.empty((1, 3))

    for i in range(3):
        b_augmented = b_guess.copy()
        b_augmented[i] += eps

        x_augmented = np.linalg.solve(A, b_augmented)
        J_augmented = 0.5 * (x_augmented - x_ref).T @ (x_augmented - x_ref)

        d_J__d_theta__finite_difference[0, i] = (J_augmented - J) / eps
    
    time_finite_differences = time.time_ns() - time_finite_differences
    
    #### Reporting the Results

    print("The loss function")
    print(J)

    print()

    np.set_printoptions(precision=16)
    print("Sensitivities by the Adjoint Method")
    print(d_J__d_theta__adjoint)
    print("Sensitivities by the Forward Method")
    print(d_J__d_theta__forward)
    print("Sensitivities by Finite Differences")
    print(d_J__d_theta__finite_difference)

    print()

    print("Time of the approaches [ns] - lower is better")
    print("Sensitivities by the Adjoint Method")
    print(time_adjoint)
    print("Sensitivities by the Forward Method")
    print(time_forward)
    print("Sensitivities by Finite Differences")
    print(time_finite_differences)

