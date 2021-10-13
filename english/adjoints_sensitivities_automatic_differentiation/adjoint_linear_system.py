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
import matplotlib.pyplot as plt

A = np.array([
    [10, 2, 1],
    [ 2, 5, 1],
    [ 1, 1, 3]
])
b_true = np.array([5, 4, 3]) # Just for generating a reference solution

x_ref = np.linalg.solve(A, b_true)

# Start with a current guess of the rhs side
b_guess = np.ones(3)

# Solve the forward problem
x = np.linalg.solve(A, b_guess)

# Evaluate the loss function
J = 0.5 * (x - x_ref).T @ (x - x_ref)

####
# Backward / Adjoint Sensitivity Analysis
####

adjoint_time = time.time_ns()

# We derived that
# d_J__d_theta = del_J__del_theta + lambda^T @ d_b__d_theta
#
# Here:
# del_J__del_theta = 0^T   (because there is no EXPLICIT dependency of J on theta)
# d_b__d_theta = eye(3)   (because theta_0, theta_1, theta_2 are the three entries of the b vector)

del_J__del_theta = np.zeros((1, 3))
d_b__d_theta = np.eye(3)

# The adjoint variable is obtained from the system
# A^T @ lambda = del_J__del_x
#
# Here:
# del_J__del_x = x - x_r

del_J__del_x__transposed = x - x_ref

adjoint_variable = np.linalg.solve(A.T, del_J__del_x__transposed)

d_J__d_theta__adjoint = del_J__del_theta + adjoint_variable.T @ d_b__d_theta


adjoint_time = time.time_ns() - adjoint_time


####
# Forward Sensitivity Analysis
####

forward_time = time.time_ns()

# We derived that
# d_J__d_theta = del_J__del_theta + del_J__del_x @ d_x__d_theta
#
# Here:
# del_J__del_theta = 0^T
# del_J__del_x = (x - x_r)^T

del_J__del_theta = np.zeros((1, 3))
del_J__del_x = (x - x_ref).T

# The solution sensitivities are given as the solution of the system
# A @ d_x__d_theta = d_b__d_theta
#
# Here:
# d_b__d_theta = eye(3)

d_b__d_theta = np.eye(3)

d_x__d_theta = np.linalg.solve(A, d_b__d_theta)

d_J__d_theta__forward = del_J__del_theta + del_J__del_x @ d_x__d_theta


forward_time = time.time_ns() - forward_time

####
# Finite Differences
####

finite_difference_time = time.time_ns()

eps = 1.0e-6

d_J__d_theta__finite_differences = np.empty((1, 3))

for i in range(3):
    b_augmented = b_guess.copy()
    b_augmented[i] += eps

    x_augmented = np.linalg.solve(A, b_augmented)
    J_augmented = 0.5 * (x_augmented - x_ref).T @ (x_augmented - x_ref)
    
    d_J__d_theta__finite_differences[0, i] = (J_augmented - J) / eps


finite_difference_time = time.time_ns() - finite_difference_time


np.set_printoptions(precision=16)

print("The loss function")
print(J)

print()

print("Sensitivities by Adjoint Method")
print(d_J__d_theta__adjoint)
print("Sensitivities by Forward Method")
print(d_J__d_theta__forward)
print("Sensitivities by Finite Differences")
print(d_J__d_theta__finite_differences)

print()

print("Times in ns [lower is better]")

print(adjoint_time)
print(forward_time)
print(finite_difference_time)
