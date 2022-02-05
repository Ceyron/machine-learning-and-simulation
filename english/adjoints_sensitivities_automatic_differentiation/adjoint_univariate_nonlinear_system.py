"""
An example script for calculating the adjoint of a nonlinear equation. The
example is taken from nonlinear mechanics.


                  /|O |
                -o |  |
               /  \|O |
              /       |
             /        |
            /         |
|/\/\/\---o-          |
         / \          |
        ----          |
        O   O         |
--------------        |
                      |

Two rollers are located on solid walls. They are rotated by 90 degrees and
connected by a rigid lever.        
The bottom left roller is connected horizontally to a solid wall by a spring.

A downward pointing force is applied to the top right roller, moving it
downwards and consequentially moving the horizontal roller to the left,
compressing the spring. This system experiences geometric nonlinearities due to
the large displacements that can occur.

The parameters to the model are: a: The vertical distance between the two
rollers b: The horizontal distance between the two rollers k: The spring
stiffness

If one sets a=1, b=1, k=200 the relation between displacement and force is the
following:

  50 +----------------------------------------------------------------------+
     |                                                              *       |
  40 |                                                             *        |
     |                                                             *        |
  30 |                                                             *        |
     |                ********                                     *        |
  20 |              ***      ****                                 *         |
     |             **            **                               *         |
  10 |            **               **                             *         |
     |            *                  **                          *          |
   0 |           *                     **                       *           |
     |           *                       **                     *           |
     |          *                          **                  *            |
 -10 |         *                             **               **            |
     |         *                               **            **             |
 -20 |         *                                 ****      ***              |
     |         *                                    ********                |
 -30 |        *                                                             |
     |        *                                                             |
 -40 |        *                                                             |
     |        *                                                             |
 -50 +----------------------------------------------------------------------+
   -0.5          0          0.5          1         1.5          2          2.5

This function is based on the residual equation: r(F, v) = k(a-v)*(a-b/(sqrt(b^2
+ 2av -v^2))) - F  !=! 0

r(F, v; a, b, k)
=
                
               ⎛           b              ⎞
-F + k⋅(a - v)⋅ ⎜- ──────────────────── + 1⎟
               ⎜     _________________    ⎟
               ⎜    ╱          2    2     ⎟
               ⎝  ╲╱  2⋅a⋅v + b  - v      ⎠


which can either be made explicit for F(v) or be solved for v(F) by a
Newton-Raphson algorithm (where one has to take to adjust the initial conditions
to differentiate between the three regions split up by the two extrema of the
function)

In this script, we want to first calculate a reference solution with known
parameters a, b and k. Then, we will assume they are unknown and are part of
parameter vector theta: theta_0 = a theta_1 = b theta_2 = k

Then we want to calculate sensitivities of a loss function J wrt to the parameter vector.
"""

import time

import numpy as np
from scipy import optimize

a_true = 1.0
b_true = 1.0
k_true = 200.0


def residuum(displacement, force, vertical_distance, horizontal_distance, spring_stiffness):
    r"""
                    /           b              \
     -F + k*(a - v)*|- -------------------- + 1|
                    |     _________________    |
                    |    /          2    2     |
                    \  \/  2*a*v + b  - v      /
    """
    residual_value = (
        spring_stiffness * (vertical_distance - displacement) * (
            1
            -
            (
                horizontal_distance
            ) / (
                np.sqrt(horizontal_distance**2 + 2*vertical_distance*displacement - displacement**2)
            )
        )
        -
        force
    )

    return residual_value


def del_residuum__del_displacement(displacement, force, vertical_distance, horizontal_distance, spring_stiffness):
    r"""
      b*k*(-a + v)*(a - v)     /           b              \
    - -------------------- - k*|- -------------------- + 1|
                       3/2     |     _________________    |
      /         2    2\        |    /          2    2     |
       \2*a*v + b  - v /        \  \/  2*a*v + b  - v      /
    """
    del_residual_value__del_displacement = spring_stiffness * (
        (
            horizontal_distance
        ) / (
            np.sqrt(horizontal_distance**2 + 2*vertical_distance*displacement - displacement**2)
        )
        +
        (
            horizontal_distance * (vertical_distance - displacement)**2
        ) / (
            np.sqrt(horizontal_distance**2 + 2*vertical_distance*displacement - displacement**2)**3
        )
        -
        1.0
    )

    return del_residual_value__del_displacement


def del_residuum__del_vertical_distance(displacement, force, vertical_distance, horizontal_distance, spring_stiffness):
    r"""
       b*k*v*(a - v)         /           b              \
    -------------------- + k*|- -------------------- + 1|
                     3/2     |     _________________    |
    /         2    2\        |    /          2    2     |
    \2*a*v + b  - v /        \  \/  2*a*v + b  - v      /
    """
    del_residual_value__del_vertical_distance = (
        (
            horizontal_distance * spring_stiffness * displacement * (vertical_distance - displacement)
        ) / (
            np.sqrt(2*vertical_distance*displacement + horizontal_distance**2 - displacement**2)**3
        )
        +
        spring_stiffness * (
            1.0
            -
            (
                horizontal_distance
            ) / (
                np.sqrt(2*vertical_distance*displacement + horizontal_distance**2 - displacement**2)
            )
        )
    )

    return del_residual_value__del_vertical_distance


def del_residuum__del_horizontal_distance(displacement, force, vertical_distance, horizontal_distance, spring_stiffness):
    r"""
              /          2                                \
              |         b                      1          |
    k*(a - v)*|-------------------- - --------------------|
              |                 3/2      _________________|
              |/         2    2\        /          2    2 |
              \\2*a*v + b  - v /      \/  2*a*v + b  - v  /
    """
    del_residual_value__del_horizontal_distance = (
        spring_stiffness * (vertical_distance - displacement) * (
            (
                horizontal_distance**2
            ) / (
                np.sqrt(2*vertical_distance*displacement + horizontal_distance**2 - displacement**2)**3
            )
            -
            (
                1.0
            ) / (
                np.sqrt(2*vertical_distance*displacement + horizontal_distance**2 - displacement**2)
            )
        )
    )

    return del_residual_value__del_horizontal_distance


def del_residuum__del_spring_stiffness(displacement, force, vertical_distance, horizontal_distance, spring_stiffness):
    r"""
            /           b              \
    (a - v)*|- -------------------- + 1|
            |     _________________    |
            |    /          2    2     |
            \  \/  2*a*v + b  - v      /
    """
    del_residual_value__del_spring_stiffness = (
        (vertical_distance - displacement) * (
            1.0
            -
            (
                horizontal_distance
            ) / (
                np.sqrt(2*vertical_distance*displacement + horizontal_distance**2 - displacement**2)
            )
        )
    )

    return del_residual_value__del_spring_stiffness

if __name__ == "__main__":

    ####
    # Creating a reference solution
    ####

    # The reference solution will be the displacement at a force of 10N, located in
    # the second region of downward going curve. We solve this by a Newton-Raphson scheme
    force_value = 10.0
    additional_args = (force_value, a_true, b_true, k_true)

    v_ref = optimize.newton(
        func=residuum,
        fprime=del_residuum__del_displacement,
        args=additional_args,
        x0=1.0
    )


    ####
    # Solving the forward problem
    ####
    theta_guess = np.array([0.9, 0.9, 180.0])
    additional_args = (force_value, *theta_guess)

    time_forward_problem = time.time_ns()

    v = optimize.newton(
        func=residuum,
        fprime=del_residuum__del_displacement,
        args=additional_args,
        x0=1.0,
    )

    time_forward_problem = time.time_ns() - time_forward_problem

    # The "J" loss function is the least-squares (quadratic loss)
    def loss_function(v):
        return 0.5 * (v - v_ref)**2

    J = loss_function(v)


    ######
    # Adjoint Method for sensitivities
    ######

    # We derived that
    # d_J__d_theta = del_J__del_theta + lambda^T @ del_f__del_theta
    #
    # Note: f = r in this case (i.e. the residuum is the non-linear equation to solve)
    #
    # Important: Despite having a partial derivative with del_f__del_theta, this actually
    # means a total derivative, but WITHOUT d_x__d_theta! (Hence, Automatic Differentiation
    # can be applied in a straight-forward fashion)
    #
    # Here:
    # del_J__del_theta = 0
    # del_f__del_theta = [
    #       d_f__d_a,
    #       d_f__d_b,
    #       d_f__d_k
    # ]^T
    # (Note: By definition the gradient is a row-vector)

    time_adjoint_sensitivities = time.time_ns()

    del_J__del_theta = np.zeros((1, 3))
    current_args_adjoint = (v, force_value, *theta_guess)
    del_f__del_theta = np.array([
        del_residuum__del_vertical_distance(*current_args_adjoint),
        del_residuum__del_horizontal_distance(*current_args_adjoint),
        del_residuum__del_spring_stiffness(*current_args_adjoint),
    ]).reshape((1, -1))

    # We acquire the adjoint solution by solving the (linear) adjoint problem
    # del_f__del_x^T @ lambda = del_J__del_x
    #
    # Note: x = v in this case (our quantity we are solving for is the displacement)
    #
    # Here:
    # del_f__del_x = del_r__del_v (i.e. the derivative of the residuum wrt. the
    # displacement, we needed this expression anyways for our first-order Newton
    # root solver)
    # del_J__del_x = (v - v_r)  (because we have a quadratic loss)

    del_f__del_x = del_residuum__del_displacement(*current_args_adjoint)
    del_J__del_x = (v - v_ref)

    # Actual we would need to solve a linear system here, i.e. using
    # np.linalg.solve. However, since the original main quantity (v = displacement)
    # is a scalar quantity, the adjoint variable is also a scalar and we can find a
    # closed-form solution to this scalar linear equation easily
    #
    # This also means we do not have to care about transposition on the Jacobian
    # del_f__del_x
    adjoint_variable = - 1.0/del_f__del_x * del_J__del_x


    # Finally, evaluating the loss function sensitivity
    #
    # Also here: no need for transposing lambda, as it is a scalar quantity
    d_J__d_theta_adjoint = del_J__del_theta + adjoint_variable * del_f__del_theta

    time_adjoint_sensitivities = time.time_ns() - time_adjoint_sensitivities

    #####
    # Forward Sensitivity Analysis
    #####

    time_forward_sensitivities = time.time_ns()

    # We derived that
    # d_J__d_theta = del_J_del_theta + del_J__del_x @ d_x__d_theta
    #
    # Here:
    # del_J__del_theta = 0^T
    # del_J__del_x = (x - x_r)

    del_J__del_theta = np.zeros((1, 3))
    del_J__del_x = v - v_ref

    # We obtain the solution sensitivities d_x__d_theta by solving (linear)
    # auxillary systems
    #
    # del_f__del_x @ d_x__d_theta = - del_f__del_theta
    # 
    # for d_x__d_theta
    #
    # Here:
    # del_f__del_x = del_r__del_v
    # del_f__del_theta = [
    #       del_r__del_a,
    #       del_r__del_b,
    #       del_r__del_k,
    # ]^T

    current_args_forward = (v, force_value, *theta_guess)
    del_f__del_x = del_residuum__del_displacement(*current_args_adjoint)
    del_f__del_theta = np.array([
        del_residuum__del_vertical_distance(*current_args_forward),
        del_residuum__del_horizontal_distance(*current_args_forward),
        del_residuum__del_spring_stiffness(*current_args_forward),
    ]).reshape((1, -1))

    # Actually we would have to solve a linear system (or since we are doing the
    # forward sensitivity analysis multiple linear systems) here, but since the
    # system matrix is a scalar, this can be explicitly solve by dividing by that scalar
    d_x__d_theta = - 1.0/del_f__del_x * del_f__del_theta

    # Finally, compute the loss function sensitivities (would need @ instead of *
    # for matrix multiplication)
    d_J__d_theta_forward = del_J__del_theta + del_J__del_x * d_x__d_theta

    time_forward_sensitivities = time.time_ns() - time_forward_sensitivities

    #####
    # Finite differences for sensitivities
    #####

    time_finite_difference_sensitivities = time.time_ns()

    eps = 1.0e-6
    d_J__d_theta_finite_differences = np.empty((1, 3))

    # Solve three additional forward problems (NOT forward sensitivity problems,
    # these are still non-linear problems)

    for i in range(3):
        theta_augmented = theta_guess.copy()
        theta_augmented[i] += eps

        additional_args_finite_differences = (force_value, *theta_augmented)
        v_augmented = optimize.newton(
            func=residuum,
            fprime=del_residuum__del_displacement,
            args=additional_args_finite_differences,
            x0=1.0,
        )
        J_augmented = loss_function(v_augmented)

        d_J__d_theta_finite_differences[0, i] = (J_augmented - J) / eps


    time_finite_difference_sensitivities = time.time_ns() - time_finite_difference_sensitivities

    ####### Pretty print solutions

    print("Reference solution")
    print(v_ref)
    print("Solution at current parameter estimates")
    print(v)
    print("Value of quadratic loss")
    print(J)
    np.set_printoptions(precision=16)
    print("Gradient by Adjoint Method")
    print(d_J__d_theta_adjoint)
    print("Gradient by Forward Method")
    print(d_J__d_theta_forward)
    print("Gradient by Finite Differences")
    print(d_J__d_theta_finite_differences)

    print()
    print("Time for the forward problem")
    print(time_forward_problem)

    print()
    print("Time for the adjoint sensitivities")
    print(time_adjoint_sensitivities)
    print("Time for the forward sensitivities")
    print(time_forward_sensitivities)
    print("Time for the Finite Difference Sensitivities")
    print(time_finite_difference_sensitivities)