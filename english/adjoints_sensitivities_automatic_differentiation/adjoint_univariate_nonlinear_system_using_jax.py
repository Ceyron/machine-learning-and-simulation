"""
Similar to the file of adjoint_univariate_nonlinear_system.py but here we use
the Automatic Differentiation of Jax to obtain the gradients of the nonlinear
function.
"""

import time 

import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad

from scipy import optimize


def residuum(displacement, force, theta):
    r"""
                    /           b              \
     -F + k*(a - v)*|- -------------------- + 1|
                    |     _________________    |
                    |    /          2    2     |
                    \  \/  2*a*v + b  - v      /
    """
    vertical_distance, horizontal_distance, spring_stiffness = theta
    residual_value = (
        spring_stiffness * (vertical_distance - displacement) * (
            1
            -
            (
                horizontal_distance
            ) / (
                jnp.sqrt(horizontal_distance**2 + 2*vertical_distance*displacement - displacement**2)
            )
        )
        -
        force
    )

    return residual_value

if __name__ == "__main__":
    a_true = 1.0
    b_true = 1.0
    k_true = 200.0

    force_value = 10.0

    v_ref = optimize.newton(
        func=residuum,
        fprime=grad(residuum),
        args=(force_value, (a_true, b_true, k_true)),
        x0=1.0
    )

    # print(v_ref)

    ######
    # Solving the Forward Problem
    ######

    theta_guess = jnp.array([0.9, 0.9, 180.0])
    additional_args = (force_value, theta_guess)

    v = optimize.newton(
        func=residuum,
        fprime=grad(residuum),
        args=additional_args,
        x0=1.0,
    )

    # The "J" loss function is the least-squares (quadratic loss)
    def loss_function(v, theta):
        return 0.5 * (v - v_ref)**2

    J = loss_function(v, theta_guess)


    ##### Adjoint Method

    time_adjoint = time.time_ns()

    current_args_adjoint = (v, force_value, theta_guess)

    del_J__del_theta = grad(loss_function, argnums=1)(v, theta_guess).reshape((1, -1))
    del_J__del_x = grad(loss_function, argnums=0)(v, theta_guess)
    del_f__del_theta = grad(residuum, argnums=2)(*current_args_adjoint).reshape((1, -1))
    del_f__del_x = grad(residuum, argnums=0)(*current_args_adjoint)

    # print(del_J__del_theta)
    # print(del_J__del_x)
    # print(del_f__del_theta)
    # print(del_f__del_x)

    adjoint_variable = -1.0 / del_f__del_x * del_J__del_x

    d_J__d_theta_adjoint = del_J__del_theta + adjoint_variable * del_f__del_theta

    time_adjoint = time.time_ns() - time_adjoint

    print("Adjoint Sensitivities using Autodiff for gradients and Jacobians")
    print(d_J__d_theta_adjoint)
    print("Timing")
    print(time_adjoint)


