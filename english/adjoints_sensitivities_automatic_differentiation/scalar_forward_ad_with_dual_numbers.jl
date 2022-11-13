"""
Implements simple (scalar) forward-mode Automatic Differentiation with
DualNumbers and operator overloading.
"""

using Base

struct DualNumber{T}
    real::T;
    dual::T;
end

DualNumber(x) = DualNumber(x, zero(x))

Base.:+(x::DualNumber, y::DualNumber) = DualNumber(x.real + y.real, x.dual + y.dual)

Base.:*(x::DualNumber, y::DualNumber) = DualNumber(x.real * y.real, x.real * y.dual + x.dual * y.real)
Base.:*(x, y::DualNumber) = DualNumber(x) * y

Base.sin(z::DualNumber) = DualNumber(sin(z.real), cos(z.real) * z.dual)

function pushforward(f, primal::Real, tangent::Real)
    input = DualNumber(primal, tangent)
    output = f(input)
    primal_out = output.real
    tangent_out = output.dual
    return primal_out, tangent_out
end

function derivative(f, x::Real)
    v = one(x)
    _, df_dx = pushforward(f, x, v)
    return df_dx
end


f(x) = x * x

x_point = 3.0

f(x_point)

f_prime(x) = 2.0 * x

f_prime(x_point)

derivative(f, x_point)

g(x) = 3.0 * x * x + sin(x)

g(x_point)

g_prime(x) = 6.0 * x + cos(x)
g_prime(x_point)

derivative(g, x_point)