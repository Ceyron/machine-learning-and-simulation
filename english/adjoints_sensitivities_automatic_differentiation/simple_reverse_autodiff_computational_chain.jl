f(x) = exp(sin(sin(x)))

f(2.0)

f_prime(x) = exp(sin(sin(x))) * cos(sin(x)) * cos(x)

f_prime(2.0)

(f(2.0 + 1e-8) - f(2.0)) / 1e-8


function backprop_rule(::typeof(sin), x)
    y = sin(x)

    function sin_pullback(y_cotangent)
        x_cotangent = y_cotangent * cos(x)
        return x_cotangent
    end

    return y, sin_pullback
end

function backprop_rule(::typeof(exp), x)
    y = exp(x)

    function exp_pullback(y_cotangent)
        x_cotangent = y_cotangent * y
        return x_cotangent
    end

    return y, exp_pullback
end

function vjp(chain, primal)
    pullback_stack = []
    current_value = primal

    # Primal Pass
    for operation in chain
        current_value, current_pullback = backprop_rule(operation, current_value)
        push!(pullback_stack, current_pullback)
    end

    function pullback(cotangent)
        current_cotangent = cotangent
        for back in reverse(pullback_stack)
            current_cotangent = back(current_cotangent)
        end
        return current_cotangent
    end

    return current_value, pullback
end

out, back = vjp([sin, sin, exp], 2.0)

back(1.0)

function val_and_grad(chain, x)
    y, pullback = vjp(chain, x)
    derivative = pullback(1.0)
    return y, derivative
end

val_and_grad([sin, sin, exp], 2.0)
f(2.0), f_prime(2.0)