using DampedLinearSolve

# Fills a Jacobian system as an optimizer's linearization does: `f(r, J, x)` writes the storage's r and J,
# then the storage forms b = Jᵀr and invalidates what it computed from the old J and r. A callable
# struct, so that an allocation test sees only the system's own work.
struct JacobianFill{F}
    f::F
end
function (t::JacobianFill)(b::AbstractVector, A, x)
    cost = t.f(getr(A), getJ(A), x)
    gradient!(b, A)
    markstale!(A)
    return cost
end
