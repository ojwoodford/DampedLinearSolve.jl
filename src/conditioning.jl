# Conditioning inside the solve (decisions 2, 3, 5 and 7): an equilibration of the damped system by a diagonal scale
# S = Diagonal(s). For a Hessian system the solve computes S(A + λD)S y = S b and returns x = S y; for a Jacobian
# system it factorizes J S, with the damping rows √(λ D_jj)·s_j, and returns x = S y. In exact arithmetic x is the
# unconditioned solution, so an iterator follows the same path with or without it, apart from rounding.
#
# s_j = 1/(1 + √d_j), with d_j the diagonal of A as filled (for a Jacobian system the squared column norms of J), read in
# Float64, rounded to the nearest power of two so that scaling is exact: with no overflow or underflow, every
# floating-point operation of a factorization without pivoting gives exactly the scaled result, so what conditioning
# changes is the range of the entries (which matters in Float32) and the decisions made by comparing magnitudes
# (pivoting, rank and indefiniteness thresholds). `A`, `J`, `b` and `r` are never scaled: the solve scales while it
# copies into its scratch.
#
# On for the dense and sparse Hessian systems and the dense and sparse Jacobian systems, in either element type; off
# for the static systems, and for the Krylov solvers, for which a diagonal scaling is a preconditioner and changes the
# iterates by more than rounding. Each conditioned system type has an internal switch, `condition`, set when it is
# built (`hessiansystem`, `jacobiansystem`), for tests only: nothing public reaches it.

const SQRT_HALF = sqrt(0.5)

# The scale of one variable from its diagonal entry `d`: 1/(1 + √d) rounded to the nearest power of two (nearest on a
# log scale), 1 where d is negative or not finite. It is at least floatmin(T), so that it is a normal number in the
# system's element type.
@inline function conditioning_scale(d::Real, ::Type{T}) where T
    dd = Float64(d)
    (dd >= 0 && isfinite(dd)) || return 1.0
    f, e = frexp(1 / (1 + sqrt(dd)))   # f in [0.5, 1)
    s = f >= SQRT_HALF ? ldexp(1.0, e) : ldexp(1.0, e - 1)
    return max(s, Float64(floatmin(T)))
end
