# Conditioning inside the solve (decisions 2, 5, 7 and 13): an equilibration of the damped system by a diagonal scale
# S = Diagonal(s). For a Hessian system the solve computes S(A + λD)S y = S b and returns x = S y; for a Jacobian
# system it factorizes J S, with the damping rows √(λ D_jj)·s_j, and returns x = S y. In exact arithmetic x is the
# unconditioned solution, so an iterator follows the same path with or without it, apart from rounding.
#
# The scale (decision 13): s_j is the power of two nearest, on a log scale, to 1/(1e-12 + √(d_j + λ₀·D_jj)), with d_j
# the diagonal of A as filled (for a Jacobian system the squared column norms of J), read in Float64, and λ₀·D the
# damping of the first solve after a fill (D = I, or the Marquardt scaling). It is computed once per fill, at that
# first solve, and reused for every later damping of the same fill; `zero!` and `markstale!` invalidate it. Powers of
# two make scaling exact: with no overflow or underflow, every floating-point operation of a factorization without
# pivoting gives exactly the scaled result, so what conditioning changes is the range of the entries (which matters in
# Float32) and the decisions made by comparing magnitudes (pivoting, rank and indefiniteness thresholds). `A`, `J`,
# `b` and `r` are never scaled: the solve scales while it copies into its scratch, forming each scaled entry in
# Float64 and narrowing it once, so that the first solve's scaled damped diagonal (about 1) cannot overflow.
#
# On for the dense and sparse Hessian systems and the dense and sparse Jacobian systems, in either element type; off
# for the static systems, and for the Krylov solvers, for which a diagonal scaling is a preconditioner and changes the
# iterates by more than rounding. Each conditioned system type has an internal switch, `condition`, set when it is
# built (`hessiansystem`, `jacobiansystem`), for tests only: nothing public reaches it.

const SQRT_HALF = sqrt(0.5)

# The `1e-12` of the scale, which bounds it (by about 2^40) where the damped diagonal is zero. A starting value
# (rules.md 26).
const CONDITIONING_EPS = 1e-12

# The scale of one variable from its damped diagonal entry `d = d_j + λ₀·D_jj`: 1/(1e-12 + √|d|) rounded to the nearest
# power of two (nearest on a log scale), 1 where d is not finite. A negative d (an indefinite A) is scaled by its
# magnitude, as a positive one is: with s = 1 there, while the positive entries are scaled to about 1, a negative pivot
# of a matrix whose entries are all small would look negligible to the indefiniteness test. It is at least
# floatmin(T), so that it is a normal number in the system's element type.
@inline function conditioning_scale(d::Real, ::Type{T}) where T
    dd = abs(Float64(d))
    isfinite(dd) || return 1.0
    f, e = frexp(1 / (CONDITIONING_EPS + sqrt(dd)))   # f in [0.5, 1)
    s = f >= SQRT_HALF ? ldexp(1.0, e) : ldexp(1.0, e - 1)
    return max(s, Float64(floatmin(T)))
end

# The damping of variable j, λ·D_jj, in Float64
@inline damping_entry(lambda::Float64, scaling, j) = isnothing(scaling) ? lambda : lambda * scaling[j]

# A later damping of the same fill uses the stored scale (SP-D9). Before each such solve, if for some variable the
# scaled damped diagonal |d_j + λ·D_jj|·s_j² lies outside [2^-16, 2^16], the scale is recomputed from the current
# damping, for that solve and those after it: going from undamped to heavily damped on a weak variable is the case it
# is for (with the stored scale its damping row or entry would be far larger than the others', which a relative rank,
# pivot or indefiniteness test reads as a defect, and in Float32 can overflow); an optimizer raising or lowering λ by
# a few factors of ten does not reach it. A variable outside the band whose scale computed now would be the same (a
# zero diagonal entry, scaled by the bound 1/1e-12 either way) does not count, so that such a variable does not
# recompute at every solve. `RESCALES` counts the recomputations, for tests and measurement.
#
# The bound 2^16 is a starting value (rules.md 26; improvements.md item 5).
const RESCALE_BOUND = 2.0^16
const RESCALES = Threads.Atomic{Int}(0)

# Whether the stored scale `s` has drifted from the damping now (`lambda`, `scaling`), the diagonal as filled read by
# `diag(j)` in Float64: one pass, allocating nothing
@inline function scale_drifted(diag::F, s::AbstractVector{Float64}, lambda::Float64, scaling, ::Type{T}) where {F, T}
    lambda = max(lambda, 0.0)
    @inbounds for j in eachindex(s)
        d = diag(j) + damping_entry(lambda, scaling, j)
        sj = s[j]
        v = abs(d) * (sj * sj)
        if !(inv(RESCALE_BOUND) <= v <= RESCALE_BOUND) && conditioning_scale(d, T) != sj
            return true
        end
    end
    return false
end
