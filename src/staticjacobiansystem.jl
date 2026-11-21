# The static residual-and-Jacobian linear system, for small problems (TP4): J an MMatrix{M, N}, r an MVector{M},
# solved as the dense Jacobian system is (a QR of J once per linearization, then per damping value a QR of the
# stacked [R; √λ D^½]), with StaticArrays' QR. Chosen at construction, without a user option, when the
# number of variables is static (a `StaticInt`: the caller decides) and M·N is at most MAX_STATIC_JACOBIAN.

"""
    MAX_STATIC_JACOBIAN = 48

The largest `m·n` (residuals times variables) for which a Jacobian system with a static number of variables
(`n` a `StaticInt`) and a dense Jacobian
(an integer `m` or a `Matrix` prototype) is a static Jacobian system. A starting value (rules.md 26): measured 2026-11-15 against the dense Jacobian system on
Julia 1.10 and 1.13, the static system was faster, per linearization and per retry, at every measured size with
`m·n ≤ 48` (`n` from 1 to 6), and slower somewhere at every larger `m·n` measured (`n = 8, m = 8` already
retries 1.26 times slower on 1.10).
"""
const MAX_STATIC_JACOBIAN = 48

# Whether n variables and an m-residual dense Jacobian get the static Jacobian system: n static, a QR solver,
# and m·n small
usesstaticjacobian(m::Integer, n, linsolver) = n isa StaticInt && needsqr(linsolver) && m * dynamic(n) <= MAX_STATIC_JACOBIAN

"""
    StaticJacobianStorage

The Hessian storage of a [`StaticJacobianLinearSystem`](@ref): `J` (an `MMatrix{M, N}`) and `r` (an
`MVector{M}`), standing for `A = JᵀJ`, which is never formed.
"""
struct StaticJacobianStorage{M, N, MN, T}
    J::MMatrix{M, N, T, MN}
    r::MVector{M, Float64}
    colnorm2::MVector{N, Float64}     # squared column norms of J, the diagonal of JᵀJ
    stale::Base.RefValue{Bool}         # the factorization (R, q) is out of date
    normsstale::Base.RefValue{Bool}
end
StaticJacobianStorage(J::MMatrix{M, N}) where {M, N} = StaticJacobianStorage(J, zeros(MVector{M, Float64}), zeros(MVector{N, Float64}), Ref(true), Ref(true))

Base.size(A::StaticJacobianStorage{M, N}) where {M, N} = (N, N)
Base.size(A::StaticJacobianStorage, d::Integer) = size(A)[d]
markstale!(A::StaticJacobianStorage) = (A.stale[] = true; A.normsstale[] = true; A)
getJ(A::StaticJacobianStorage) = A.J
getr(A::StaticJacobianStorage) = A.r

function jmul!(y::AbstractVector, J::MMatrix, v::AbstractVector)
    m, n = size(J)
    @inbounds for i in 1:m
        total = 0.0
        for j in 1:n
            total += Float64(J[i, j]) * v[j]
        end
        y[i] = total
    end
    return y
end
function jtr!(g::AbstractVector, J::MMatrix, r::AbstractVector)
    m, n = size(J)
    @inbounds for j in 1:n
        total = 0.0
        for i in 1:m
            total += Float64(J[i, j]) * r[i]
        end
        g[j] = total
    end
    return g
end

function quadform(A::StaticJacobianStorage{M, N}, v::AbstractVector) where {M, N}
    J = A.J
    total = 0.0
    @inbounds for i in 1:M
        s = 0.0
        for j in 1:N
            s += Float64(J[i, j]) * v[j]
        end
        total += s * s
    end
    return total
end

function colnorms2!(A::StaticJacobianStorage{M, N}) where {M, N}
    if A.normsstale[]
        @inbounds for j in 1:N
            total = 0.0
            for i in 1:M
                Jij = Float64(A.J[i, j])
                total += Jij * Jij
            end
            A.colnorm2[j] = total
        end
        A.normsstale[] = false
    end
    return A.colnorm2
end
function max_diag(A::StaticJacobianStorage)
    md = 0.0
    for d in colnorms2!(A)
        md = max(md, d)
    end
    return md
end

"""
    StaticJacobianLinearSystem{M, N, MN, K, KN}

The Jacobian system of [`DenseJacobianLinearSystem`](@ref), with `J` an `MMatrix{M, N}` and the QR factorizations done
by StaticArrays, for problems with `M·N ≤ MAX_STATIC_JACOBIAN`. Same solves, statuses and damping floor. `R` and
`q` (`K = min(M, N)` rows) are kept per linearization.
"""
mutable struct StaticJacobianLinearSystem{M, N, MN, K, KN, T} <: AbstractJacobianLinearSystem
    const A::StaticJacobianStorage{M, N, MN, T}
    const b::MVector{N, Float64}
    const x::MVector{N, Float64}
    const R::MMatrix{K, N, T, KN}
    const q::MVector{K, T}
    nonfinite::Bool
    lambda::Float64
    scaling::Union{Nothing, MVector{N, Float64}, Vector{Float64}}
end
function StaticJacobianLinearSystem(J::MMatrix{M, N, T}) where {M, N, T}
    K = min(M, N)
    return StaticJacobianLinearSystem{M, N, M * N, K, K * N, T}(StaticJacobianStorage(J), zeros(MVector{N, Float64}), zeros(MVector{N, Float64}),
                                                        zeros(MMatrix{K, N, T, K * N}), zeros(MVector{K, T}), false, 0.0, nothing)
end
isfinitesystem(ls::StaticJacobianLinearSystem) = all(isfinite, ls.A.J) && all(isfinite, ls.A.r)
StaticJacobianLinearSystem(J::Matrix{T}) where T = StaticJacobianLinearSystem(MMatrix{size(J, 1), size(J, 2), T}(J))

zero!(ls::StaticJacobianLinearSystem) = (fill!(ls.b, 0.0); markstale!(ls.A); ls)

function copy_structure(ls::StaticJacobianLinearSystem{M, N, MN, K, KN, T}) where {M, N, MN, K, KN, T}
    return StaticJacobianLinearSystem{M, N, MN, K, KN, T}(StaticJacobianStorage(copy(ls.A.J)), zeros(MVector{N, Float64}), ls.x,
                                                  zeros(MMatrix{K, N, T, KN}), zeros(MVector{K, T}), false, ls.lambda, ls.scaling)
end
sharex(ls::StaticJacobianLinearSystem{M, N, MN, K, KN, T}, from) where {M, N, MN, K, KN, T} = ls.x === from.x ? ls :
    StaticJacobianLinearSystem{M, N, MN, K, KN, T}(ls.A, ls.b, from.x, ls.R, ls.q, ls.nonfinite, ls.lambda, ls.scaling)

gradient!(g::AbstractVector, A::StaticJacobianStorage) = jtr!(g, A.J, A.r)

# R and q = (Qᵀr)[1:K] from the thin QR of J
function factor_static!(ls::StaticJacobianLinearSystem{M, N, MN, K}) where {M, N, MN, K}
    A = ls.A
    A.stale[] = false
    ls.nonfinite = !(all(isfinite, A.J) && all(isfinite, A.r))
    ls.nonfinite && return nothing
    J = SMatrix(A.J)
    F = qr(J)
    ls.R .= F.R
    ls.q .= F.Q' * SVector{M, eltype(J)}(A.r)   # r narrowed to the system's element type
    return nothing
end

# Numerical full rank of the N×N upper triangle of R, as `fullrank_upper` decides for the dense system
@inline function static_fullrank(R::StaticMatrix, rows::Integer, n::Integer)
    tol = max(rows, n) * eps(Float64)
    maxabs, minabs = 0.0, Inf
    @inbounds for i in 1:n
        d = abs(R[i, i])
        maxabs = max(maxabs, d)
        minabs = min(minabs, d)
    end
    return minabs > tol * maxabs
end

function solve!(ls::StaticJacobianLinearSystem{M, N, MN, K, KN, T}, ::Union{StaticSymbol{:auto}, StaticSymbol{:qr}}) where {M, N, MN, K, KN, T}
    ls.A.stale[] && factor_static!(ls)
    if ls.nonfinite
        fill!(ls.x, NaN)
        return NotPositiveDefinite
    end
    R, q = SMatrix(ls.R), SVector(ls.q)
    lambda = ls.lambda
    if !(lambda > 0)
        (M >= N && static_fullrank(R, M, N)) || return NotPositiveDefinite
        ls.x .= UpperTriangular(SMatrix{N, N}(R[SOneTo(N), :])) \ SVector{N}(q[SOneTo(N)])
        return Success
    end
    scaling = ls.scaling
    d = isnothing(scaling) ? ones(SVector{N, Float64}) : SVector{N, Float64}(scaling)
    S = vcat(R, SMatrix{N, N, T}(Diagonal(sqrt.(lambda .* d))))   # the damping, formed in Float64, narrowed to T
    F = qr(S)
    static_fullrank(F.R, K + N, N) || return NotPositiveDefinite
    qq = F.Q' * vcat(q, zeros(SVector{N, T}))
    ls.x .= UpperTriangular(F.R) \ qq
    return Success
end
function solve!(ls::StaticJacobianLinearSystem, ::StaticSymbol{S}) where S
    throw(ArgumentError("linsolver = :$S is not supported by a StaticJacobianLinearSystem; use :auto or :qr."))
end

conformance_matrix(A::StaticJacobianStorage) = conformance_gram(Matrix(A.J))
conformance_fill!(A::StaticJacobianStorage, M::AbstractMatrix) = (A.J .= conformance_factor(M, size(A.J)...); A)
conformance_traits(A::StaticJacobianStorage) = (exact = false, maxrank = min(size(A.J)...), indefinite = false, accumulates = false)
