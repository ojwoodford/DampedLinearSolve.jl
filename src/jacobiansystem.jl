# The residual-and-Jacobian linear system: A = JᵀJ held implicitly, b = Jᵀr. Solved by a
# Householder QR of [J r] once per linearization, then, per damping value, a QR of the small
# stacked matrix [R q; √λ D^½ 0]. With QR (and LSMR) JᵀJ is never formed; the normal-equations
# solvers, `:cholesky` and `:ldl`, form it once per linearization (jacobiannormalsystem.jl).

# Householder QR of the strided matrix `A` in place (LAPACK `dgeqrf`, `sgeqrf`), with preallocated `tau`
# and work array, so that it does not allocate; and the optimal work-array length for an m×n matrix. A
# workspace query (lwork = -1) reads only the dimensions and lda, and touches neither the matrix nor tau, so
# one-element stand-ins are passed for both, with the true m, n and lda.
for (T, geqrf) in ((Float64, :dgeqrf_), (Float32, :sgeqrf_))
    @eval begin
        function geqrf_ws!(A::StridedMatrix{$T}, tau::Vector{$T}, work::Vector{$T})
            m, n = size(A)
            length(tau) >= min(m, n) || throw(DimensionMismatch("tau has length $(length(tau)), needs $(min(m, n))"))
            (m == 0 || n == 0) && return A
            info = Ref{BlasInt}()
            ccall((LinearAlgebra.BLAS.@blasfunc($geqrf), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                   Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}),
                  m, n, A, max(1, stride(A, 2)), tau, work, BlasInt(length(work)), info)
            LinearAlgebra.LAPACK.chkargsok(info[])
            return A
        end

        function geqrf_worksize(m::Integer, n::Integer, ::Type{$T})
            (m == 0 || n == 0) && return 1
            A = Ref(zero($T))
            tau = Ref(zero($T))
            work = Ref(zero($T))
            info = Ref{BlasInt}()
            ccall((LinearAlgebra.BLAS.@blasfunc($geqrf), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ref{$T}, Ref{BlasInt},
                   Ref{$T}, Ref{$T}, Ref{BlasInt}, Ref{BlasInt}),
                  m, n, A, max(1, m), tau, work, BlasInt(-1), info)
            LinearAlgebra.LAPACK.chkargsok(info[])
            return max(1, Int(work[]))
        end
    end
end
geqrf_worksize(m::Integer, n::Integer) = geqrf_worksize(m, n, Float64)

# The factorization scratch: [J r] factored in place, the stacked matrix of the damped solve,
# and their LAPACK arrays. A system and its `copy_structure` copy share one: only one of them
# has a live factorization at a time (the trial system is not solved until it becomes the main
# one). `owner` is the token of the system whose factorization the scratch holds (0: none).
# In `T`, the system's element type: `r` is narrowed into the last column of `Jr`.
# Conditioned (conditioning.jl), `Jr` holds [J S r] and `cs` the scale s of the factorization it holds.
mutable struct JacobianQRScratch{T}
    const Jr::Matrix{T}     # m × (n+1): [J S r], factored in place; R and q = (Qᵀr)[1:k] on top
    const tau::Vector{T}
    const S::Matrix{T}      # 2n × (n+1): [R q; √λ D^½ S 0], factored in place per damping
    const tau2::Vector{T}
    const work::Vector{T}
    const cs::Vector{Float64}
    owner::Int
end
function JacobianQRScratch{T}(m::Integer, n::Integer) where T
    lwork = max(geqrf_worksize(m, n + 1, T), geqrf_worksize(2n, n + 1, T))
    return JacobianQRScratch{T}(Matrix{T}(undef, m, n + 1), Vector{T}(undef, min(m, n + 1)),
                         Matrix{T}(undef, 2n, n + 1), Vector{T}(undef, min(2n, n + 1)),
                         Vector{T}(undef, lwork), ones(n), 0)
end
JacobianQRScratch(m::Integer, n::Integer) = JacobianQRScratch{Float64}(m, n)

# The Jacobian systems, dense and sparse: they share the Marquardt scaling (from the squared column
# norms of J, `colnorms2!`) and the damping floor
abstract type AbstractJacobianLinearSystem <: AbstractLinearSystem end

const JACOBIAN_TOKENS = Threads.Atomic{Int}(0)

# The state of one system's linearization: whether its factorization and the squared column norms
# of J (the diagonal of JᵀJ) are current. `stale` and `normsstale` are both set, through
# `markstale!`, by `zero!` and by the linearization wrapper after `resjac!` has written J and r;
# the QR clears `stale`, and computing the column norms clears `normsstale`. The factorization
# itself is in the (possibly shared) scratch, and is this system's only while the scratch's owner
# is this system's token: a solve refactors otherwise.
mutable struct JacobianQRWorkspace{T}
    const scratch::JacobianQRScratch{T}
    const token::Int
    const colnorm2::Vector{Float64}   # n: squared column norms of J, the diagonal of JᵀJ
    stale::Bool
    normsstale::Bool
    nonfinite::Bool               # J or r held a NaN or Inf at the last factorization
    nfactor::Int                  # number of QR factorizations of [J r], for tests
    nnorms::Int                   # number of column-norm computations, for tests
end
JacobianQRWorkspace(scratch::JacobianQRScratch{T}, n::Integer) where T =
    JacobianQRWorkspace{T}(scratch, Threads.atomic_add!(JACOBIAN_TOKENS, 1) + 1, Vector{Float64}(undef, n), true, true, false, 0, 0)
JacobianQRWorkspace(m::Integer, n::Integer) = JacobianQRWorkspace(JacobianQRScratch(m, n), n)
# Whether the scratch holds this system's current factorization
factorized(ws::JacobianQRWorkspace) = !ws.stale && ws.scratch.owner == ws.token

"""
    JacobianStorage

The Hessian storage of a [`DenseJacobianLinearSystem`](@ref): the Jacobian `J` (`m × n`) and the
residual `r` (length `m`), standing for `A = JᵀJ`, which is never formed.
`quadform(A, v)` is `‖J v‖²`, and the diagonal of `A` is the squared column norms of `J`.
"""
struct JacobianStorage{T}
    J::Matrix{T}
    r::Vector{Float64}
    Jv::Vector{Float64}           # scratch for quadform
    ws::JacobianQRWorkspace{T}
end
JacobianStorage(J::Matrix{T}) where T = JacobianStorage(J, zeros(size(J, 1)), zeros(size(J, 1)), JacobianQRWorkspace(JacobianQRScratch{T}(size(J)...), size(J, 2)))

Base.size(A::JacobianStorage) = (size(A.J, 2), size(A.J, 2))
Base.size(A::JacobianStorage, d::Integer) = size(A)[d]
markstale!(A::JacobianStorage) = (A.ws.stale = true; A.ws.normsstale = true; A)
# The storage the linearization writes, J and r, and the gradient Jᵀr it then forms
getJ(A::JacobianStorage) = A.J
getr(A::JacobianStorage) = A.r
gradient!(g::AbstractVector, A::JacobianStorage) = jtr!(g, A.J, A.r)

# y = J v, one scaled column at a time: a plain loop, not BLAS gemv, so that per-cost work stays
# single-threaded (rules.md 23). The first column is written, not added, so y needs no zeroing.
function jmul!(y::AbstractVector, J::Matrix{<:Union{Float64, Float32}}, v::AbstractVector)
    m, n = size(J)
    if n == 0
        fill!(y, 0.0)
        return y
    end
    @inbounds begin
        v1 = v[1]
        @fastmath @simd for i in 1:m
            y[i] = J[i, 1] * v1
        end   # COV_EXCL_LINE: the end of an @simd loop; Julia 1.10 counts it, but never runs it
        for j in 2:n
            vj = v[j]
            @fastmath @simd for i in 1:m
                y[i] += J[i, j] * vj
            end   # COV_EXCL_LINE: the end of an @simd loop; Julia 1.10 counts it, but never runs it
        end
    end
    return y
end

# ‖J v‖², with J v formed into the scratch `Jv` (Float64: the products of J's entries with a Float64 v
# promote, so a Float32 J is accumulated in Float64)
function quadform(A::JacobianStorage, v::AbstractVector)
    size(A.J, 2) == 0 && return 0.0
    Jv = jmul!(A.Jv, A.J, v)
    total = 0.0
    @inbounds @simd for i in eachindex(Jv)
        total += Jv[i] * Jv[i]
    end   # COV_EXCL_LINE: the end of an @simd loop; Julia 1.10 counts it, but never runs it
    return total
end

# g = Jᵀ r as one dot product per column of J: a plain loop, not BLAS gemv (rules.md 23)
function jtr!(g::AbstractVector, J::Matrix{<:Union{Float64, Float32}}, r::Vector{Float64})
    m, n = size(J)
    @inbounds for j in 1:n
        total = 0.0
        @fastmath @simd for i in 1:m
            total += J[i, j] * r[i]
        end   # COV_EXCL_LINE: the end of an @simd loop; Julia 1.10 counts it, but never runs it
        g[j] = total
    end
    return g
end

# Squared norm of column j of J, summed in row order (as `conformance_matrix` sums), so the two agree exactly
@inline function sqcolnorm(J::Matrix, j::Integer)
    total = 0.0
    @inbounds for i in axes(J, 1)
        Jij = Float64(J[i, j])
        total += Jij * Jij
    end
    return total
end

# The squared column norms of J, the diagonal of JᵀJ: computed once per linearization and shared
# by `max_diag` and the Marquardt scaling
function colnorms2!(A::JacobianStorage)
    ws = A.ws
    if ws.normsstale
        @inbounds for j in axes(A.J, 2)
            ws.colnorm2[j] = sqcolnorm(A.J, j)
        end
        ws.normsstale = false
        ws.nnorms += 1
    end
    return ws.colnorm2
end

# The largest squared column norm; NaN if any is NaN (`max` propagates NaN)
function max_diag(A::JacobianStorage)
    md = 0.0
    for d in colnorms2!(A)
        md = max(md, d)
    end
    return md
end

"""
    DenseJacobianLinearSystem(J::Matrix{Float64}; linsolver = :auto)
    DenseJacobianLinearSystem(m::Integer, n::Integer; linsolver = :auto)

The Jacobian linear system of a residual-and-Jacobian problem: the normal equations `(JᵀJ + λD) x = Jᵀr`,
with `J` dense (`m × n`, writing into the given matrix, or its own) and, with QR (the default) or
LSMR, `JᵀJ` never formed.
It meets the [`AbstractLinearSystem`](@ref) interface with `getA(ls)` a [`JacobianStorage`](@ref)
holding `J` and `r`, and `getb(ls)` the gradient `Jᵀr`.

`solve!(ls, :auto)` (or `:qr`) factors `[J r]` by Householder QR once per linearization
(`O(mn²)`), and then, for each damping value, factors the `2n × n` stacked matrix
`[R; √λ D^½]` (`O(n³)`, independent of `m`), so retries with a larger `λ` are cheap.
Memory: `J`, one `m × (n+1)` scratch copy of `[J r]` for the factorization, and `O(n²)` more.
`copy_structure` (the spare system an optimizer keeps for trial points) adds its own `J` and
shares the scratch: about three times the storage of `J` in all. An undamped
solve on a numerically rank-deficient `J` (smallest `|R[i,i]|` at or below
`max(m, n)·eps` times the largest, which includes `m < n`) returns `NotPositiveDefinite`, as
Cholesky does on singular normal equations; a `NaN` or `Inf` in `J` or `r` also returns
`NotPositiveDefinite`, with `x` set to `NaN`. Nothing is allocated after construction.

The factorization and the squared column norms of `J` (the diagonal of `JᵀJ`, used by `max_diag`
and `getdiag!`, for a Marquardt scaling) are computed once per linearization. `zero!(ls)` does not
zero `J` or `r`: the linearization sets them, so `zero!` only invalidates the factorization and the
column norms (and zeroes `b`, which the linearization then overwrites). Code that writes
`getJ(getA(ls))` or `getr(getA(ls))` must call `zero!` (before or after writing), or
`markstale!(getA(ls))`.

`DenseJacobianLinearSystem(J)` uses `J` as its storage with the values it holds, as does
`makejacobiansystem` with a `Matrix` prototype; with an integer `m`, `J` starts as zeros.
`copy_structure` copies `J`, so a trial system starts with the same entries.

# The normal equations: `:cholesky` and `:ldl`

Built with `linsolver = :cholesky` or `:ldl` (`DenseJacobianLinearSystem(J; linsolver = :cholesky)`,
or `makejacobiansystem` with that solver), the system solves the normal equations instead: once per
linearization it forms the upper triangle of `JᵀJ` (BLAS `syrk`, about `mn²/2` multiply-adds), and
for each damping value it factors `JᵀJ + λD` (`n³/3`, independent of `m`) by Cholesky (`:cholesky`)
or Bunch–Kaufman (`:ldl`), as the dense Hessian system does, and returns that solve's status
(`Success`, `NotPositiveDefinite`, or `Indefinite` from `:ldl`). QR costs about `2mn² − 2n³/3` per
linearization, so the normal equations are cheaper when `m` is well above `n`. Memory: `J` and two
`n × n` matrices, with no copy of `[J r]`. The product and the factorization are BLAS and LAPACK
calls, and use their threads.

The normal equations square the condition number of `J`, so they lose accuracy that QR keeps on an
ill-conditioned `J`; `:auto` is QR. Rank deficiency is reported only as the factorization reports it
(an exactly or numerically singular `JᵀJ` may still factorize), except that an undamped solve with
`m < n` returns `NotPositiveDefinite` without factorizing. A `NaN` or `Inf` in `J` or `r` returns
`NotPositiveDefinite` with `x` set to `NaN`, and forms nothing. [`dampingfloor`](@ref) is
`eps(Float64)`, as for a factorization of a Hessian. A system is solved with the solver it was built
for: QR or LSMR asked of a system built for `:cholesky`, or the reverse, is an `ArgumentError`.
"""
mutable struct DenseJacobianLinearSystem{W, T} <: AbstractJacobianLinearSystem
    const A::JacobianStorage{T}
    const b::Vector{Float64}      # the gradient Jᵀr
    const x::Vector{Float64}
    const solver_ws::W            # the workspace of an iterative solver or of the normal equations (`makejacobianws`), or nothing for QR
    const condition::Bool         # the internal switch of the conditioning (conditioning.jl)
    lambda::Float64
    scaling::Union{Nothing, Vector{Float64}}
end
# `T`, the element type of J, is the system's: J, the factorization and its scratch are in T
function DenseJacobianLinearSystem(J::Matrix{T}; linsolver = :auto, condition::Bool = true) where T
    m, n = size(J)
    A = JacobianStorage(J, zeros(m), zeros(m), JacobianQRWorkspace(qrscratch(m, n, linsolver, T), n))
    return DenseJacobianLinearSystem(A, zeros(n), zeros(n), makejacobianws(m, n, linsolver, T, condition), condition, 0.0, nothing)
end
DenseJacobianLinearSystem(m::Integer, n::Integer; linsolver = :auto, condition::Bool = true) = DenseJacobianLinearSystem(zeros(Float64, m, n); linsolver, condition)
isfinitesystem(ls::DenseJacobianLinearSystem) = all(isfinite, ls.A.J) && all(isfinite, ls.A.r)

# The solvers of the Jacobian systems: QR (`:auto`, `:qr`) and, for the dense system, the normal
# equations (`:cholesky`, `:ldl`, jacobiannormalsystem.jl) in this package, and those an extension adds
# through `makejacobianws` (LSMR, in KrylovExt), which builds the solver's workspace
needsqr(linsolver) = linsolver_symbol(linsolver) in (:auto, :qr)
makejacobianws(m, n, linsolver, ::Type{T}=Float64, condition::Bool=true) where T = makejacobianws(m, n, static(linsolver_symbol(linsolver)), T, condition)
makejacobianws(m, n, ::StaticSymbol, ::Type, ::Bool) = nothing
# The QR scratch of the dense system: [J r] and the stacked matrix for QR; an iterative solver keeps
# only the empty scratch's owner token
qrscratch(m, n, linsolver, ::Type{T}=Float64) where T = qrscratch(m, n, static(linsolver_symbol(linsolver)), T)
qrscratch(m, n, ::Union{StaticSymbol{:auto}, StaticSymbol{:qr}}, ::Type{T}) where T = JacobianQRScratch{T}(m, n)
qrscratch(m, n, ::StaticSymbol, ::Type{T}) where T = JacobianQRScratch{T}(0, n)

# LSMR's stopping tolerances, for LSMR on J (KrylovExt). Starting values, not measured (rules.md 26;
# improvements.md item 5):
# Krylov's own defaults, stated here so that a change in Krylov does not change them. itmax is Krylov's
# default, m + n iterations.
const LSMR_ETOL = sqrt(eps(Float64))     # stopping on the normal-equation residual
const LSMR_AXTOL = sqrt(eps(Float64))
const LSMR_BTOL = sqrt(eps(Float64))
const LSMR_CONLIM = 1 / sqrt(eps(Float64))

# The Jacobian system of `n` variables for the Jacobian prototype `J`. A Matrix prototype is the storage
# resjac! writes into, used with the values it holds: entries resjac! never writes keep the caller's
# values. With an integer m the matrix starts as zeros. The prototype's type selects the system (a
# sparse one in sparsejacobiansystem.jl).
# A small problem with a static number of variables (`n` a `StaticInt`) gets the static system
# (staticjacobiansystem.jl). The choice depends on m, known only at run time, so it is made here, once,
# when the system is built.
# The element type of a Matrix prototype is the system's (Float64 or Float32); with an integer m it is Float64. A solver
# that does not offer it is an ArgumentError here (decision 10).
# `makejacobiansystem` builds the system conditioned; `jacobiansystem(J, n, linsolver, condition)` is the internal
# builder (as `hessiansystem` is for a Hessian system)
makejacobiansystem(m::Integer, n, linsolver) = jacobiansystem(m, n, linsolver, true)
makejacobiansystem(J::Matrix, n, linsolver) = jacobiansystem(J, n, linsolver, true)
function jacobiansystem(m::Integer, n, linsolver, condition::Bool)
    usesstaticjacobian(m, n, linsolver) && return StaticJacobianLinearSystem(zeros(MMatrix{Int(m), dynamic(n), Float64}))
    return DenseJacobianLinearSystem(m, dynamic(n); linsolver, condition)
end
function jacobiansystem(J::Matrix{T}, n, linsolver, condition::Bool) where T
    check_eltype(T, "The Jacobian prototype")
    if usesstaticjacobian(size(J, 1), n, linsolver)
        check_offers(T, :jacobian_static, linsolver)
        return StaticJacobianLinearSystem(J)
    end
    check_offers(T, :jacobian_dense, linsolver)
    return DenseJacobianLinearSystem(J; linsolver, condition)
end

# resjac! sets every entry of r and every entry of J that can be non-zero, so a fresh
# linearization needs no zeroing: only the cached factorization and column norms are invalidated.
# b is zeroed (n entries) so that zero! leaves the system as the interface describes; the
# linearization wrapper overwrites it.
function zero!(ls::DenseJacobianLinearSystem)
    fill!(ls.b, 0.0)
    markstale!(ls.A)
    return ls
end

# The trial system starts from the same J (its unwritten entries must match the main system's),
# and shares the factorization scratch: its own J, r, b, stale flags and column norms, so it costs
# one more J, not two
function copy_structure(ls::DenseJacobianLinearSystem)
    A = ls.A
    m, n = size(A.J)
    At = JacobianStorage(copy(A.J), zeros(m), zeros(m), JacobianQRWorkspace(A.ws.scratch, n))
    return DenseJacobianLinearSystem(At, zeros(length(ls.b)), ls.x, ls.solver_ws, ls.condition, ls.lambda, ls.scaling)
end
sharex(ls::DenseJacobianLinearSystem, from) = ls.x === from.x ? ls : DenseJacobianLinearSystem(ls.A, ls.b, from.x, ls.solver_ws, ls.condition, ls.lambda, ls.scaling)

# QR of [J r]; afterwards Jr[1:k, 1:n] holds R (upper trapezoidal) and Jr[1:k, n+1] holds
# q = (Qᵀr)[1:k], with k = min(m, n)
# Conditioned, the columns of J are scaled by s (from the column norms) as they are copied, and s is kept in the
# scratch with the factorization
function factor_jacobian!(A::JacobianStorage{T}, condition::Bool) where T
    ws = A.ws
    J, r = A.J, A.r
    m, n = size(J)
    sc = ws.scratch
    ws.stale = false
    sc.owner = ws.token
    ws.nonfinite = !(all(isfinite, J) && all(isfinite, r))
    ws.nonfinite && return nothing
    Jr = sc.Jr
    if condition
        d = colnorms2!(A)
        @inbounds for j in 1:n
            sc.cs[j] = conditioning_scale(d[j], T)
            sj = T(sc.cs[j])
            for i in 1:m
                Jr[i, j] = J[i, j] * sj
            end
        end
    else
        fill!(sc.cs, 1.0)
        copyto!(view(Jr, :, 1:n), J)
    end
    copyto!(view(Jr, :, n + 1), r)   # narrowed to the scratch's element type
    geqrf_ws!(Jr, sc.tau, sc.work)
    ws.nfactor += 1
    return nothing
end

# Numerical full rank of the n×n upper triangle stored in the top of the factored `R` (of
# size(R, 1) rows): the smallest |R[i,i]| must exceed max(rows, n)·eps times the largest, with eps that of R's
# element type. A
# threshold of n·eps alone misses an exactly duplicated column of a J with more rows than columns, whose R[i,i] is left
# at the rounding error of the Householder reflections, which grows with the number of rows.
function fullrank_upper(R::Matrix, n::Integer)
    tol = max(size(R, 1), n) * Float64(eps(eltype(R)))
    maxabs = 0.0
    minabs = Inf
    @inbounds for i in 1:n
        d = abs(R[i, i])
        maxabs = max(maxabs, d)
        minabs = min(minabs, d)
    end
    return minabs > tol * maxabs
end

# x = U \ x for the upper triangle U = R[1:n, 1:n], column-oriented
function backsubstitute!(x::AbstractVector, R::Matrix, n::Integer)
    @inbounds for j in n:-1:1
        x[j] /= R[j, j]
        xj = x[j]
        for i in 1:j-1
            x[i] -= R[i, j] * xj
        end
    end
    return x
end

function solve!(ls::DenseJacobianLinearSystem, s::Union{StaticSymbol{:auto}, StaticSymbol{:qr}})
    A = ls.A
    ws = A.ws
    m, n = size(A.J)
    size(ws.scratch.Jr, 1) == m || solvermismatch(ls, dynamic(s))
    factorized(ws) || factor_jacobian!(A, ls.condition)
    if ws.nonfinite
        fill!(ls.x, NaN)
        return NotPositiveDefinite
    end
    sc = ws.scratch
    k = min(m, n)
    Jr = sc.Jr
    lambda = ls.lambda
    if !(lambda > 0)
        # Undamped: R x = q, if R is square and numerically of full rank
        (m >= n && fullrank_upper(Jr, n)) || return NotPositiveDefinite
        @inbounds for i in 1:n
            ls.x[i] = Jr[i, n + 1]
        end
        backsubstitute!(ls.x, Jr, n)
        unscale!(ls.x, sc.cs, ls.condition)
        return Success
    end
    # Damped: QR of [R q; √λ D^½ 0], then back-substitution
    S = sc.S
    scaling = ls.scaling
    @inbounds begin
        for j in 1:n+1, i in 1:2n
            S[i, j] = 0.0
        end
        for j in 1:n+1, i in 1:min(j, k)
            S[i, j] = Jr[i, j]
        end
        for i in 1:n
            S[n + i, i] = sqrt(lambda * (isnothing(scaling) ? 1.0 : scaling[i])) * sc.cs[i]
        end
    end
    geqrf_ws!(S, sc.tau2, sc.work)
    fullrank_upper(S, n) || return NotPositiveDefinite
    @inbounds for i in 1:n
        ls.x[i] = S[i, n + 1]
    end
    backsubstitute!(ls.x, S, n)
    unscale!(ls.x, sc.cs, ls.condition)
    return Success
end

# x = S y, for a conditioned solve
function unscale!(x::AbstractVector, s::Vector{Float64}, condition::Bool)
    if condition
        @inbounds for i in eachindex(x)
            x[i] *= s[i]
        end
    end
    return x
end

# A solver the system takes but was not built for (the normal equations' are in jacobiannormalsystem.jl)
# is a mismatch; any other is not supported
function solve!(ls::DenseJacobianLinearSystem, ::StaticSymbol{S}) where S
    S in (:cholesky, :ldl) && solvermismatch(ls, S)
    throw(ArgumentError("linsolver = :$S is not supported by a DenseJacobianLinearSystem; use :auto, :qr, :cholesky or :ldl, or :lsmr with Krylov.jl loaded."))
end

# QR of J works with cond(J), not cond(JᵀJ) = cond(J)², so a damping down to about eps² times the
# diagonal of JᵀJ still changes the solve (the normal equations' floor is eps: jacobiannormalsystem.jl)
dampingfloor(ls::AbstractJacobianLinearSystem) = Float64(eps(systemeltype(ls)))^2
systemeltype(::DenseJacobianLinearSystem{W, T}) where {W, T} = T

# The diagonal of JᵀJ, the squared column norms of J (computed once per linearization)
getdiag!(d::AbstractVector, ls::AbstractJacobianLinearSystem) = copyto!(d, colnorms2!(ls.A))

# The Jacobian prototypes the Jacobian systems are built from: their checks, size and solvers
const JACOBIAN_SUPPORT = "A Jacobian system is built from a Matrix or SparseMatrixCSC{T, Int} Jacobian prototype, with T = Float64 or Float32 (the system's element type), or an integer number of residuals m (Float64);" *
    " each takes linsolver = :auto or :qr, or :lsmr with Krylov.jl loaded, and a Matrix prototype or an integer m also takes :cholesky or :ldl (the normal equations, JᵀJ formed from J);" *
    " a package that implements the linear-system contract for its own prototype adds it (BlockLinearSystems.jl adds SchurBlockJacobian)"

# The Jacobian prototype: a Matrix{Float64} or Matrix{Float32} with at least one row, the sparse matrix of either, or
# an integer m >= 1
function checkjacobian(J)
    if J isa Integer && !(J isa Bool)
        J >= 1 || throw(ArgumentError("The number of residuals m must be at least 1; got $J. $JACOBIAN_SUPPORT."))
    elseif J isa Union{Matrix{Float64}, Matrix{Float32}}
        size(J, 1) >= 1 || throw(ArgumentError("The Jacobian prototype has no rows (size $(size(J))): it needs one row per residual. $JACOBIAN_SUPPORT."))
    elseif J isa Union{SparseMatrixCSC{Float64, Int}, SparseMatrixCSC{Float32, Int}}
        size(J, 1) >= 1 || throw(ArgumentError("The Jacobian prototype has no rows (size $(size(J))): it needs one row per residual. $JACOBIAN_SUPPORT."))
    elseif J isa AbstractSparseMatrix
        throw(ArgumentError("A sparse Jacobian prototype of type $(typeof(J)) is not supported. $JACOBIAN_SUPPORT."))
    elseif J isa AbstractMatrix && !(eltype(J) in SYSTEM_ELTYPES)
        throw(ArgumentError("The Jacobian must be Float64 or Float32; got a prototype with element type $(eltype(J)). $JACOBIAN_SUPPORT."))
    else
        throw(ArgumentError("A Jacobian prototype of type $(typeof(J)) is not supported. $JACOBIAN_SUPPORT."))
    end
    return nothing
end

# The number of residuals a Jacobian prototype stands for
jacobian_nrows(J) = J isa Integer ? Int(J) : size(J, 1)

# The prototype's column count (nothing for an integer m, whose J is sized by the number of variables), and the solvers it
# accepts. An extension adds methods for its prototype types.
jacobian_ncols(J::AbstractMatrix) = size(J, 2)
jacobian_ncols(J) = nothing
function check_jacobian_linsolver(J, s::Symbol)
    s in (:auto, :qr, :lsmr) && return nothing
    s in (:cholesky, :ldl) && densejacobianprototype(J) && return nothing
    throw(ArgumentError("linsolver = :$s is not supported by the Jacobian system of a $(typeof(J)) Jacobian prototype. $JACOBIAN_SUPPORT."))
end
# The prototypes of the dense Jacobian system, which alone takes the normal-equations solvers
densejacobianprototype(J) = (J isa Integer && !(J isa Bool)) || J isa Union{Matrix{Float64}, Matrix{Float32}}
