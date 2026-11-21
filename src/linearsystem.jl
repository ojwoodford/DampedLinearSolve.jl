@enum SolveStatus Success NotPositiveDefinite Indefinite   # COV_EXCL_LINE: some Julia versions count the macro's generated code here, but never run it

# A solver is named by a Symbol or a StaticSymbol
linsolver_symbol(s::Symbol) = s
linsolver_symbol(s::StaticSymbol) = dynamic(s)

abstract type AbstractLinearSystem end
function solve! end
function dampen! end

# The systems are parametric in `T`, the element type of `A` (and of its copies and factorization): Float64 or
# Float32 (precision.jl). `b` and `x`, the damping `λ` and a Marquardt scaling are Float64 whatever `T` is. A
# solve narrows `b` into the scratch `rhs` (in `T`), solves there, and widens the result into `x`.

# Static linear system
mutable struct StaticLinearSystem{N, N2, T} <: AbstractLinearSystem
    const A::MMatrix{N, N, T, N2}
    const b::MVector{N, Float64}
    const x::MVector{N, Float64}
    lambda::Float64
    scaling::Union{Nothing, SVector{N, Float64}}
end
function StaticLinearSystem{N, N2, T}(A::MMatrix{N, N, T, N2}, b::MVector{N, Float64}, x::MVector{N, Float64}) where {N, N2, T}
    return StaticLinearSystem{N, N2, T}(A, b, x, 0.0, nothing)
end
function StaticLinearSystem{N, T}() where {N, T}
    N2 = N * N
    return StaticLinearSystem{N, N2, T}(MMatrix{N, N, T, N2}(undef), MVector{N, Float64}(undef), MVector{N, Float64}(undef), 0.0, nothing)
end
StaticLinearSystem{N}() where N = StaticLinearSystem{N, Float64}()

# Dense linear system. `solver_ws` holds preallocated factorization or Krylov workspaces
# (`nothing` when the solver needs none), built by `makedensews`. `rhs` (length n, in `T`) holds the
# right-hand side of the factorized system and then its solution.
# `cs` is the conditioning scale of the last solve and `condition` the internal switch (conditioning.jl).
mutable struct DenseLinearSystem{T, W} <: AbstractLinearSystem
    const A::Matrix{T}
    const b::Vector{Float64}
    const x::Vector{Float64}
    const A_scratch::Matrix{T}
    const rhs::Vector{T}
    const cs::Vector{Float64}
    const condition::Bool
    const solver_ws::W
    lambda::Float64
    scaling::Union{Nothing, Vector{Float64}}
end
function DenseLinearSystem{T}(A::Matrix{T}, b::Vector{Float64}, x::Vector{Float64}, A_scratch::Matrix{T}, solver_ws::W, lambda, scaling; condition::Bool=true) where {T, W}
    n = length(b)
    return DenseLinearSystem{T, W}(A, b, x, A_scratch, Vector{T}(undef, n), Vector{Float64}(undef, n), condition, solver_ws, lambda, scaling)
end
function DenseLinearSystem{T}(A::Matrix{T}, b::Vector{Float64}, x::Vector{Float64}, A_scratch::Matrix{T}, lambda, scaling; condition::Bool=true) where T
    return DenseLinearSystem{T}(A, b, x, A_scratch, nothing, lambda, scaling; condition)
end
function DenseLinearSystem{T}(n::Integer; solver_ws=nothing, condition::Bool=true) where T
    return DenseLinearSystem{T}(Matrix{T}(undef, n, n), Vector{Float64}(undef, n), Vector{Float64}(undef, n), Matrix{T}(undef, n, n), solver_ws, 0.0, nothing; condition)
end
function DenseLinearSystem{T}(A::Matrix{T}, b::Vector{Float64}, x::Vector{Float64}, A_scratch::Matrix{T}=Matrix{T}(undef, size(A)); solver_ws=nothing, condition::Bool=true) where T
    return DenseLinearSystem{T}(A, b, x, A_scratch, solver_ws, 0.0, nothing; condition)
end
DenseLinearSystem(n::Integer; kwargs...) = DenseLinearSystem{Float64}(n; kwargs...)
DenseLinearSystem(A::Matrix{T}, b::Vector{Float64}, x::Vector{Float64}; kwargs...) where T = DenseLinearSystem{T}(A, b, x; kwargs...)

# Sparse linear system. A direct solve factorizes `Aw`, a matrix of A's pattern with values of its own, into which it
# writes the conditioned, damped matrix, so `A` is never written; `rhs` and `sol` (length n, in `T`) are the
# right-hand side and the solution of the factorized system, and `cs` the conditioning scale. A system and its
# `copy_structure` copy share `Aw`, `rhs`, `sol` and `cs`: each solve writes them in full before reading them. The
# Krylov and Pardiso solves (extensions) damp `A` in place and restore its diagonal from `diag_cache`.
mutable struct SparseLinearSystem{T, Ti, SWS} <: AbstractLinearSystem
    const A::SparseMatrixCSC{T, Ti}
    const b::Vector{Float64}
    const x::Vector{Float64}
    const diag_inds::Vector{Ti}
    const solver_ws::SWS
    const diag_cache::Vector{T}
    const Aw::SparseMatrixCSC{T, Ti}
    const rhs::Vector{T}
    const sol::Vector{T}
    const cs::Vector{Float64}
    const condition::Bool
    lambda::Float64
    scaling::Union{Nothing, Vector{Float64}}
end
function SparseLinearSystem(A::SparseMatrixCSC{T, Ti}, b::Vector{Float64}, x::Vector{Float64}, diag_inds::Vector{Ti}, solver_ws::SWS, diag_cache::Vector{T}=copy(A.nzval[diag_inds]); condition::Bool=true) where {T, Ti, SWS}
    n = length(b)
    Aw = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, similar(A.nzval))
    return SparseLinearSystem{T, Ti, SWS}(A, b, x, diag_inds, solver_ws, diag_cache, Aw, Vector{T}(undef, n), Vector{T}(undef, n),
                                          Vector{Float64}(undef, n), condition, 0.0, nothing)
end

function ensure_diag_and_inds(A::SparseMatrixCSC{T, Ti}) where {T, Ti}
    @assert A.m == A.n "Hessian matrix must be square"
    n = A.n
    diag_inds = Vector{Ti}(undef, n)
    missing_diag = false
    for col in 1:n
        found = false
        for idx in nzrange(A, col)
            if A.rowval[idx] == col
                diag_inds[col] = idx
                found = true
                break
            end
        end
        if !found
            missing_diag = true
            break
        end
    end

    if !missing_diag
        return A, diag_inds
    end

    # Insert structural zeros on the diagonal. The result is a new matrix with a different
    # `nzval` layout from `A`, so `fdf!` is then handed this matrix, not the caller's prototype.
    I = Ti[]
    J = Ti[]
    V = T[]
    sizehint!(I, nnz(A) + n)
    sizehint!(J, nnz(A) + n)
    sizehint!(V, nnz(A) + n)
    for col in 1:n
        has_diag = false
        for idx in nzrange(A, col)
            r = A.rowval[idx]
            if r == col
                has_diag = true
            end
            push!(I, r)
            push!(J, col)
            push!(V, A.nzval[idx])
        end
        if !has_diag
            push!(I, col)
            push!(J, col)
            push!(V, zero(T))
        end
    end
    new_A = sparse(I, J, V, n, n)::SparseMatrixCSC{T, Ti}
    new_diag_inds = Vector{Ti}(undef, n)
    for col in 1:n
        for idx in nzrange(new_A, col)
            if new_A.rowval[idx] == col
                new_diag_inds[col] = idx
                break
            end
        end
    end
    return new_A, new_diag_inds
end

makesparsews(A, ::Union{StaticSymbol{:auto}, StaticSymbol{:ldl}}) = LDLFactorizations.ldl_analyze(A)
makesparsews(A, ::StaticSymbol{:cholesky}) = SparseArrays.cholesky(Symmetric(A, :U); check=false)
makesparsews(A, ::StaticSymbol{:qr}) = nothing
makesparsews(A, ::StaticSymbol{S}) where S = nothing
makesparsews(A, s::Symbol) = makesparsews(A, static(s))

# The prototype's element type is the system's (Float64 or Float32, precision.jl); a solver that does not offer it
# is an ArgumentError here, before anything is built (decision 10).
# `makehessiansystem` builds the system conditioned (conditioning.jl); `hessiansystem(H, linsolver, condition)` is the
# internal builder, whose `condition = false` (for tests only) builds a conditioned system type unconditioned. The
# static systems are never conditioned and ignore it.
makehessiansystem(H::AbstractMatrix, linsolver) = hessiansystem(H, linsolver, true)
makehessiansystem(n::Union{Integer, StaticInt}, linsolver) = hessiansystem(n, linsolver, true)

function hessiansystem(H::SparseMatrixCSC{T}, linsolver, condition::Bool) where T
    check_eltype(T, "The sparse Hessian prototype")
    check_offers(T, :sparse, linsolver)
    A, diag_inds = ensure_diag_and_inds(H)
    sym = static(linsolver_symbol(linsolver))
    ws = makesparsews(A, sym)
    diag_cache = copy(A.nzval[diag_inds])
    return SparseLinearSystem(A, zeros(Float64, size(A, 1)), zeros(Float64, size(A, 1)), diag_inds, ws, diag_cache; condition)
end

function hessiansystem(H::MMatrix{N, N, T, N2}, linsolver, ::Bool) where {N, T, N2}
    check_eltype(T, "The static Hessian prototype")
    check_offers(T, :static, linsolver)
    return StaticLinearSystem{N, N2, T}(H, MVector{N, Float64}(undef), MVector{N, Float64}(undef))
end

function hessiansystem(H::StaticMatrix{N, N}, linsolver, ::Bool) where N
    T = eltype(H)
    check_eltype(T, "The static Hessian prototype")
    check_offers(T, :static, linsolver)
    return StaticLinearSystem{N, T}()
end

# Preallocated workspace for a dense solve, or `nothing`. Extensions add methods for their solvers.
# `T` is the element type of the system the workspace is for
makedensews(n, s, ::Type{T}=Float64) where T = makedensews(n, static(linsolver_symbol(s)), T)
makedensews(n, ::StaticSymbol{:ldl}, ::Type{T}) where T = DenseLDLWorkspace{T}(n)
makedensews(n, ::StaticSymbol, ::Type) = nothing

function hessiansystem(H::Matrix{T}, linsolver, condition::Bool) where T
    check_eltype(T, "The dense Hessian prototype")
    check_offers(T, :dense, linsolver)
    n = size(H, 1)
    return DenseLinearSystem{T}(H, zeros(Float64, n), zeros(Float64, n); solver_ws=makedensews(n, linsolver, T), condition)
end

# Another dense prototype (a `Diagonal`, a `Symmetric`, ...) is copied into a `Matrix` of its element type
function hessiansystem(H::AbstractMatrix, linsolver, condition::Bool)
    T = eltype(H)
    check_eltype(T, "The Hessian prototype")
    return hessiansystem(Matrix{T}(H), linsolver, condition)
end

# Without a prototype: a system of `n` variables, in Float64, static (StaticArrays) when `n` is a
# `StaticInt` and dense when it is an `Int`. The caller decides which by the type of `n`.
hessiansystem(n::StaticInt, linsolver, ::Bool) = StaticLinearSystem{dynamic(n), Float64}()
hessiansystem(n::Integer, linsolver, condition::Bool) = DenseLinearSystem{Float64}(Int(n); solver_ws=makedensews(Int(n), linsolver), condition)

getA(ls::AbstractLinearSystem) = ls.A
getb(ls::AbstractLinearSystem) = ls.b
getx(ls::AbstractLinearSystem) = ls.x

# Whether every stored value of the system's matrix is finite. An optimizer asks after a failed solve:
# damping cannot repair a non-finite matrix, so they stop retrying (each system type has a method).
isfinitesystem(ls::Union{StaticLinearSystem, DenseLinearSystem}) = all(isfinite, ls.A)
isfinitesystem(ls::SparseLinearSystem) = all(isfinite, nonzeros(ls.A))


function zero!(ls::AbstractLinearSystem)
    fill!(getb(ls), zero(eltype(getb(ls))))
    fill!(getA(ls), zero(eltype(getA(ls))))
    return ls
end
function zero!(ls::SparseLinearSystem)
    fill!(ls.b, zero(eltype(ls.b)))
    fill!(ls.A.nzval, zero(eltype(ls.A.nzval)))
    return ls
end

# Unified damping interface: dampen! records, solve! applies
dampen!(ls::AbstractLinearSystem, lambda::Real) = (ls.lambda = convert(eltype(getb(ls)), lambda); ls.scaling = nothing; ls)
dampen!(ls::AbstractLinearSystem, lambda::Real, scaling) = (ls.lambda = convert(eltype(getb(ls)), lambda); ls.scaling = scaling; ls)

# Copy structure for trial relinearization (LevMarDataTrial)
copy_structure(ls::StaticLinearSystem{N, N2, T}) where {N, N2, T} = StaticLinearSystem{N, N2, T}(MMatrix{N, N, T, N2}(undef), MVector{N, Float64}(undef), ls.x, ls.lambda, ls.scaling)
copy_structure(ls::DenseLinearSystem{T, W}) where {T, W} = DenseLinearSystem{T, W}(Matrix{T}(undef, size(ls.A)), Vector{Float64}(undef, length(ls.b)), ls.x, Matrix{T}(undef, size(ls.A)), Vector{T}(undef, length(ls.b)), Vector{Float64}(undef, length(ls.b)), ls.condition, ls.solver_ws, ls.lambda, ls.scaling)
# The copy shares the solve's scratch (`rhs`, `sol`): each solve writes it in full before reading it
copy_structure(ls::SparseLinearSystem{T, Ti, SWS}) where {T, Ti, SWS} = SparseLinearSystem{T, Ti, SWS}(
    SparseMatrixCSC(ls.A.m, ls.A.n, ls.A.colptr, ls.A.rowval, Vector{T}(undef, length(ls.A.nzval))),
    Vector{Float64}(undef, length(ls.b)),
    ls.x,
    ls.diag_inds,
    ls.solver_ws,
    copy(ls.diag_cache),
    ls.Aw,
    ls.rhs,
    ls.sol,
    ls.cs,
    ls.condition,
    ls.lambda,
    ls.scaling
)

sharex(ls::StaticLinearSystem{N, N2, T}, from) where {N, N2, T} = ls.x === from.x ? ls : StaticLinearSystem{N, N2, T}(ls.A, ls.b, from.x, ls.lambda, ls.scaling)
sharex(ls::DenseLinearSystem{T, W}, from) where {T, W} = ls.x === from.x ? ls : DenseLinearSystem{T, W}(ls.A, ls.b, from.x, ls.A_scratch, ls.rhs, ls.cs, ls.condition, ls.solver_ws, ls.lambda, ls.scaling)
sharex(ls::SparseLinearSystem{T, Ti, SWS}, from) where {T, Ti, SWS} = ls.x === from.x ? ls : SparseLinearSystem{T, Ti, SWS}(ls.A, ls.b, from.x, ls.diag_inds, ls.solver_ws, ls.diag_cache, ls.Aw, ls.rhs, ls.sol, ls.cs, ls.condition, ls.lambda, ls.scaling)

# Diagonal operations for damping
function uniformscaling!(A::AbstractMatrix, k)
    LinearAlgebra.checksquare(A)
    @inbounds for i in axes(A, 1)
        A[i, i] += k
    end
end

function uniformscaling!(ls, k)
    uniformscaling!(getA(ls), k)
end

# The sparse system damps A in place, so these first save the undamped diagonal, from which every
# sparse solve restores it afterwards. Saving here, when the damping is applied, means a solve
# leaves A as it found it however A was written.
function uniformscaling!(ls::SparseLinearSystem, k)
    @inbounds for (i, idx) in enumerate(ls.diag_inds)
        ls.diag_cache[i] = ls.A.nzval[idx]
        ls.A.nzval[idx] += k
    end
end

# A Float64, whatever the element type of A
function max_diag(A::AbstractMatrix)
    LinearAlgebra.checksquare(A)
    m = 0.0
    @inbounds for i in axes(A, 1)
        m = max(m, Float64(abs(A[i, i])))
    end
    return m
end
function max_diag(ls::SparseLinearSystem)
    m = 0.0
    @inbounds for idx in ls.diag_inds
        m = max(m, Float64(abs(ls.A.nzval[idx])))
    end
    return m
end
max_diag(ls) = max_diag(getA(ls))

function damp_marquardt!(A::AbstractMatrix, scaling, dlambda)
    LinearAlgebra.checksquare(A)
    @inbounds for i in axes(A, 1)
        A[i, i] += dlambda * scaling[i]
    end
end
function damp_marquardt!(ls::SparseLinearSystem, scaling, dlambda)
    @inbounds for (i, idx) in enumerate(ls.diag_inds)
        ls.diag_cache[i] = ls.A.nzval[idx]
        ls.A.nzval[idx] += dlambda * scaling[i]
    end
end
damp_marquardt!(ls, scaling, dlambda) = damp_marquardt!(getA(ls), scaling, dlambda)

# A value equal for two prototypes from which systems of the same structure are built, for a caller
# that caches systems. A package with its own prototype type adds a method.
structurekey(H::AbstractMatrix) = (typeof(H), size(H))

# The diagonal of the undamped A, written into d
function getdiag!(d::AbstractVector, ls::AbstractLinearSystem)
    A = getA(ls)
    @inbounds for i in axes(A, 1)
        d[i] = A[i, i]
    end
    return d
end
function getdiag!(d::AbstractVector, ls::SparseLinearSystem)
    @inbounds for (i, idx) in enumerate(ls.diag_inds)
        d[i] = ls.A.nzval[idx]
    end
    return d
end

# The smallest damping the solver resolves, relative to the diagonal: below about eps times the
# diagonal, Cholesky of A + λD no longer sees the damping. The Jacobian system, solved by QR of J,
# tolerates about eps² (see jacobiansystem.jl).
dampingfloor(ls) = eps(Float64)

# Change in the undamped quadratic model, 0.5 dx'H dx + g'dx, for a step dx (negative for a
# descent step)
function quadratic_model_change(ls::AbstractLinearSystem, dx)
    return 0.5 * quadform(getA(ls), dx) + fast_dot(getb(ls), dx)
end
