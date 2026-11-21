module KrylovExt

using DampedLinearSolve
using Krylov
using SparseArrays
using LinearAlgebra
using Static

import DampedLinearSolve: validatelinsolver, makesparsews, makedensews, solve!
import DampedLinearSolve: AbstractLinearSystem, SparseLinearSystem, DenseLinearSystem
import DampedLinearSolve: SolveStatus, Success, NotPositiveDefinite, Indefinite
import DampedLinearSolve: uniformscaling!, damp_marquardt!, builtfor, solvermismatch
import DampedLinearSolve: DenseJacobianLinearSystem, SparseJacobianLinearSystem, makejacobianws, jmul!, jtr!
import DampedLinearSolve: lsmr_tolerances, offers_eltype

# Validation of Krylov solver symbols
validatelinsolver(::StaticSymbol{:cg}) = true
validatelinsolver(::StaticSymbol{:minres}) = true
validatelinsolver(::StaticSymbol{:lsmr}) = true   # for a Jacobian system: on the normal equations it would square their condition number again

# The Krylov solvers run in the system's element type: the workspace's vectors are in T, b (and r) is narrowed into
# them and the solution widened into x. They are not conditioned: for an iterative solve a diagonal scaling is a
# preconditioner, which changes the iterates by more than rounding (single-precision.md, decision 11, proposed).
const KrylovHessian = Union{StaticSymbol{:cg}, StaticSymbol{:minres}}
offers_eltype(::Type{Float32}, ::Val{:dense}, ::KrylovHessian) = true
offers_eltype(::Type{Float32}, ::Val{:sparse}, ::KrylovHessian) = true
offers_eltype(::Type{Float32}, ::Val{:jacobian_dense}, ::StaticSymbol{:lsmr}) = true
offers_eltype(::Type{Float32}, ::Val{:jacobian_sparse}, ::StaticSymbol{:lsmr}) = true

# A Krylov workspace plus storage for a Jacobi (diagonal) preconditioner
struct JacobiWorkspace{W, T}
    ws::W
    invdiag::Vector{T}
end
JacobiWorkspace(ws, n::Integer, ::Type{T}) where T = JacobiWorkspace(ws, Vector{T}(undef, n))

# Fill the inverse of the (damped) diagonal of `ls.A`. Returns false, meaning "don't
# precondition", when a diagonal entry is not positive and finite: the Jacobi preconditioner
# must be positive definite, and an indefinite matrix should still be reported as such.
function jacobi!(invdiag::Vector, ls::SparseLinearSystem)
    @inbounds for (i, idx) in enumerate(ls.diag_inds)
        d = ls.A.nzval[idx]
        (d > 0 && isfinite(d)) || return false
        invdiag[i] = inv(d)
    end
    return true
end

# Preallocated workspace creation for SparseLinearSystem
makesparsews(A, ::StaticSymbol{:cg}) = JacobiWorkspace(Krylov.CgWorkspace(A, zeros(eltype(A), size(A, 1))), size(A, 1), eltype(A))
makesparsews(A, ::StaticSymbol{:minres}) = JacobiWorkspace(Krylov.MinresWorkspace(A, zeros(eltype(A), size(A, 1))), size(A, 1), eltype(A))

# Preallocated workspace creation for DenseLinearSystem
makedensews(n, ::StaticSymbol{:cg}, ::Type{T}) where T = Krylov.CgWorkspace(n, n, Vector{T})
makedensews(n, ::StaticSymbol{:minres}, ::Type{T}) where T = Krylov.MinresWorkspace(n, n, Vector{T})

builtfor(::JacobiWorkspace{<:Krylov.CgWorkspace}) = ":cg"
builtfor(::JacobiWorkspace{<:Krylov.MinresWorkspace}) = ":minres"

# The sparse workspace is built once, by `makesparsews`, and `solver_ws` is a `const` field,
# so it cannot be replaced when it doesn't match the requested solver.
function check_krylov_ws(ls, ::Type{W}, asked::Symbol) where W
    ls.solver_ws isa W && return ls.solver_ws
    solvermismatch(ls, asked)
end

# With `linesearch=true`, nonpositive curvature sets both `indefinite` and `solved`, so check it first
function cg_status(stats)
    stats.indefinite && return Indefinite
    return stats.solved ? Success : NotPositiveDefinite
end

# SparseLinearSystem execution
function solve!(ls::SparseLinearSystem, ::StaticSymbol{:cg})
    jws = check_krylov_ws(ls, JacobiWorkspace{<:Krylov.CgWorkspace}, :cg)
    ws = jws.ws
    # Krylov's CG throws, rather than reporting, on a non-finite operator
    DampedLinearSolve.isfinitesystem(ls) || return NotPositiveDefinite
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    # `linesearch=true` makes CG stop and flag `stats.indefinite` on nonpositive curvature;
    # without it, CG keeps iterating on an indefinite matrix. Jacobi-preconditioned when the
    # diagonal is positive (`M` is applied as a product, so it holds the inverse diagonal).
    if jacobi!(jws.invdiag, ls)
        Krylov.cg!(ws, ls.A, copyto!(ls.rhs, ls.b); M=Diagonal(jws.invdiag), linesearch=true)
    else
        Krylov.cg!(ws, ls.A, copyto!(ls.rhs, ls.b); linesearch=true)
    end
    copyto!(ls.x, ws.x)
    if ls.lambda > 0
        @inbounds for (i, idx) in enumerate(ls.diag_inds)
            ls.A.nzval[idx] = ls.diag_cache[i]
        end
    end
    return cg_status(ws.stats)
end

function solve!(ls::SparseLinearSystem, ::StaticSymbol{:minres})
    jws = check_krylov_ws(ls, JacobiWorkspace{<:Krylov.MinresWorkspace}, :minres)
    ws = jws.ws
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls, ls.lambda)
        else
            damp_marquardt!(ls, ls.scaling, ls.lambda)
        end
    end
    if jacobi!(jws.invdiag, ls)
        Krylov.minres!(ws, ls.A, copyto!(ls.rhs, ls.b); M=Diagonal(jws.invdiag))
    else
        Krylov.minres!(ws, ls.A, copyto!(ls.rhs, ls.b))
    end
    copyto!(ls.x, ws.x)
    if ls.lambda > 0
        @inbounds for (i, idx) in enumerate(ls.diag_inds)
            ls.A.nzval[idx] = ls.diag_cache[i]
        end
    end
    return ws.stats.solved ? Success : NotPositiveDefinite
end

# DenseLinearSystem execution
function solve!(ls::DenseLinearSystem, ::StaticSymbol{:cg})
    DampedLinearSolve.isfinitesystem(ls) || return NotPositiveDefinite   # Krylov's CG would throw
    copyto!(ls.A_scratch, ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls.A_scratch, ls.lambda)
        else
            damp_marquardt!(ls.A_scratch, ls.scaling, ls.lambda)
        end
    end
    ws = ls.solver_ws
    copyto!(ls.rhs, ls.b)
    if ws isa Krylov.CgWorkspace
        Krylov.cg!(ws, ls.A_scratch, ls.rhs; linesearch=true)
        copyto!(ls.x, ws.x)
        return cg_status(ws.stats)
    end
    # Not built by `makehessiansystem` for :cg, so no preallocated workspace
    (x, stats) = Krylov.cg(ls.A_scratch, ls.rhs; linesearch=true)
    copyto!(ls.x, x)
    return cg_status(stats)
end

function solve!(ls::DenseLinearSystem, ::StaticSymbol{:minres})
    copyto!(ls.A_scratch, ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            uniformscaling!(ls.A_scratch, ls.lambda)
        else
            damp_marquardt!(ls.A_scratch, ls.scaling, ls.lambda)
        end
    end
    ws = ls.solver_ws
    copyto!(ls.rhs, ls.b)
    if ws isa Krylov.MinresWorkspace
        Krylov.minres!(ws, ls.A_scratch, ls.rhs)
        copyto!(ls.x, ws.x)
        return ws.stats.solved ? Success : NotPositiveDefinite
    end
    (x, stats) = Krylov.minres(ls.A_scratch, ls.rhs)
    copyto!(ls.x, x)
    return stats.solved ? Success : NotPositiveDefinite
end

# ------------------------------------------------------------------------------------------------
# LSMR on J, for a Jacobian system (TP2)
#
# The damped step solves min ‖J x − r‖² + λ Σᵢ sᵢ xᵢ², with λ the optimizer's damping and s the
# Marquardt scaling (all ones without it). That is LSMR's regularized problem
# min ‖b − A x‖² + λₗ² ‖x‖²_F with A = J, b = r, λₗ = √λ and F = Diagonal(s), passed as LSMR's
# preconditioner N = F⁻¹. Nothing is augmented and JᵀJ is never formed. J's products are this
# package's plain loops (`jmul!`, `jtr!`), not BLAS (rules.md 23).

# The tolerances (`lsmr_tolerances`, in jacobiansystem.jl) are starting values, not measured. In T, the system's element
# type: the workspace's vectors, the inverse scaling, and r, narrowed into `rhs`.

struct JacobianLsmrWorkspace{W, T}
    ws::W
    invscale::Vector{T}     # N = F⁻¹, the inverse of the Marquardt scaling
    rhs::Vector{T}          # r, narrowed
end
makejacobianws(m, n, ::StaticSymbol{:lsmr}, ::Type{T}, ::Bool) where T = JacobianLsmrWorkspace(Krylov.LsmrWorkspace(m, n, Vector{T}), Vector{T}(undef, n), Vector{T}(undef, m))
builtfor(::JacobianLsmrWorkspace) = ":lsmr"

# J as an operator, and its adjoint, for Krylov
struct JacobianOperator{M}
    J::M
end
struct JacobianOperatorT{M}
    J::M
end
Base.size(A::JacobianOperator) = size(A.J)
Base.size(A::JacobianOperator, d::Integer) = size(A.J, d)
Base.size(A::JacobianOperatorT) = reverse(size(A.J))
Base.size(A::JacobianOperatorT, d::Integer) = size(A)[d]
Base.eltype(A::Union{JacobianOperator, JacobianOperatorT}) = eltype(A.J)
Base.adjoint(A::JacobianOperator) = JacobianOperatorT(A.J)
LinearAlgebra.mul!(y::AbstractVector, A::JacobianOperator, v::AbstractVector) = jmul!(y, A.J, v)
LinearAlgebra.mul!(y::AbstractVector, A::JacobianOperatorT, w::AbstractVector) = jtr!(y, A.J, w)

const LsmrJacobianSystem = Union{DenseJacobianLinearSystem{<:JacobianLsmrWorkspace}, SparseJacobianLinearSystem{<:JacobianLsmrWorkspace}}

function solve!(ls::LsmrJacobianSystem, ::StaticSymbol{:lsmr})
    A = ls.A
    J, r = A.J, A.r
    m, n = size(J)
    lw = ls.solver_ws
    vals = J isa SparseMatrixCSC ? nonzeros(J) : J
    if !(all(isfinite, vals) && all(isfinite, r))
        fill!(ls.x, NaN)
        return NotPositiveDefinite
    end
    lambda = max(ls.lambda, 0.0)
    # Undamped with fewer residuals than variables, J is rank deficient: report it as the QR solve
    # does. (LSMR cannot detect numerical rank deficiency with m ≥ n; it returns a least-squares
    # solution there.)
    lambda == 0 && m < n && return NotPositiveDefinite
    op = JacobianOperator(J)
    T = eltype(J)
    etol, axtol, btol, conlim = lsmr_tolerances(T)
    rhs = copyto!(lw.rhs, r)
    if lambda > 0 && !isnothing(ls.scaling)
        @inbounds for i in 1:n
            lw.invscale[i] = inv(ls.scaling[i])
        end
        Krylov.lsmr!(lw.ws, op, rhs; λ = T(sqrt(lambda)), N = Diagonal(lw.invscale), etol, axtol, btol, conlim)
    else
        Krylov.lsmr!(lw.ws, op, rhs; λ = T(sqrt(lambda)), etol, axtol, btol, conlim)
    end
    x = lw.ws.x
    (lw.ws.stats.solved && all(isfinite, x)) || return NotPositiveDefinite
    copyto!(ls.x, x)
    return Success
end
solve!(ls::DenseJacobianLinearSystem, ::StaticSymbol{:lsmr}) = solvermismatch(ls, :lsmr)
solve!(ls::SparseJacobianLinearSystem, ::StaticSymbol{:lsmr}) = solvermismatch(ls, :lsmr)

end # module KrylovExt
