# The normal equations of the dense Jacobian system: `linsolver = :cholesky` or `:ldl` on a
# `DenseJacobianLinearSystem`. The model writes J and r as for QR; the solve forms the upper triangle
# of JᵀJ once per linearization (BLAS `syrk`) and, per damping value, factors JᵀJ + λD with the dense
# Hessian system's own solver: Cholesky for `:cholesky`, Bunch–Kaufman for `:ldl`.

# The solver's workspace (the system's `solver_ws`): a dense Hessian system of the n variables, built
# by `makehessiansystem` for the same solver (so `:ldl` carries its `DenseLDLWorkspace`). Its `A`
# holds the product, upper triangle only (the lower stays zero, as built); its `b` and `x` are its
# own, copied in and out around each solve. Shared by a system and its `copy_structure` copy, like
# the QR scratch: the product is a system's own only while the scratch's owner is that system's
# token (`factorized`), and is formed again otherwise.
struct JacobianNormalWorkspace{S, H}
    hs::H
end
# In `T`, the element type of J: the product JᵀJ and its factorization
function JacobianNormalWorkspace{S}(n::Integer, ::Type{T}=Float64) where {S, T}
    hs = makehessiansystem(zeros(T, n, n), static(S))
    return JacobianNormalWorkspace{S, typeof(hs)}(hs)
end
makejacobianws(m, n, ::StaticSymbol{:cholesky}, ::Type{T}) where T = JacobianNormalWorkspace{:cholesky}(n, T)
makejacobianws(m, n, ::StaticSymbol{:ldl}, ::Type{T}) where T = JacobianNormalWorkspace{:ldl}(n, T)
builtfor(::JacobianNormalWorkspace{S}) where S = ":$S"

# No QR is done, so the QR scratch holds no copy of [J r] and no stacked matrix: it is kept only for
# its `owner` token, which says whose product the shared workspace holds
qrscratch(m, n, ::Union{StaticSymbol{:cholesky}, StaticSymbol{:ldl}}, ::Type{T}) where T = JacobianQRScratch{T}(0, 0)

const NormalJacobianSystem = DenseJacobianLinearSystem{<:JacobianNormalWorkspace}

# Form the upper triangle of JᵀJ into the workspace, for this system's current J, and record that the
# workspace holds it. A non-finite J or r is recorded instead, and nothing is formed.
function form_normal!(ls::NormalJacobianSystem)
    A = ls.A
    ws = A.ws
    ws.stale = false
    ws.scratch.owner = ws.token
    ws.nonfinite = !(all(isfinite, A.J) && all(isfinite, A.r))
    ws.nonfinite && return nothing
    J = A.J
    LinearAlgebra.BLAS.syrk!('U', 'T', one(eltype(J)), J, zero(eltype(J)), ls.solver_ws.hs.A)
    ws.nfactor += 1
    return nothing
end

function normal_solve!(ls::NormalJacobianSystem, s::StaticSymbol)
    A = ls.A
    ws = A.ws
    m, n = size(A.J)
    current = factorized(ws)
    nonfinite = current ? ws.nonfinite : !(all(isfinite, A.J) && all(isfinite, A.r))
    if nonfinite
        current || form_normal!(ls)   # records the non-finite linearization; forms nothing
        fill!(ls.x, NaN)
        return NotPositiveDefinite
    end
    # Fewer residuals than variables, undamped: JᵀJ is singular. Reported without factorizing, as
    # QR and LSMR report it.
    !(ls.lambda > 0) && m < n && return NotPositiveDefinite
    current || form_normal!(ls)
    hs = ls.solver_ws.hs
    copyto!(hs.b, ls.b)
    hs.lambda = ls.lambda
    hs.scaling = ls.scaling
    status = solve!(hs, s)
    status == NotPositiveDefinite || copyto!(ls.x, hs.x)   # an Indefinite solve has solved, too
    return status
end

solve!(ls::DenseJacobianLinearSystem{<:JacobianNormalWorkspace{:cholesky}}, s::StaticSymbol{:cholesky}) = normal_solve!(ls, s)
solve!(ls::DenseJacobianLinearSystem{<:JacobianNormalWorkspace{:ldl}}, s::StaticSymbol{:ldl}) = normal_solve!(ls, s)
# A system built for another solver (QR, LSMR, or the other of the two) asked for one of these is a
# solver mismatch, raised by the catch-all `solve!` of the dense Jacobian system (jacobiansystem.jl): a
# method of its own here would add one to the methods a call with a solver known only at run time
# must consider, and the call's status would then be boxed (an allocation)

# The normal equations work with cond(JᵀJ) = cond(J)², so, as for a factorization of the Hessian, a
# damping below about eps times the diagonal is not resolved
dampingfloor(ls::NormalJacobianSystem) = eps(Float64)
