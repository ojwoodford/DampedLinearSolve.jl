# The held factors of a dense Jacobian system's QR solve, public (SP-D12): for a caller that refines a larger
# least-squares problem of which this system's is a part (BlockLinearSystems' whole-system refinement of its
# QR-eliminated systems, SP-B7).
#
# The factors are those of the matrix the system stands for, as filled, at the damping of its last solve: M = J
# undamped, M = [J; √(λD)] damped, with M = Q [R; 0]. The solve factorizes M S (S the internal conditioning scale), as
# Q R̂, so R = R̂ S⁻¹: the scale stays internal (it has one owner, conditioning.jl), and its recomputation (SP-D9, which
# rescales R̂'s columns by powers of two) changes neither R⁻¹v nor R⁻ᵀv. Q is applied in its two stages, as the solve
# applies it: the reflectors of the QR of J S on rows 1:m, then, damped, those of the stacked [R₁; √(λD) S] on rows
# 1:n and m+1:m+n. Everything is applied in Float64 to Float64 vectors, the T factors widened, by the kernels the solve
# and the refinement use (jacobiansystem.jl, qr_refinement.jl).
#
# Only m ≥ n: with fewer rows the damped stacked matrix has n − m zero rows, which its reflectors fill, so its Q does not
# act on the m + n rows of M.

# Q (p × p, p = m undamped, m + n damped) and R (n × n). Internal types: what they support is in `qrfactors`'s docstring.
struct HeldQRQ{T}
    Jr::Matrix{T}
    tau::Vector{T}
    S::Matrix{T}
    tau2::Vector{T}
    m::Int
    n::Int
    damped::Bool
end
struct HeldQRR{T}
    R::Matrix{T}
    s::Vector{Float64}
    n::Int
    condition::Bool
end
# Q' and R'
struct HeldQRAdjoint{F}
    parent::F
end
Base.adjoint(Q::HeldQRQ) = HeldQRAdjoint(Q)
Base.adjoint(R::HeldQRR) = HeldQRAdjoint(R)
Base.size(Q::HeldQRQ) = (p = Q.damped ? Q.m + Q.n : Q.m; (p, p))
Base.size(R::HeldQRR) = (R.n, R.n)
Base.size(F::HeldQRAdjoint) = size(F.parent)

function checklength(F, v::AbstractVector)
    p = size(F)[1]
    length(v) == p || throw(DimensionMismatch("the vector has length $(length(v)); the factor is $p × $p"))
    return nothing
end

# v = Qᵀv: the first stage, then the second
function LinearAlgebra.lmul!(Qt::HeldQRAdjoint{<:HeldQRQ}, v::AbstractVector{Float64})
    Q = Qt.parent
    checklength(Q, v)
    m, n = Q.m, Q.n
    apply_reflectors!(view(v, 1:m), Q.Jr, Q.tau, n)
    Q.damped && apply_reflectors!(v, Q.S, Q.tau2, n, n, m - n)
    return v
end
# v = Q v: the second stage, then the first
function LinearAlgebra.lmul!(Q::HeldQRQ, v::AbstractVector{Float64})
    checklength(Q, v)
    m, n = Q.m, Q.n
    Q.damped && apply_reflectors_reverse!(v, Q.S, Q.tau2, n, n, m - n)
    apply_reflectors_reverse!(view(v, 1:m), Q.Jr, Q.tau, n)
    return v
end
# v = R⁻¹v = S R̂⁻¹v
function LinearAlgebra.ldiv!(R::HeldQRR, v::AbstractVector{Float64})
    checklength(R, v)
    backsubstitute!(v, R.R, R.n)
    return unscale!(v, R.s, R.condition)
end
# v = R⁻ᵀv = R̂⁻ᵀ(S v)
function LinearAlgebra.ldiv!(Rt::HeldQRAdjoint{<:HeldQRR}, v::AbstractVector{Float64})
    R = Rt.parent
    checklength(R, v)
    unscale!(v, R.s, R.condition)
    return forwardsubstitute_transpose!(v, R.R, R.n)
end

const QRFACTORS_SUPPORT = "qrfactors is offered by the dense Jacobian system (a Matrix or an integer Jacobian prototype) built for linsolver = :auto or :qr, with at least as many residuals as variables"

function qrfactors(ls::DenseJacobianLinearSystem{W, T}) where {W, T}
    A = ls.A
    ws = A.ws
    sc = ws.scratch
    m, n = size(A.J)
    size(sc.Jr, 1) == m || throw(ArgumentError("This dense Jacobian system was built for a solver other than QR (the normal equations or LSMR), and holds no QR factors. $QRFACTORS_SUPPORT."))
    m >= n || throw(ArgumentError("This dense Jacobian system has fewer residuals ($m) than variables ($n). $QRFACTORS_SUPPORT."))
    if !ws.stale && sc.owner != ws.token
        error("The QR factors of this system are no longer held: a system sharing its workspace (copy_structure, sharex) has been solved since. Solve this system again first.")
    end
    ws.held && factorized(ws) || error("This system holds no QR factors: it has not been solved by QR since its fill (zero! or markstale!), or its last solve did not return Success.")
    damped = ws.helddamped
    Q = HeldQRQ{T}(sc.Jr, sc.tau, sc.S, sc.tau2, m, n, damped)
    R = HeldQRR{T}(damped ? sc.S : sc.Jr, ws.cs, n, ls.condition)
    return Q, R
end
qrfactors(ls::AbstractLinearSystem) = throw(ArgumentError("A $(nameof(typeof(ls))) does not offer its factors. $QRFACTORS_SUPPORT."))
