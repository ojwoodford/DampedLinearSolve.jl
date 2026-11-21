# The linsolver a sparse system's solver workspace was built for by `makesparsews`, as text, for
# the error raised when `solve!` is asked for another solver. Extensions add methods for their
# workspace types.
builtfor(::LDLFactorizations.LDLFactorization) = ":auto or :ldl"
builtfor(::SparseArrays.CHOLMOD.Factor) = ":cholesky"
builtfor(::Nothing) = ":qr"
builtfor(ws) = "an unknown solver (workspace type $(typeof(ws)))"

# The error for a system solved with a solver it was not built for: a system is solved with the
# linsolver it was built for, so only a caller that mixes the two can reach it.
solverworkspace(ls) = ls.solver_ws
function solvermismatch(ls, asked::Symbol)
    throw(ArgumentError("Internal inconsistency: solve! was asked for linsolver = :$asked, but this " *
        "$(nameof(typeof(ls))) was built by makehessiansystem or makejacobiansystem for linsolver = $(builtfor(solverworkspace(ls))). " *
        "A linear system must be solved with the linsolver it was built for."))
end

validatelinsolver(::StaticSymbol{:auto}) = true
validatelinsolver(::StaticSymbol{:cholesky}) = true
validatelinsolver(::StaticSymbol{:ldl}) = true
validatelinsolver(::StaticSymbol{:qr}) = true
validatelinsolver(::StaticSymbol{S}) where S = false
validatelinsolver(s::Symbol) = validatelinsolver(static(s))

# Checks on the Hessian prototype `H` of a system of `n` variables, run before the system is built
checkhessian(H, n::Integer) = nothing
function checkhessian(H::AbstractMatrix, n::Integer)
    size(H) == (n, n) || throw(DimensionMismatch("Hessian prototype H has size $(size(H)), but the variables have $n degrees of freedom, so H must be $n×$n."))
    return nothing
end

function checklinsolver(s::Symbol)
    validatelinsolver(s) && return true
    if s === :pardiso
        throw(ArgumentError("Linear solver :pardiso requires Pardiso.jl. Please run `using Pardiso` first."))
    elseif s in (:cg, :minres, :lsmr)
        throw(ArgumentError("Linear solver :$s requires Krylov.jl. Please run `using Krylov` first."))
    end
    throw(ArgumentError("Unknown linear solver: $s"))
end
checklinsolver(s::StaticSymbol) = checklinsolver(dynamic(s))

# Indefiniteness tests on a factorization's pivots. A pivot counts as negative when it is below
# `-rtol` times the largest pivot magnitude, so the test is independent of the matrix's scale. `rtol` is 1e-6 for a
# Float64 factorization and its single-precision counterpart for Float32 (`indefinite_rtol`, precision.jl).

# Eigenvalues of the symmetric 2x2 block [a b; b c], smallest first
@inline function eig2x2(a, b, c)
    m = 0.5 * (a + c)
    r = hypot(0.5 * (a - c), b)
    return m - r, m + r
end

# Indefiniteness of the block-diagonal factor of a (non-rook) Bunch-Kaufman factorization, read
# straight from the packed factor `LD` and pivots `ipiv` as LAPACK's `sytrf` leaves them: a
# negative `ipiv[k]` starts a 2x2 block at (k, k+1), whose off-diagonal entry is stored in the
# triangle `uplo`. Reading the packed factor avoids materialising `bk.D`.
function is_indefinite_packed(LD::AbstractMatrix, ipiv::AbstractVector, uplo::Char, rtol::Float64=indefinite_rtol(eltype(LD)))
    n = length(ipiv)
    # The largest pivot magnitude, and the smallest pivot (eigenvalue, for 2x2 blocks)
    maxabs = 0.0
    minpivot = Inf
    k = 1
    @inbounds while k <= n
        if ipiv[k] < 0 && k < n
            off = uplo == 'U' ? LD[k, k+1] : LD[k+1, k]
            lo, hi = eig2x2(LD[k, k], off, LD[k+1, k+1])
            maxabs = max(maxabs, abs(lo), abs(hi))
            minpivot = min(minpivot, lo)
            k += 2
        else
            d = LD[k, k]
            maxabs = max(maxabs, abs(d))
            minpivot = min(minpivot, d)
            k += 1
        end
    end
    return minpivot < -rtol * maxabs
end

function is_indefinite(bk::BunchKaufman, rtol::Float64=indefinite_rtol(eltype(bk)))
    bk.rook || return is_indefinite_packed(bk.LD, bk.ipiv, bk.uplo, rtol)
    # Rook pivoting stores its pivots differently, so go through `bk.D`, materialised once
    # (it rebuilds the block-diagonal matrix on every access)
    D = bk.D
    n = length(bk.ipiv)
    maxabs = 0.0
    minpivot = Inf
    k = 1
    @inbounds while k <= n
        if bk.ipiv[k] < 0
            lo, hi = eig2x2(D[k, k], D[k, k+1], D[k+1, k+1])
            maxabs = max(maxabs, abs(lo), abs(hi))
            minpivot = min(minpivot, lo)
            k += 2
        else
            d = D[k, k]
            maxabs = max(maxabs, abs(d))
            minpivot = min(minpivot, d)
            k += 1
        end
    end
    return minpivot < -rtol * maxabs
end

function is_indefinite(F::LDLFactorizations.LDLFactorization, rtol::Float64=indefinite_rtol(eltype(F.d)))
    maxabs = 0.0
    minpivot = Inf
    @inbounds for d in F.d
        maxabs = max(maxabs, abs(d))
        minpivot = min(minpivot, d)
    end
    return minpivot < -rtol * maxabs
end

@inline function has_positive_pivots(C::Cholesky)
    @inbounds for i in 1:size(C.factors, 1)
        if !(C.factors[i, i] > 0)
            return false
        end
    end
    return true
end

function try_cholesky!(x::StaticVector, A::StaticMatrix, b::StaticVector)
    A_ = SMatrix(A)
    b_ = SVector(b)
    cholfac = cholesky(Symmetric(A_); check=false)
    if issuccess(cholfac) && has_positive_pivots(cholfac)
        x .= cholfac \ b_
        return Success
    end
    return NotPositiveDefinite
end

function try_cholesky!(x, A, b)
    cholfac = cholesky(Symmetric(A); check=false)
    if issuccess(cholfac)
        ldiv!(x, cholfac, b)
        return Success
    end
    return NotPositiveDefinite
end

# Linear solve dispatching on solver symbol: :auto, :cholesky, :ldl, :qr
solve!(ls) = solve!(ls, static(:auto))

# StaticLinearSystem: A_ is the damped matrix in `T`; the damping is formed in Float64 and narrowed once
@inline function damped_static(ls::StaticLinearSystem{N, N2, T}) where {N, N2, T}
    A_ = SMatrix(ls.A)
    if ls.lambda > 0
        if isnothing(ls.scaling)
            A_ = A_ + convert(T, ls.lambda) * I
        else
            A_ = A_ + Diagonal(SVector{N, T}(ls.lambda .* ls.scaling))
        end
    end
    return A_
end
@inline static_rhs(ls::StaticLinearSystem{N, N2, T}) where {N, N2, T} = SVector{N, T}(ls.b)

function solve!(ls::StaticLinearSystem, ::Union{StaticSymbol{:auto}, StaticSymbol{:cholesky}})
    A_ = damped_static(ls)
    cholfac = cholesky(Symmetric(A_); check=false)
    if issuccess(cholfac) && has_positive_pivots(cholfac)
        ls.x .= cholfac \ static_rhs(ls)
        return Success
    end
    return NotPositiveDefinite
end

# Static `:ldl`: Cholesky when positive definite, otherwise LU (StaticArrays has no
# Bunch-Kaufman, and its `eigen` allocates above 3x3). Once Cholesky has failed, a nonsingular
# symmetric matrix has a negative eigenvalue, so the solve is reported as `Indefinite`, as the
# dense and sparse `:ldl` do.
function solve!(ls::StaticLinearSystem, ::StaticSymbol{:ldl})
    A_ = damped_static(ls)
    cholfac = cholesky(Symmetric(A_); check=false)
    if issuccess(cholfac) && has_positive_pivots(cholfac)
        ls.x .= cholfac \ static_rhs(ls)
        return Success
    end
    all(isfinite, A_) || return NotPositiveDefinite
    lufac = lu(A_; check=false)
    issuccess(lufac) || return NotPositiveDefinite
    x_ = lufac \ static_rhs(ls)
    all(isfinite, x_) || return NotPositiveDefinite
    ls.x .= x_
    return Indefinite
end

# Static `:qr` uses LU: StaticArrays has no pivoted QR. A singular or non-finite matrix is a failed
# solve, not an exception.
function solve!(ls::StaticLinearSystem, ::StaticSymbol{:qr})
    A_ = damped_static(ls)
    lufac = lu(A_; check=false)
    issuccess(lufac) || return NotPositiveDefinite
    x_ = lufac \ static_rhs(ls)
    all(isfinite, x_) || return NotPositiveDefinite
    ls.x .= x_
    return Success
end

# DenseLinearSystem. `prepare_dense!` writes the conditioned, damped matrix S(A + λD)S into `A_scratch` and the
# right-hand side S b into `rhs` (narrowed to `T`); the factorization solves `rhs` in place, and `finish_dense!` widens
# x = S y into `x`. Unconditioned, S = I. The damping of entry j, λ·D_jj·s_j², is formed in Float64 and added there,
# and the sum narrowed once.
function prepare_dense!(ls::DenseLinearSystem{T}) where T
    A, M, n = ls.A, ls.A_scratch, length(ls.b)
    lambda, scaling = ls.lambda, ls.scaling
    if ls.condition
        s = ls.cs
        @inbounds for j in 1:n
            s[j] = conditioning_scale(A[j, j], T)
        end
        @inbounds for j in 1:n
            sj = T(s[j])
            for i in 1:n
                M[i, j] = (A[i, j] * T(s[i])) * sj
            end
        end
        if lambda > 0
            @inbounds for j in 1:n
                M[j, j] += (isnothing(scaling) ? lambda : lambda * scaling[j]) * (s[j] * s[j])
            end
        end
        @inbounds for i in 1:n
            ls.rhs[i] = s[i] * ls.b[i]
        end
    else
        copyto!(M, A)
        if lambda > 0
            if isnothing(scaling)
                uniformscaling!(M, lambda)
            else
                damp_marquardt!(M, scaling, lambda)
            end
        end
        copyto!(ls.rhs, ls.b)
    end
    return ls
end
# x from the solved `rhs`; a non-finite solution is a failed solve (whether a NaN fails a factorization
# depends on the LAPACK)
function finish_dense!(ls)
    if ls.condition
        @inbounds for i in eachindex(ls.x)
            ls.x[i] = ls.cs[i] * ls.rhs[i]
        end
    else
        copyto!(ls.x, ls.rhs)
    end
    return all(isfinite, ls.x)
end

function solve!(ls::DenseLinearSystem, ::Union{StaticSymbol{:auto}, StaticSymbol{:cholesky}})
    prepare_dense!(ls)
    cholfac = cholesky!(Symmetric(ls.A_scratch); check=false)
    if issuccess(cholfac)
        ldiv!(cholfac, ls.rhs)
        return finish_dense!(ls) ? Success : NotPositiveDefinite
    end
    return NotPositiveDefinite
end

const BlasInt = LinearAlgebra.BlasInt

# Preallocated pivots and LAPACK work array for dense Bunch-Kaufman (`dsytrf`, `ssytrf`), so the
# factorization doesn't allocate in the loop
struct DenseLDLWorkspace{T}
    ipiv::Vector{BlasInt}
    work::Vector{T}
end
DenseLDLWorkspace(n::Integer) = DenseLDLWorkspace{Float64}(n)

for (T, sytrf) in ((Float64, :dsytrf_), (Float32, :ssytrf_))
    @eval begin
        function DenseLDLWorkspace{$T}(n::Integer)
            ipiv = Vector{BlasInt}(undef, n)
            work = Vector{$T}(undef, 1)
            if n > 0
                # Workspace size query (lwork = -1): only the dimensions are read
                A = Matrix{$T}(undef, n, n)
                info = Ref{BlasInt}()
                ccall((LinearAlgebra.BLAS.@blasfunc($sytrf), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                       Ptr{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong),
                      'U', n, A, max(1, n), ipiv, work, BlasInt(-1), info, 1)
                resize!(work, max(1, Int(work[1])))
            end
            return DenseLDLWorkspace{$T}(ipiv, work)
        end

        # Bunch-Kaufman factorization of the upper triangle of `A` in place; returns LAPACK's `info`
        # (positive when the block-diagonal factor is exactly singular)
        function sytrf_ws!(A::Matrix{$T}, ws::DenseLDLWorkspace{$T})
            n = LinearAlgebra.checksquare(A)
            length(ws.ipiv) == n || throw(DimensionMismatch("DenseLDLWorkspace is for size $(length(ws.ipiv)), got $n"))
            n == 0 && return BlasInt(0)
            info = Ref{BlasInt}()
            ccall((LinearAlgebra.BLAS.@blasfunc($sytrf), LinearAlgebra.BLAS.libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong),
                  'U', n, A, max(1, stride(A, 2)), ws.ipiv, ws.work, BlasInt(length(ws.work)), info, 1)
            LinearAlgebra.LAPACK.chkargsok(info[])
            return info[]
        end
    end
end

function solve!(ls::DenseLinearSystem{T, DenseLDLWorkspace{T}}, ::StaticSymbol{:ldl}) where T
    prepare_dense!(ls)
    sytrf_ws!(ls.A_scratch, ls.solver_ws) == 0 || return NotPositiveDefinite
    LinearAlgebra.LAPACK.sytrs!('U', ls.A_scratch, ls.solver_ws.ipiv, ls.rhs)
    finish_dense!(ls) || return NotPositiveDefinite
    return is_indefinite_packed(ls.A_scratch, ls.solver_ws.ipiv, 'U') ? Indefinite : Success
end

# Without a preallocated workspace (a system not built by `makehessiansystem`)
function solve!(ls::DenseLinearSystem, ::StaticSymbol{:ldl})
    prepare_dense!(ls)
    bk = bunchkaufman!(Symmetric(ls.A_scratch); check=false)
    if issuccess(bk)
        ldiv!(bk, ls.rhs)
        finish_dense!(ls) || return NotPositiveDefinite
        if is_indefinite(bk)
            return Indefinite
        end
        return Success
    end
    return NotPositiveDefinite
end

function solve!(ls::DenseLinearSystem, ::StaticSymbol{:qr})
    prepare_dense!(ls)
    qrfac = qr!(ls.A_scratch, ColumnNorm())
    ldiv!(qrfac, ls.rhs)
    return finish_dense!(ls) ? Success : NotPositiveDefinite
end

# SparseLinearSystem. `prepare_sparse!` writes the conditioned, damped matrix S(A + λD)S into `Aw` (A's pattern, its
# own values) and S b into `rhs`, as `prepare_dense!` does for the dense system; `finish_sparse!` widens x = S y from
# the solution `y`. A is never written.
function prepare_sparse!(ls::SparseLinearSystem{T}) where T
    A, w, n = ls.A, nonzeros(ls.Aw), length(ls.b)
    nzv, rv = nonzeros(A), rowvals(A)
    lambda, scaling = ls.lambda, ls.scaling
    if ls.condition
        s = ls.cs
        @inbounds for (j, idx) in enumerate(ls.diag_inds)
            s[j] = conditioning_scale(nzv[idx], T)
        end
        @inbounds for j in 1:n
            sj = T(s[j])
            for k in nzrange(A, j)
                w[k] = (nzv[k] * T(s[rv[k]])) * sj
            end
        end
        if lambda > 0
            @inbounds for (j, idx) in enumerate(ls.diag_inds)
                w[idx] += (isnothing(scaling) ? lambda : lambda * scaling[j]) * (s[j] * s[j])
            end
        end
        @inbounds for i in 1:n
            ls.rhs[i] = s[i] * ls.b[i]
        end
    else
        copyto!(w, nzv)
        if lambda > 0
            @inbounds for (j, idx) in enumerate(ls.diag_inds)
                w[idx] += isnothing(scaling) ? lambda : lambda * scaling[j]
            end
        end
        copyto!(ls.rhs, ls.b)
    end
    return ls
end
function finish_sparse!(ls::SparseLinearSystem, y::AbstractVector)
    if ls.condition
        @inbounds for i in eachindex(ls.x)
            ls.x[i] = ls.cs[i] * y[i]
        end
    else
        copyto!(ls.x, y)
    end
    return all(isfinite, ls.x)
end

function solve!(ls::SparseLinearSystem, ::Union{StaticSymbol{:auto}, StaticSymbol{:ldl}})
    prepare_sparse!(ls)
    F = ldl_factorize!(ls.Aw, ls.solver_ws)
    if LDLFactorizations.factorized(F)
        ldiv!(F, ls.rhs)
        finish_sparse!(ls, ls.rhs) || return NotPositiveDefinite
        if is_indefinite(F)
            return Indefinite
        end
        return Success
    end
    return NotPositiveDefinite
end

# The damping is on the diagonal of `Aw`, for uniform damping too: λS² is not uniform, so CHOLMOD's `shift` does not
# apply to a conditioned system, and one route serves both
function solve!(ls::SparseLinearSystem, ::StaticSymbol{:cholesky})
    prepare_sparse!(ls)
    cholfac = cholesky!(ls.solver_ws, Symmetric(ls.Aw, :U); check=false)
    if issuccess(cholfac)
        @static if VERSION >= v"1.12.0-"   # COV_EXCL_LINE: Julia 1.10 counts this line, but it runs at macro expansion
            ldiv!(ls.sol, cholfac, ls.rhs)   # in place; this method first appears in Julia 1.12
        else
            ls.sol .= cholfac \ ls.rhs
        end
        return finish_sparse!(ls, ls.sol) ? Success : NotPositiveDefinite
    end
    return NotPositiveDefinite
end

function solve!(ls::SparseLinearSystem, ::StaticSymbol{:qr})
    prepare_sparse!(ls)
    ls.sol .= qr(ls.Aw) \ ls.rhs
    return finish_sparse!(ls, ls.sol) ? Success : NotPositiveDefinite
end

# Method dispatch for a Symbol
solve!(ls::AbstractLinearSystem, s::Symbol) = solve!(ls, static(s))

# Symmetric matrix inversion
invsym(A::AbstractMatrix) = inv(bunchkaufman(A))
invsym(A::StaticMatrix) = Size(A)[1] <= 14 ? inv(A) : inv(bunchkaufman(A))
