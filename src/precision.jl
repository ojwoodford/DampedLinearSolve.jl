# The element type of a linear system: the type its matrix (A, or J of a Jacobian system) is stored, reduced and
# factorized in. `b`, `x` and `r` are always Float64, and so is every value a query returns.

# The element types a system can be built in; any other is an ArgumentError when the system is built (rules.md 3).
const SYSTEM_ELTYPES = (Float64, Float32)

function check_eltype(::Type{T}, what::AbstractString) where T
    T in SYSTEM_ELTYPES || throw(ArgumentError("$what has element type $T; a linear system is built in Float64 or Float32 (the element type of the prototype)."))
    return nothing
end

# Whether SuiteSparse's CHOLMOD and SPQR, as this Julia's SparseArrays binds them, compute in single precision. A
# fact about the library (checked by building and solving, on Julia 1.10 and on the release): before it offers Float32
# it converts a Float32 matrix to Float64, which would be a silent substitution and an allocation.
const SUITESPARSE_SINGLE = Float32 <: SparseArrays.CHOLMOD.VTypes

# The kinds of system, for decision 10's table of which solver offers which element type
#   :static, :dense, :sparse                 Hessian systems
#   :jacobian_static, :jacobian_dense, :jacobian_sparse
# The solvers each kind takes, in the order an error lists them
system_solvers(::Val{:static}) = (:auto, :cholesky, :ldl, :qr)
system_solvers(::Val{:dense}) = (:auto, :cholesky, :ldl, :qr, :cg, :minres, :pardiso)
system_solvers(::Val{:sparse}) = (:auto, :cholesky, :ldl, :qr, :cg, :minres, :pardiso)
system_solvers(::Val{:jacobian_static}) = (:auto, :qr)
system_solvers(::Val{:jacobian_dense}) = (:auto, :qr, :cholesky, :ldl, :lsmr)
system_solvers(::Val{:jacobian_sparse}) = (:auto, :qr, :lsmr)

# Whether `linsolver` offers element type T for a system of kind `kind`. Every solver offers Float64; Float32 only where
# a method says so (an extension adds methods for its solvers).
offers_eltype(::Type{Float64}, kind, s) = true
offers_eltype(::Type{Float32}, kind::Symbol, s) = offers_eltype(Float32, Val(kind), static(linsolver_symbol(s)))
offers_eltype(::Type{Float32}, ::Val, ::StaticSymbol) = false
const GenericDirect = Union{StaticSymbol{:auto}, StaticSymbol{:cholesky}, StaticSymbol{:ldl}, StaticSymbol{:qr}}
# StaticArrays, LAPACK (Cholesky, Bunch–Kaufman, pivoted and Householder QR, BLAS syrk) and LDLFactorizations are generic
offers_eltype(::Type{Float32}, ::Val{:static}, ::GenericDirect) = true
offers_eltype(::Type{Float32}, ::Val{:dense}, ::GenericDirect) = true
offers_eltype(::Type{Float32}, ::Val{:jacobian_static}, ::Union{StaticSymbol{:auto}, StaticSymbol{:qr}}) = true
offers_eltype(::Type{Float32}, ::Val{:jacobian_dense}, ::GenericDirect) = true
# Sparse: LDLᵀ is LDLFactorizations (generic); CHOLMOD (:cholesky) and SPQR (:qr) are SuiteSparse
offers_eltype(::Type{Float32}, ::Val{:sparse}, ::Union{StaticSymbol{:auto}, StaticSymbol{:ldl}}) = true
offers_eltype(::Type{Float32}, ::Val{:sparse}, ::Union{StaticSymbol{:cholesky}, StaticSymbol{:qr}}) = SUITESPARSE_SINGLE
offers_eltype(::Type{Float32}, ::Val{:jacobian_sparse}, ::Union{StaticSymbol{:auto}, StaticSymbol{:qr}}) = SUITESPARSE_SINGLE

const SYSTEM_KIND_NAMES = (static = "static Hessian system", dense = "dense Hessian system", sparse = "sparse Hessian system",
    jacobian_static = "static Jacobian system", jacobian_dense = "dense Jacobian system", jacobian_sparse = "sparse Jacobian system")

# Decision 10: a solver that does not offer the system's element type is an ArgumentError when the system is built,
# naming the solvers that do (of those whose packages are loaded). No silent substitution, `:auto` included.
function check_offers(::Type{T}, kind::Symbol, linsolver) where T
    offers_eltype(T, kind, linsolver) && return nothing
    s = linsolver_symbol(linsolver)
    ok = [x for x in system_solvers(Val(kind)) if validatelinsolver(x) && offers_eltype(T, kind, x)]
    why = s in (:cholesky, :qr, :auto) && kind in (:sparse, :jacobian_sparse) && !SUITESPARSE_SINGLE ?
        " (SuiteSparse's CHOLMOD and SPQR are Float64 only in SparseArrays $(pkgversion(SparseArrays)), on Julia $VERSION)" : ""
    throw(ArgumentError("linsolver = :$s does not offer $T for a $(SYSTEM_KIND_NAMES[kind])$why. " *
        (isempty(ok) ? "No loaded solver offers $T for it." : "The solvers that offer $T for it are $(join((":$x" for x in ok), ", ", " and ")).") *
        " A Float64 prototype gives a Float64 system, which every solver offers."))
end
