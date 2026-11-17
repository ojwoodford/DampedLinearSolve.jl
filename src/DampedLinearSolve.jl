"""
    DampedLinearSolve

Solve, without allocating, the damped linear problems a second-order optimizer poses,
`(H + λD) x = −g` and `min ‖J x + r‖² + λ‖x‖²_D`, and report the outcome well enough for the
optimizer to decide what to do next. It applies the damping it is given and never chooses it.

The contract a linear system meets is in the documentation ("The linear-system contract"); the
systems for `SMatrix`, `Matrix` and `SparseMatrixCSC` Hessians and Jacobians are built by
[`makelinearsystem`](@ref) and [`maketallsystem`](@ref), and [`check_linear_system`](@ref) checks
an implementation.
"""
module DampedLinearSolve

using LinearAlgebra, SparseArrays, StaticArrays, Static, LDLFactorizations

# The public API: the linear-system contract, its constructors and its conformance check. Every
# exported name is documented in the API reference, and nothing else is exported. The test suite
# pins this list (test/publicapi.jl).
export AbstractLinearSystem, SolveStatus, Success, NotPositiveDefinite, Indefinite
export makelinearsystem, maketallsystem, structurekey
export checkhessian, checkjacobian, check_jacobian_linsolver, jacobian_ncols, jacobian_nrows
export checklinsolver, validatelinsolver
export getA, getb, getx, zero!, getJ, getr, gradient!, markstale!
export dampen!, solve!
export max_diag, getdiag!, quadform, quadratic_model_change, dampingfloor, isfinitesystem
export copy_structure, sharex
export check_linear_system, conformance_matrix, conformance_fill!, conformance_fillb!, conformance_traits

# Utilities
include("utils.jl")

# The contract, and the systems with a square Hessian: static, dense and sparse
include("linearsystem.jl")
include("linearsolver.jl")

# The tall systems, from a Jacobian: dense, sparse and static
include("tallsystem.jl")
include("sparsetallsystem.jl")
include("statictallsystem.jl")

# Conformance check for linear systems
include("conformance.jl")

# API docstrings
include("docstrings.jl")

end # module DampedLinearSolve
