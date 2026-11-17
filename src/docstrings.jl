"""
    SolveStatus

Enum returned by [`solve!`](@ref) indicating the status of the linear system solution:
- [`Success`](@ref): The linear system was solved successfully.
- [`NotPositiveDefinite`](@ref): The matrix was found not to be positive definite during Cholesky or LDL factorization.
- [`Indefinite`](@ref): The matrix was detected to be indefinite or negative curvature was encountered during solving.
"""
SolveStatus

"""
    Success

[`SolveStatus`](@ref) indicating that the linear system was solved successfully.
"""
Success

"""
    NotPositiveDefinite

[`SolveStatus`](@ref) indicating that the matrix failed a positive-definiteness test (e.g. non-positive pivot in Cholesky).
"""
NotPositiveDefinite

"""
    Indefinite

[`SolveStatus`](@ref) indicating that the matrix was detected to be indefinite or negative curvature was encountered.
"""
Indefinite

"""
    zero!(ls)

Prepares the linear system `ls` for a fresh linearization: zeroes the gradient vector and the
Hessian matrix that `fdf!` accumulates into. For the tall system of a
[`JacobianObjective`](@ref), whose `resjac!` sets `J` and `r` rather than adding to them, it
only invalidates the cached factorization and column norms, and zeroes the gradient.

# Example
```julia
ls = SecondOrderOptimization.DenseLinearSystem(3)
zero!(ls)
```
"""
zero!

"""
    solve!(ls, solver=static(:auto)) -> SolveStatus

Solves the linear system `A * x = b` in-place using the specified `solver` (a `Symbol` or a `StaticSymbol`) and returns the [`SolveStatus`](@ref) enum (`Success`, `NotPositiveDefinite`, `Indefinite`).

# Supported Solver Identifiers
- Core solvers: `:auto`, `:cholesky`, `:ldl`, `:qr`. (`OptimizationOptions` also accepts the enum constants [`auto_solver`](@ref), [`cholesky_solver`](@ref), [`ldl_solver`](@ref), [`qr_solver`](@ref), and holds the solver as a `StaticSymbol`.)
- Extension solvers (when package is loaded):
  - `using Krylov`: `:cg` (Conjugate Gradient), `:minres` (MINRES). On sparse systems, `:cg` and `:minres` are Jacobi (diagonal) preconditioned whenever the damped diagonal is positive.
  - `using Pardiso`: `:pardiso` (Intel MKL PARDISO or Panua Pardiso multi-threaded direct sparse solver).

Note: On `StaticLinearSystem`, `:qr` executes static LU decomposition via `\`, and `:ldl` uses Cholesky, falling back to LU when the matrix is not positive definite (the solve is then reported as `Indefinite`).

# Example
```julia
ls = SecondOrderOptimization.makelinearsystem([2.0 0.0; 0.0 4.0], :auto)
ls.b .= [4.0, 8.0]
solve!(ls, :cholesky)   # Success; ls.x == [2.0, 2.0]
```
"""
solve!

"""
    validatelinsolver(solver::Union{Symbol, StaticSymbol, OptimizationLinearSolver}) -> Bool

Returns `true` if `solver` is a supported linear solver backend, or `false` otherwise.
Extension solvers (`:cg`, `:minres` and `:lsmr`, or `:pardiso`) register validation methods when their respective packages (`Krylov.jl` or `Pardiso.jl`) are loaded.
"""
validatelinsolver

"""
    AbstractLinearSystem

Supertype of the linear systems the iterators solve: the Hessian `A` and gradient `b` that `fdf!`
fills, the step `x`, and the damping to apply. The built-in systems are `StaticLinearSystem`,
`DenseLinearSystem`, `SparseLinearSystem` and [`DenseTallLinearSystem`](@ref) (for a
[`JacobianObjective`](@ref): `A = JᵀJ` held implicitly as `J` and `r`, `b = Jᵀr`, and the
iterators' `fdf!` a wrapper that calls `resjac!` and forms `b`), and, with BlockLinearSystems.jl loaded,
`DenseSchurLinearSystem` and `SparseSchurLinearSystem`. A new system type must provide the functions
below (those not exported are extended as `SecondOrderOptimization.name`), and should pass
[`check_linear_system`](@ref).

# Linear system interface

| Function | Contract |
|:---|:---|
| `makelinearsystem(H, linsolver)`, `makelinearsystem(n, linsolver)` | Builds the system from the Hessian prototype `H` (see below for which type each `H` gives), or, without one, for `n` variables: static when `n` is a `StaticInt`, dense when it is an `Int`. `linsolver` is used to preallocate a factorization or Krylov workspace; a sparse system must then be solved with that same solver. |
| `getA(ls)`, `getb(ls)`, `getx(ls)` | `A` is the Hessian storage and `b` the gradient vector that `fdf!(b, A, x)` fills; `x` is the vector `solve!` writes. `b` and `x` have length `n`. The iterators read and negate `x`, and dogleg writes its own step into it. `A` must support `dot(v, A, v)` (`LinearAlgebra`'s three-argument `dot`), or have a method of the internal `quadform(A, v)`, for `v` of the type of `b`. |
| `zero!(ls)` | Prepares the system for a fresh linearization, keeping its structure (pattern, workspaces): storage that `fdf!` accumulates into (`A`, `b`) is zeroed. A tall system only invalidates its factorization and column norms (and zeroes `b`), since `resjac!` overwrites `J` and `r`. |
| `dampen!(ls, λ)`, `dampen!(ls, λ, s)` | Records damping without changing `A`: `solve!` then solves with `A + λI`, or `A + λ Diagonal(s)` (`s` in the units of `diag(A)`, not squared). Each call replaces the damping set before, including the scaling; `dampen!(ls, λ, nothing)` is `dampen!(ls, λ)`, and `λ = 0` is undamped. |
| `solve!(ls, linsolver) -> SolveStatus` | Solves the damped system `A x = b` into `getx(ls)` (the step is `-x`; the iterators negate it). Returns `Success` when `x` solves it; `NotPositiveDefinite` when the factorization failed (`x` is then unspecified); `Indefinite` when it solved but found `A` indefinite (from `:ldl`, and CG's nonpositive curvature). The iterators treat anything but `Success` as failure: Newton shifts the diagonal, Levenberg-Marquardt raises `λ`, and dogleg shifts the diagonal as Newton does, taking the Cauchy step only if every shift fails. Leaves `A` and `b` exactly as written, damped or not, whether the solve succeeds or fails. `linsolver` may be a `Symbol` or a `StaticSymbol`. |
| `copy_structure(ls)` | A new system with the same structure and its own `A` and `b` (contents unspecified). It may share `x` and factorization workspaces with `ls`, provided each `solve!` is self-contained, so that solving one never changes the other's next solution. |
| `sharex(ls, from)` | `ls`, or a copy of it sharing its `A` and `b`, whose `getx` is `getx(from)`. |
| `SecondOrderOptimization.max_diag(ls)` | The largest absolute diagonal entry of `A`; `NaN` if a diagonal entry is `NaN` (as `max` propagates it). |
| `SecondOrderOptimization.dampingfloor(ls)` | The smallest damping the solver resolves, relative to the diagonal: `eps(Float64)` by default (Cholesky of `A + λD`), `eps(Float64)^2` for the tall system (QR of `J`). Levenberg-Marquardt floors its damping at `min(dampingfloor(ls) * max_diag(ls), 1e-12)` for `:levenbergmarquardt` (1e-12 if `max_diag` is not finite or is zero), and at `min(dampingfloor(ls), 1e-12)` for `:levenbergmarquardt_dtd`, whose `λ` multiplies the scaled diagonal: the floor can be lower than the former constant 1e-12, never higher. |
| `SecondOrderOptimization.getdiag!(d, ls)` | Writes the diagonal of the undamped `A` into `d` and returns it (for a tall system, the squared column norms of `J`). `:levenbergmarquardt_dtd` builds its scaling from it: the running maximum of its absolute values, with a relative floor. The generic method reads `A[i, i]`; a system whose `A` can't be indexed needs its own method (the sparse, Schur and tall systems have one). |
| `quadratic_model_change(ls, dx)` | `dx' A dx / 2 + b' dx`, undamped. The generic method uses `quadform(getA(ls), dx)` and `fast_dot(getb(ls), dx)`. |

Which system `makelinearsystem` builds:
- a size `n`: a static system for a `StaticInt`, a dense one for an `Int`. The optimizer asks for one
  when `H = nothing`: static when `nvars(x0)` is a static size (a `Number`, a `StaticVector`, a tuple
  of them) of at most [`MAX_STATIC_SYSTEM`](@ref) (16), otherwise dense of size `nvars(x0)`
  ([`usesstaticsystem`](@ref)`(x0)` says which).
- `H::MMatrix`: a static system that writes into `H`; any other `StaticMatrix`: a static system of
  its own.
- `H::Matrix{Float64}`: a dense system that writes into `H`; any other `AbstractMatrix`: a dense
  system on a `Float64` copy.
- `H::SparseMatrixCSC{Float64}`: a sparse system that writes into `H`, or into a copy with explicit
  zeros added if a diagonal entry is missing.
- `H::SchurBlockHessian` (BlockLinearSystems.jl): a Schur-complement system, dense or sparse
  according to the storage of `H.R`; `linsolver` (`:auto`, `:cholesky` or `:ldl`) names the solver
  for the reduced system.
"""
AbstractLinearSystem
