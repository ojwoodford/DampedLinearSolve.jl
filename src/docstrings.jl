"""
    AbstractLinearSystem

Supertype of the linear systems: a symmetric matrix `A`, a vector `b` and a solution `x` of
dimension `n`, and a damping to apply when solving. The built-in systems are built by
[`makehessiansystem`](@ref) (static, dense and sparse `A`) and [`makejacobiansystem`](@ref) (`A = JᵀJ`
held as a Jacobian `J` and a residual `r`, dense, sparse or static). A new system type implements
the functions of the contract below, as methods of `DampedLinearSolve`'s functions, and should
pass [`check_linear_system`](@ref). The documentation's "The linear-system contract" page states
each rule.

| Function | Contract |
|:---|:---|
| [`getA`](@ref), [`getb`](@ref), [`getx`](@ref) | The storage of `A`, `b` (length `n`) and `x` (length `n`). |
| [`zero!`](@ref) | Prepares a fresh fill: zeroes `b` and accumulated storage, invalidates what was computed from the old fill. |
| [`getJ`](@ref), [`getr`](@ref), [`gradient!`](@ref), [`markstale!`](@ref) | A Jacobian system's storage: `J` and `r`, `b = Jᵀr`, and "`J` or `r` was written". |
| [`dampen!`](@ref) | Records `λI` or `λ·Diagonal(s)`, applied only inside `solve!`. |
| [`solve!`](@ref) | Solves the damped system into `x`, leaving `A` and `b` as filled; returns a [`SolveStatus`](@ref). |
| [`max_diag`](@ref), [`getdiag!`](@ref), [`quadform`](@ref), [`quadratic_model_change`](@ref), [`dampingfloor`](@ref), [`isfinitesystem`](@ref), [`systemeltype`](@ref) | Queries on the undamped system, and its element type. |
| [`copy_structure`](@ref), [`sharex`](@ref) | A spare system of the same structure, and sharing `x` with another. |
"""
AbstractLinearSystem

"""
    SolveStatus

Enum returned by [`solve!`](@ref):
- [`Success`](@ref): `x` solves the damped system, and is finite.
- [`NotPositiveDefinite`](@ref): the factorization failed, or the solution is not finite: the damped
  matrix is not positive definite, singular to working precision, or not finite. `x` is unspecified.
- [`Indefinite`](@ref): the solve completed but found the damped matrix indefinite (`:ldl` from its
  pivots, `:cg` from nonpositive curvature). `x` solves the damped system, as for `Success`, but is
  not a descent direction to rely on.
"""
SolveStatus

"""
    Success

[`SolveStatus`](@ref) indicating that the linear system was solved successfully.
"""
Success

"""
    NotPositiveDefinite

[`SolveStatus`](@ref) indicating that the factorization failed (for example a non-positive pivot in
Cholesky) or that the solution is not finite.
"""
NotPositiveDefinite

"""
    Indefinite

[`SolveStatus`](@ref) indicating that the matrix was found indefinite, or negative curvature was
encountered, by a solve that completed: `x` solves the damped system, but is not a descent direction
to rely on. An implementation of the contract that returns `Indefinite` must leave the whole of `x`
solved, not part of it.
"""
Indefinite

"""
    makehessiansystem(H, linsolver) -> AbstractLinearSystem
    makehessiansystem(n, linsolver) -> AbstractLinearSystem

A linear system built from a Hessian prototype `H`, whose `A` is stored as `H` describes, or,
without a prototype, a system of `n` variables: static (`StaticArrays`, stack allocated) when `n`
is a `Static.StaticInt`, dense when it is an `Int`.

- `H::MMatrix`: a static system that writes into `H`; any other `StaticMatrix`: a static system of
  its own.
- `H::Matrix`: a dense system that writes into `H`; any other `AbstractMatrix`: a dense system on a
  copy of the same element type.
- `H::SparseMatrixCSC`: a sparse system that writes into `H`, or into a copy with explicit
  zeros added if a diagonal entry is missing.

The element type of `H`, `Float64` or `Float32`, is the system's ([`systemeltype`](@ref)): `A` and its
factorization are held in it, while `b`, `x`, the damping and the queries are `Float64`. Any other element
type is an `ArgumentError`, as is a `linsolver` that does not offer `Float32` for that kind of system
(see the contract's "Precision"). Without a prototype the system is `Float64`. The dense and sparse
systems are conditioned inside the solve (the contract's "Conditioning"); the static system is not.

A package that implements the contract adds methods for its own prototypes. `linsolver` (a
`Symbol` or a `StaticSymbol`) is used to preallocate the solver's workspace, and the system must be
solved with it. Validate `H` first with [`checkhessian`](@ref).

# Example
```julia
ls = makehessiansystem(zeros(3, 3), :ldl)      # dense, Bunch–Kaufman workspace preallocated
ls = makehessiansystem(static(2), :auto)        # static 2 × 2
```
"""
makehessiansystem

"""
    makejacobiansystem(J, n, linsolver) -> AbstractLinearSystem

A Jacobian linear system built from a Jacobian prototype `J`, for `n` variables: `A = JᵀJ` held as `J`
and the residual `r` ([`getJ`](@ref), [`getr`](@ref)), `b = Jᵀr`, and each solve the solution of
`min ‖J x − r‖² + λ xᵀD x` from `J`, without forming `JᵀJ` unless `linsolver` is `:cholesky` or `:ldl`.

- `J::Matrix`: a dense system that writes into `J` (with the values it holds); static
  (`StaticArrays`) when `n` is a `Static.StaticInt`, `linsolver` is QR (`:auto` or `:qr`) and
  `m·n` is small.
- `J::Integer`: the same, for `m = J` residuals, with its own `J` starting as zeros.
- `J::SparseMatrixCSC{T, Int}`: a sparse system, solved by SuiteSparse's SPQR, its symbolic
  analysis done here, once.

The element type of a matrix `J`, `Float64` or `Float32`, is the system's ([`systemeltype`](@ref)): `J`,
its reductions and factorizations are held in it, while `r`, `b`, `x` and the queries are `Float64`
(an integer `m` gives a `Float64` system). Any other element type is an `ArgumentError`, as is a
`linsolver` that does not offer `Float32` for the system: SPQR, for one (see the contract's
"Precision"). The dense and sparse systems are conditioned inside the solve; the static one is not.

`linsolver` is `:auto` or `:qr` (QR of `J`), or `:lsmr` with Krylov.jl loaded. For a dense `J` (a
`Matrix` or an integer `m`) it can also be `:cholesky` or `:ldl`: the normal equations,
`JᵀJ` formed once per linearization (BLAS `syrk`) and `JᵀJ + λD` factored per damping by Cholesky or
Bunch–Kaufman; always the dense system, never the static one, whatever `n`. They square the condition
number of `J`, so `:auto` is QR (see `DenseJacobianLinearSystem` for costs). Validate `J` first with
[`checkjacobian`](@ref) and [`check_jacobian_linsolver`](@ref).

# Example
```julia
ls = makejacobiansystem(20, 3, :auto)     # 20 residuals, 3 variables, dense
```
"""
makejacobiansystem

"""
    structurekey(P)

A value equal for two prototypes from which systems of the same structure are built (for a matrix,
its type and size), for a caller that caches systems. A package with its own prototype type adds a
method.
"""
structurekey

"""
    checkhessian(H, n::Integer) -> nothing

Throws, before anything is built, if `H` cannot be the Hessian prototype of a system of `n`
variables: a `DimensionMismatch` for a matrix that is not `n × n`. `H = nothing` (no prototype)
passes.
"""
checkhessian

"""
    checkjacobian(J) -> nothing

Throws an `ArgumentError` if `J` is not a Jacobian prototype a Jacobian system can be built from: a
`Matrix` or `SparseMatrixCSC{T, Int}` with `T` = `Float64` or `Float32` and at least one row, or an integer
number of residuals `m ≥ 1` (or a prototype a package that implements the contract adds). Whether the
solver offers `Float32` is checked when the system is built.
"""
checkjacobian

"""
    check_jacobian_linsolver(J, linsolver::Symbol) -> nothing

Throws an `ArgumentError` if the Jacobian system of the Jacobian prototype `J` does not take
`linsolver`. The built-in prototypes take `:auto`, `:qr` and `:lsmr` (the last with Krylov.jl
loaded, which [`checklinsolver`](@ref) checks); a dense one (a `Matrix` or an integer `m`)
also takes `:cholesky` and `:ldl` (the normal equations), and a `SparseMatrixCSC` one does not.
"""
check_jacobian_linsolver

"""
    jacobian_ncols(J) -> Union{Int, Nothing}

The number of columns of the Jacobian prototype `J` (the number of variables of its system), or
`nothing` for an integer `m`, whose system is sized by the `n` passed to [`makejacobiansystem`](@ref).
"""
jacobian_ncols

"""
    jacobian_nrows(J) -> Int

The number of residuals the Jacobian prototype `J` stands for: `m` for an integer, the number of
rows of a matrix.
"""
jacobian_nrows

"""
    checklinsolver(linsolver) -> true

Returns `true` if a solver of that name is available, and otherwise throws an `ArgumentError` that
says which package provides it (`:cg`, `:minres` and `:lsmr`: Krylov.jl; `:pardiso`: Pardiso.jl), or
that the name is unknown. `linsolver` is a `Symbol` or a `StaticSymbol`.
"""
checklinsolver

"""
    validatelinsolver(linsolver::Union{Symbol, StaticSymbol}) -> Bool

Returns `true` if `linsolver` names a solver of a loaded package, `false` otherwise. The built-in
solvers are `:auto`, `:cholesky`, `:ldl` and `:qr`; loading Krylov.jl adds `:cg`, `:minres` and
`:lsmr`, and loading Pardiso.jl adds `:pardiso`.
"""
validatelinsolver

"""
    getA(ls)

The storage of the system's matrix `A`: a `Matrix`, an `MMatrix`, a `SparseMatrixCSC`, or a storage
type of the system's own (for a Jacobian system, one holding `J` and `r`: see [`getJ`](@ref)).
"""
getA

"""
    getb(ls) -> AbstractVector

The system's right-hand side `b`, of length `n`. For a Jacobian system it is `Jᵀr`, formed by
[`gradient!`](@ref).
"""
getb

"""
    getx(ls) -> AbstractVector

The vector [`solve!`](@ref) writes the solution into, of length `n`.
"""
getx

"""
    zero!(ls)

Prepares the system for a fresh fill, keeping its structure (pattern, workspaces): zeroes `b` and
any storage that is accumulated into (`A` of a Hessian system), and invalidates everything computed
from the previous fill. A Jacobian system's `J` and `r` are set by the fill, not accumulated, so they are
not zeroed; their factorization and column norms are invalidated.

# Example
```julia
ls = makehessiansystem(3, :auto)
zero!(ls)
```
"""
zero!

"""
    getJ(A)

The Jacobian `J` in the storage `A = getA(ls)` of a Jacobian system, which the caller writes (it is set,
not accumulated). After writing it and [`getr`](@ref)`(A)`, call [`gradient!`](@ref) and
[`markstale!`](@ref).
"""
getJ

"""
    getr(A)

The residual `r` in the storage `A = getA(ls)` of a Jacobian system, which the caller writes.
"""
getr

"""
    gradient!(b, A) -> b

Writes `b = Jᵀr` from the Jacobian storage `A`'s current `J` and `r` (call it with `b = getb(ls)`). For a
sparse `J` it first checks that the pattern is the one the system was built for, and throws an
`ArgumentError` if an entry was added.
"""
gradient!

"""
    markstale!(A) -> A

Marks the Jacobian storage `A`'s factorization and column norms as out of date. Call it after writing
`J` or `r` and before the next solve or query.
"""
markstale!

"""
    dampen!(ls, λ)
    dampen!(ls, λ, s)

Records the damping `λI`, or `λ·Diagonal(s)` with `s` in the units of the diagonal of `A` (for a Jacobian
system, of `JᵀJ`), not squared. The damping is applied only inside [`solve!`](@ref); `A` is left as
it was filled. Each call replaces the damping set before, scaling included; `dampen!(ls, λ, nothing)`
is `dampen!(ls, λ)`, and `λ = 0` is undamped.
"""
dampen!

"""
    solve!(ls, linsolver = static(:auto)) -> SolveStatus

Solves the damped system `(A + λD) x = b` into [`getx`](@ref)`(ls)` with the solver `linsolver` (a
`Symbol` or a `StaticSymbol`, the one the system was built for) and returns a [`SolveStatus`](@ref).
`A` and `b` are left exactly as they were filled, whether the solve succeeds or fails.

# Solvers
- `:auto` (Cholesky for a static or dense `A`, LDLᵀ for a sparse one, QR of `J` for a Jacobian system),
  `:cholesky`, `:ldl` (Bunch–Kaufman for a dense `A`, LDLᵀ for a sparse one) and `:qr`.
- With Krylov.jl loaded: `:cg` and `:minres` (dense and sparse `A`; Jacobi preconditioned on a sparse
  `A` whenever the damped diagonal is positive) and `:lsmr` (a Jacobian system, on `J`).
- With Pardiso.jl loaded: `:pardiso` (sparse, and dense through a sparse copy).

On a static system, `:qr` is an LU solve (StaticArrays has no pivoted QR), and `:ldl` uses Cholesky,
falling back to LU when the matrix is not positive definite (the solve is then reported as
`Indefinite`).

# Example
```julia
ls = makehessiansystem([2.0 0.0; 0.0 4.0], :auto)
getb(ls) .= [4.0, 8.0]
solve!(ls, :cholesky)   # Success; getx(ls) == [2.0, 2.0]
```
"""
solve!

"""
    qrfactors(ls) -> (Q, R)

The factors of the last QR solve of a dense Jacobian system `ls` (built from a `Matrix` or an integer
prototype for `:auto` or `:qr`, with `m ≥ n`): `M = Q·[R; 0]`, where `M` is the matrix the system stands for, as
filled, at the damping of that solve: `J` (`m × n`) undamped, `[J; √(λD)]` (`(m + n) × n`) damped. `Q` is
orthogonal (`p × p`, `p` the rows of `M`) and `R` is upper triangular (`n × n`), both in the system's own
variables (the conditioning inside the solve does not show). They support only, on `Float64` vectors in place,
in `Float64` arithmetic with the factors (held in the system's element type) widened:

- `lmul!(Q', v)` and `lmul!(Q, v)`, `v` of length `p`;
- `ldiv!(R, v)` and `ldiv!(R', v)`, `v` of length `n`;
- `size(Q)` and `size(R)`.

So the solve's `x` is `ldiv!(R, (Q'·[r; 0])[1:n])` before any refinement. They are valid after a call of
[`solve!`](@ref) that returned `Success`, until the next [`zero!`](@ref), [`markstale!`](@ref) or solve of `ls` or
of a system sharing its workspace ([`copy_structure`](@ref), [`sharex`](@ref)); a later [`dampen!`](@ref) does
not change them. With none held, `qrfactors` throws, naming the reason. Any other system, a dense Jacobian
system built for another solver, or one with `m < n`, is an `ArgumentError`. Nothing allocates. This is not part
of the linear-system contract: no other system offers it.

# Example
```julia
using LinearAlgebra
ls = makejacobiansystem([1.0 0.0; 0.0 2.0; 1.0 1.0], 2, :qr)
getr(getA(ls)) .= [1.0, 2.0, 3.0]; gradient!(getb(ls), getA(ls)); markstale!(getA(ls))
solve!(ls, :qr)                       # Success
Q, R = qrfactors(ls)
d = lmul!(Q', copy(getr(getA(ls))))   # Qᵀr
ldiv!(R, d[1:2]) ≈ getx(ls)           # true
```
"""
qrfactors

"""
    max_diag(ls)

The largest absolute diagonal entry of the undamped `A` (for a Jacobian system, the largest squared
column norm of `J`); `NaN` if a diagonal entry is `NaN`.
"""
max_diag

"""
    getdiag!(d, ls) -> d

Writes the diagonal of the undamped `A` into `d` (for a Jacobian system, the squared column norms of
`J`) and returns `d`. For building a Marquardt scaling.
"""
getdiag!

"""
    quadform(A, v)

The quadratic form `vᵀ A v` of the storage `A = getA(ls)` (for a Jacobian system, `‖J v‖²`), with
single-threaded loops for `Matrix`, `StaticArray` and `SparseMatrixCSC`; any other storage falls back
to `LinearAlgebra.dot(v, A, v)`.
"""
quadform

"""
    quadratic_model_change(ls, dx)

`dxᵀ A dx / 2 + bᵀ dx` for the undamped system: the change in the quadratic model for a step `dx`
when `b` is the gradient (negative for a descent step).
"""
quadratic_model_change

"""
    dampingfloor(ls) -> Float64

The smallest damping the system's solver resolves, relative to the diagonal: `eps(T)` for a
factorization of `A + λD`, including a dense Jacobian system built for `:cholesky` or `:ldl` (which
factors `JᵀJ + λD`), and `eps(T)^2` for a Jacobian system solved by QR of `J` (or LSMR), which
works with the condition number of `J`. `T` is the system's element type ([`systemeltype`](@ref)); the
value is a `Float64` either way.
"""
dampingfloor

"""
    systemeltype(ls) -> Type

The element type of the linear system: the type its matrix (`A`, or `J` of a Jacobian system) and every
copy, reduction and factorization of it are held in, `Float64` or `Float32`. It is the element type of the
prototype the system was built from (`Float64` for a system built from a number of variables or residuals).
`b`, `x`, `r`, the damping and every query's result are `Float64` whatever it is. An optimizer reads it to
set the tolerances that depend on the precision of the matrix, such as its damping floor. A system of
another package that does not implement it is taken to be `Float64`.
"""
systemeltype

"""
    isfinitesystem(ls) -> Bool

Whether every stored value of `A` (for a Jacobian system, of `J` and `r`) is finite. Damping cannot repair
a non-finite matrix.
"""
isfinitesystem

"""
    copy_structure(ls)

A system of the same structure with its own `A` and `b` (contents unspecified; a Jacobian system's copy
starts with the same `J`), for a trial point. It may share `x` and solver workspaces with `ls`;
filling and solving one never changes the other's next solution.
"""
copy_structure

"""
    sharex(ls, from)

`ls`, or a copy of it sharing its `A` and `b`, whose [`getx`](@ref) is `getx(from)`.
"""
sharex
