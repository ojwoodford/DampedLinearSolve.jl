# The linear-system contract

A *linear system* here is the object a second-order optimizer hands its linearization to and gets a
step back from. This page is the contract between the two sides: what the optimizer may call, and
what an implementation must do. DampedLinearSolve's own systems meet it, and
[`check_linear_system`](@ref) checks that a system does. Another package can implement it for its own
storage (BlockLinearSystems.jl does, for its Schur and QR-eliminated systems: see
[Implementing the contract](implementing.md)).

## The problems

A system holds a symmetric matrix `A`, a vector `b` and a solution `x`, all of dimension `n`, and a
damping `λ ≥ 0` with an optional diagonal scaling `s`. A solve computes

```math
x = (A + λD)^{-1} b, \qquad D = I \text{ or } D = \operatorname{Diagonal}(s).
```

An optimizer fills `A` with the Hessian `H` (or its approximation) and `b` with the gradient `g`, so
that `−x` is the damped Newton step of `(H + λD) Δ = −g`.

A *tall* system holds a Jacobian `J` (`m × n`) and a residual `r` (length `m`) instead, and stands for
`A = JᵀJ` and `b = Jᵀr` without forming `JᵀJ`. Its solve is the solution of the damped least-squares
problem

```math
\min_x \; \lVert J x − r \rVert^2 + λ\, x^\mathsf{T} D\, x,
```

which is the same `x` in exact arithmetic, computed from `J` (QR, or LSMR) so that the condition
number is that of `J`, not of `JᵀJ`.

The system applies the damping it is given; it never chooses it.

## Construction

- `makehessiansystem(H, linsolver)`: a system built from a Hessian prototype `H`, whose `A` is
  stored as `H` describes;
  `makehessiansystem(n, linsolver)`: a system of `n` variables with no prototype, static (stack
  allocated, `StaticArrays`) when `n` is a `Static.StaticInt`, dense when it is an `Int`. The caller
  decides between the two by the type of `n`.
- `makejacobiansystem(J, n, linsolver)`: a tall system built from a Jacobian prototype `J` (a
  `Matrix{Float64}`, a `SparseMatrixCSC{Float64, Int}`, or an integer number of residuals `m`) and `n`
  variables. With a static `n`, a dense `J` and a QR solver, the system is static when `m·n` is small.
- `linsolver` (a `Symbol` or a `Static.StaticSymbol`) is used to preallocate the solver's workspace
  (factorization, pivots, Krylov vectors). A system is solved with the solver it was built for; asking
  it for another is an `ArgumentError`.
- Before building, the caller may check a prototype: `checkhessian(H, n)` and `checkjacobian(J)` throw
  for a prototype no system accepts, `jacobian_ncols(J)` gives a Jacobian prototype's number of
  columns (or `nothing`), and `check_jacobian_linsolver(J, linsolver)` throws for a solver the tall
  system of `J` does not take. `checklinsolver(linsolver)` throws for a solver name no loaded package
  provides, and `validatelinsolver(linsolver)` says whether one does.
- `structurekey(P)` is a value that is equal for two prototypes from which systems of the same
  structure are built, for a caller that caches systems.

Construction may allocate; nothing after it does (see [Allocation](#Allocation)).

## Filling

- `getA(ls)`, `getb(ls)` and `getx(ls)` return the storage of `A`, `b` and `x`. `b` and `x` are
  vectors of length `n`. `getA(ls)` is whatever the system stores `A` as: a `Matrix`, an `MMatrix`, a
  `SparseMatrixCSC`, or a storage type of the system's own.
- `zero!(ls)` prepares the system for a fresh fill, keeping its structure (pattern, workspaces). It
  zeroes `b` and any storage that is accumulated into (`A` of a Hessian system), and invalidates
  everything computed from the previous fill (a factorization, the column norms).
- A Hessian system is filled by writing `getA(ls)` and `getb(ls)` (typically adding into them after
  `zero!`).
- A tall system's storage `A = getA(ls)` holds `getJ(A)` and `getr(A)`, which are *set*, not
  accumulated: `zero!` does not zero them. After writing them, the caller calls
  `gradient!(getb(ls), A)`, which forms `b = Jᵀr` (and, for a sparse `J`, checks that its pattern is
  the one the system was built for), and `markstale!(A)`, which invalidates the factorization and the
  column norms. Code that writes `J` or `r` must call `markstale!` (or `zero!`) before the next solve.

## Damping

- `dampen!(ls, λ)` records the damping `λI`; `dampen!(ls, λ, s)` records `λ·Diagonal(s)`, with `s` in
  the units of the diagonal of `A` (for a tall system, of `JᵀJ`: the squared column norms), not
  squared. `dampen!(ls, λ, nothing)` is `dampen!(ls, λ)`. Each call replaces the damping set before,
  scaling included, so returning to an earlier damping gives the identical `x`; `λ = 0` is undamped.
- The damping is applied only inside `solve!`. `A` and `b` are left exactly as they were filled,
  whether the solve succeeds or fails, so a caller can solve the same fill at several dampings.

## Solving

`solve!(ls, linsolver)` solves the damped system into `getx(ls)` and returns a `SolveStatus`:

| Status | Meaning |
|---|---|
| `Success` | `x` solves the damped system, and is finite. |
| `NotPositiveDefinite` | The factorization failed or the solution is not finite: the damped matrix is not positive definite, singular to working precision, or not finite. `x` is unspecified (a tall system with a non-finite `J` or `r` sets it to `NaN`). |
| `Indefinite` | The solve completed, but found the damped matrix indefinite (`:ldl` from its pivots; `:cg` from nonpositive curvature). `x` solves the system (all of it, as for `Success`), but is not a descent direction to rely on. |

An optimizer treats anything but `Success` as a failed solve and raises the damping; a status of
`Indefinite` lets it tell negative curvature from a factorization that failed.

## Queries on the undamped system

These read the system as filled, whatever damping is recorded:

- `max_diag(ls)`: the largest absolute diagonal entry of `A` (`NaN` if a diagonal entry is `NaN`).
- `getdiag!(d, ls)`: writes the diagonal of `A` into `d` (for a tall system the squared column norms
  of `J`) and returns `d`.
- `quadform(getA(ls), v)`: `vᵀ A v` (for a tall system `‖J v‖²`).
- `quadratic_model_change(ls, dx)`: `dxᵀ A dx / 2 + bᵀ dx`, the change in the quadratic model for a
  step `dx`.
- `dampingfloor(ls)`: the smallest damping the solver resolves, relative to the diagonal: about
  `eps(Float64)` for a factorization of `A + λD`, and `eps(Float64)^2` for QR of `J`, which works with
  the condition number of `J`.
- `isfinitesystem(ls)`: whether every stored value of `A` (of `J` and `r`) is finite. Damping cannot
  repair a non-finite matrix, so an optimizer stops retrying when this is false.

## The spare system

An optimizer that evaluates a trial point keeps a second system and swaps the two when the trial is
accepted:

- `copy_structure(ls)` returns a system of the same structure with its own `A` and `b` (contents
  unspecified; a tall system's copy starts with the same `J`). It may share `x` and solver workspaces
  with `ls`, provided each solve is self-contained: filling and solving one never changes the other's
  next solution.
- `sharex(ls, from)` returns `ls`, or a copy of it sharing its `A` and `b`, whose `x` is `getx(from)`.

## Solvers

| System (from) | `:auto` | `:cholesky` | `:ldl` | `:qr` | `:cg`, `:minres` (Krylov.jl) | `:lsmr` (Krylov.jl) | `:pardiso` (Pardiso.jl) |
|---|---|---|---|---|---|---|---|
| static (`MMatrix`, `StaticMatrix`, static `n`) | Cholesky | Cholesky | Cholesky, then LU (`Indefinite`) | LU | | | |
| dense (`Matrix`, `Int` `n`) | Cholesky | Cholesky | Bunch–Kaufman | pivoted QR of `A` | yes | | yes (copies to sparse per solve) |
| sparse (`SparseMatrixCSC`) | LDLᵀ (LDLFactorizations) | CHOLMOD | LDLᵀ | sparse QR of `A` | yes, Jacobi preconditioned | | yes |
| tall, dense or static (`Matrix`, `Int` `m`) | QR of `J` | | | QR of `J` | | LSMR on `J` (not static) | |
| tall, sparse (`SparseMatrixCSC`) | SPQR of `J` | | | SPQR of `J` | | LSMR on `J` | |

`:auto` is the default. `validatelinsolver` knows the solver names of the packages loaded.

## Allocation

Nothing allocates after construction: filling, damping, solving, the queries and `sharex`. The
exceptions are inside libraries that offer no preallocated route: SuiteSparse's CHOLMOD, SPQR and
sparse QR allocate their working memory per factorization (on Julia 1.10, also the CHOLMOD solve);
`:pardiso` on a dense system builds a sparse copy per solve; a Krylov solve of a dense system built for
another solver allocates its workspace.

## Checking an implementation

`check_linear_system(make_ls; n, linsolvers)` builds systems with `make_ls`, fills them through the
conformance hooks (`conformance_fill!`, `conformance_fillb!`, `conformance_matrix`,
`conformance_traits`) and checks every rule on this page that can be checked from outside: solutions
against a dense reference, damping replaced and not accumulated, `A` and `b` untouched by a solve, the
spare system independent, the queries, `zero!`, and indefinite matrices. It throws naming the first
rule broken.
