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

A *Jacobian* system holds a Jacobian `J` (`m × n`) and a residual `r` (length `m`) instead, and stands for
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
- `makejacobiansystem(J, n, linsolver)`: a Jacobian system built from a Jacobian prototype `J` (a
  `Matrix`, a `SparseMatrixCSC{T, Int}`, or an integer number of residuals `m`) and `n`
  variables. With a static `n`, a dense `J` and a QR solver, the system is static when `m·n` is small.
- The prototype's element type is the system's (see [Precision](#Precision)).
- `linsolver` (a `Symbol` or a `Static.StaticSymbol`) is used to preallocate the solver's workspace
  (factorization, pivots, Krylov vectors). A system is solved with the solver it was built for; asking
  it for another is an `ArgumentError`.
- Before building, the caller may check a prototype: `checkhessian(H, n)` and `checkjacobian(J)` throw
  for a prototype no system accepts, `jacobian_ncols(J)` gives a Jacobian prototype's number of
  columns (or `nothing`), and `check_jacobian_linsolver(J, linsolver)` throws for a solver the Jacobian
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
- A Jacobian system's storage `A = getA(ls)` holds `getJ(A)` and `getr(A)`, which are *set*, not
  accumulated: `zero!` does not zero them. After writing them, the caller calls
  `gradient!(getb(ls), A)`, which forms `b = Jᵀr` (and, for a sparse `J`, checks that its pattern is
  the one the system was built for), and `markstale!(A)`, which invalidates the factorization and the
  column norms. Code that writes `J` or `r` must call `markstale!` (or `zero!`) before the next solve.

## Damping

- `dampen!(ls, λ)` records the damping `λI`; `dampen!(ls, λ, s)` records `λ·Diagonal(s)`, with `s` in
  the units of the diagonal of `A` (for a Jacobian system, of `JᵀJ`: the squared column norms), not
  squared. `dampen!(ls, λ, nothing)` is `dampen!(ls, λ)`. Each call replaces the damping set before,
  scaling included, so returning to an earlier damping gives the identical `x`; `λ = 0` is undamped.
- The damping is applied only inside `solve!`. `A` and `b` are left exactly as they were filled,
  whether the solve succeeds or fails, so a caller can solve the same fill at several dampings.

## Solving

`solve!(ls, linsolver)` solves the damped system into `getx(ls)` and returns a `SolveStatus`:

| Status | Meaning |
|---|---|
| `Success` | `x` solves the damped system, and is finite. |
| `NotPositiveDefinite` | The factorization failed or the solution is not finite: the damped matrix is not positive definite, singular to working precision, or not finite. `x` is unspecified (a Jacobian system with a non-finite `J` or `r` sets it to `NaN`). |
| `Indefinite` | The solve completed, but found the damped matrix indefinite (`:ldl` from its pivots; `:cg` from nonpositive curvature). `x` solves the system (all of it, as for `Success`), but is not a descent direction to rely on. |

An optimizer treats anything but `Success` as a failed solve and raises the damping; a status of
`Indefinite` lets it tell negative curvature from a factorization that failed.

## Queries on the undamped system

These read the system as filled, whatever damping is recorded:

- `max_diag(ls)`: the largest absolute diagonal entry of `A` (`NaN` if a diagonal entry is `NaN`).
- `getdiag!(d, ls)`: writes the diagonal of `A` into `d` (for a Jacobian system the squared column norms
  of `J`) and returns `d`.
- `quadform(getA(ls), v)`: `vᵀ A v` (for a Jacobian system `‖J v‖²`).
- `quadratic_model_change(ls, dx)`: `dxᵀ A dx / 2 + bᵀ dx`, the change in the quadratic model for a
  step `dx`.
- `dampingfloor(ls)`: the smallest damping the solver resolves, relative to the diagonal: about
  `eps(T)` for a factorization of `A + λD` (including the normal equations of a dense Jacobian
  system, `:cholesky` and `:ldl`, which factor `JᵀJ + λD`), and `eps(T)^2` for QR of `J` (and
  LSMR on `J`), which works with the condition number of `J`.
- `systemeltype(ls)`: the element type `T` of the system (see [Precision](#Precision)).
- `isfinitesystem(ls)`: whether every stored value of `A` (of `J` and `r`) is finite. Damping cannot
  repair a non-finite matrix, so an optimizer stops retrying when this is false.

Every query returns a `Float64`, whatever the system's element type, and every quantity it reduces over `A`
or `J` is accumulated in `Float64`.

## Precision

A system has an element type `T`, `Float64` or `Float32` ([`systemeltype`](@ref)): the type its matrix `A`
(or `J`) is stored in, with every copy, reduction and factorization of it. It is the element type of the
prototype; a system built without one (an integer `n` or `m`) is `Float64`, and any other element type is
an `ArgumentError` when the system is built.

| Quantity | Type |
|---|---|
| `A` of a Hessian system, `J` of a Jacobian system, and their copies, reductions and factorizations | `T` |
| `b` (the gradient), `x` (the solution), `r` (a Jacobian system's residual) | `Float64` |
| `λ`, a Marquardt scaling `s`, and every value a query returns | `Float64` |

A solve narrows what it needs of `b` (and `r`) into the factorization's scratch in `T`, solves there, and
widens the result into `x`. Whoever fills `A` or `J` writes `T` values.

Whether a solver offers `Float32` is a fact about its library on that Julia version. A solver that does not
offer the prototype's element type is an `ArgumentError` when the system is built, naming the solvers that
do: there is no silent substitution, `:auto` included. Checked by building and solving on Julia 1.10
(SparseArrays 1.10) and 1.13 (SparseArrays 1.13):

| System | Offers `Float32` | Does not |
|---|---|---|
| static | `:auto`, `:cholesky`, `:ldl`, `:qr` (StaticArrays) | |
| dense | `:auto`, `:cholesky`, `:ldl`, `:qr` (LAPACK); `:cg`, `:minres` (Krylov.jl) | `:pardiso` |
| sparse | `:auto`, `:ldl` (LDLFactorizations); `:cholesky` (CHOLMOD) from SparseArrays 1.13, which factorizes single precision; `:cg`, `:minres` | `:cholesky` on SparseArrays 1.10 (which converts to `Float64`); `:qr` (SPQR converts to `Float64` on both); `:pardiso` |
| Jacobian, dense | `:auto`, `:qr` (LAPACK QR), `:cholesky`, `:ldl` (BLAS `syrk`, LAPACK); `:lsmr` | |
| Jacobian, static | `:auto`, `:qr` | |
| Jacobian, sparse | `:lsmr` | `:auto`, `:qr` (SPQR) |

The Krylov solvers run in `T` (their vectors in `T`, `b` or `r` narrowed into them), with their tolerances
`√eps(T)` (LSMR's `conlim` `1/√eps(T)`). `:pardiso` is not offered for `Float32`: it could not be checked,
neither MKL nor Panua Pardiso being available on the machine this was built on.

The `Float32` QR of a dense Jacobian system loses the direction of a heavily damped step: the
Levenberg step's direction is within 1e-3 of the gradient's up to `λ = 1e7·max_diag` on the package's
test problem, and 0.065 from it at `1e12·max_diag` (`Float64`: 1e-13), a documented limit of that solver.

The tolerances that compare with rounding in the matrix follow `T`: [`dampingfloor`](@ref) is `eps(T)` or
`eps(T)^2`, the rank tests of the Jacobian QR use `eps(T)`, and the indefiniteness test of `:ldl` counts a
pivot as negative below `-rtol` times the largest, with `rtol = 1e-6` for `Float64` and
`eps(Float32)^(3/8) ≈ 2.5e-3` for `Float32` (a starting value).

## Conditioning

The dense and sparse systems, Hessian and Jacobian, condition the damped problem inside `solve!`, in either
element type: with `S = Diagonal(s)`, a Hessian system solves `S(A + λD)S y = S b` and returns `x = S y`; a
Jacobian system factorizes `J S`, with the damping rows `√(λ D_jj)·s_j`, and returns `x = S y`. In exact
arithmetic `x` is the unconditioned solution, so an optimizer follows the same path with or without it,
apart from rounding, and its damping, trust region and tests are all in the original variables.

`s_j` is the power of two nearest, on a log scale, to `1/(1e-12 + √|d_j + λ₀·D_jj|)`, with `d_j` the
diagonal of `A` as filled (for a Jacobian system, the squared column norms of `J`) and `λ₀·D` the damping of
the first solve after a fill; `s_j = 1` where that is not finite. It is computed once per fill, at that
first solve, and reused for later dampings of the same fill (a Jacobian system factorizes `J S` once per
fill, as without conditioning); [`zero!`](@ref) and [`markstale!`](@ref) invalidate it. Each scaled entry is
formed in `Float64` and narrowed once. When a later damping has moved far from the one the scale was computed
at (for some variable the scaled damped diagonal `|d_j + λ·D_jj|·s_j²` lies outside `[2^-16, 2^16]`), the scale
is recomputed at that damping, for that solve and the ones after it: going from undamped to heavily damped on a
weak variable is the case it is for, and an optimizer raising or lowering `λ` by a few factors of ten does not
reach it. The dense Jacobian system then rescales the columns of its held `R` (exact: the scales are powers of
two) rather than factorizing `J` again. The `1e-12` and the `2^16` are starting values.
Scaling by powers of two is exact, so with no overflow or underflow a factorization that does not pivot
(Cholesky, LDLᵀ, Householder QR) gives the scaled result of the unconditioned one exactly: what
conditioning changes is the range of the entries, which matters in `Float32`, and the decisions made by
comparing magnitudes (pivoting, and the rank and indefiniteness tests). `A`, `J`, `b` and `r` are never
scaled: the solve scales as it copies into its scratch, and every query reads the system as filled.

The static systems are not conditioned, and neither are the Krylov solves: for an iterative solver a
diagonal scaling is a preconditioner, which changes the iterates by more than rounding. There is no
public switch.

## The spare system

An optimizer that evaluates a trial point keeps a second system and swaps the two when the trial is
accepted:

- `copy_structure(ls)` returns a system of the same structure with its own `A` and `b` (contents
  unspecified; a Jacobian system's copy starts with the same `J`). It may share `x` and solver workspaces
  with `ls`, provided each solve is self-contained: filling and solving one never changes the other's
  next solution.
- `sharex(ls, from)` returns `ls`, or a copy of it sharing its `A` and `b`, whose `x` is `getx(from)`.

## Solvers

| System (from) | `:auto` | `:cholesky` | `:ldl` | `:qr` | `:cg`, `:minres` (Krylov.jl) | `:lsmr` (Krylov.jl) | `:pardiso` (Pardiso.jl) |
|---|---|---|---|---|---|---|---|
| static (`MMatrix`, `StaticMatrix`, static `n`) | Cholesky | Cholesky | Cholesky, then LU (`Indefinite`) | LU | | | |
| dense (`Matrix`, `Int` `n`) | Cholesky | Cholesky | Bunch–Kaufman | pivoted QR of `A` | yes | | yes (copies to sparse per solve) |
| sparse (`SparseMatrixCSC`) | LDLᵀ (LDLFactorizations) | CHOLMOD | LDLᵀ | sparse QR of `A` | yes, Jacobi preconditioned | | yes |
| Jacobian, dense (`Matrix`, `Int` `m`) | QR of `J` | Cholesky of `JᵀJ`, formed once per linearization (BLAS `syrk`) | Bunch–Kaufman of `JᵀJ` | QR of `J` | | LSMR on `J` | |
| Jacobian, static (static `n`, `m·n ≤ 48`, a QR solver) | QR of `J` | | | QR of `J` | | | |
| Jacobian, sparse (`SparseMatrixCSC`) | SPQR of `J` | | | SPQR of `J` | | LSMR on `J` | |

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
rule broken. A system of element type `T` is held to the same rules, with the tolerances on its solutions in
proportion to `eps(T)`: those stated, for `Float64`.
