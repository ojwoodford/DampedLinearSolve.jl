# DampedLinearSolve.jl

[![docs-stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ojwoodford.github.io/DampedLinearSolve.jl/stable/)
[![docs-dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ojwoodford.github.io/DampedLinearSolve.jl/dev/)
[![CI](https://github.com/ojwoodford/DampedLinearSolve.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ojwoodford/DampedLinearSolve.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/ojwoodford/DampedLinearSolve.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ojwoodford/DampedLinearSolve.jl)

Solve, without allocating, the damped linear problems a second-order optimizer poses,
`(H + λD) x = −g` and `min ‖J x + r‖² + λ‖x‖²_D`, and report the outcome well enough for the
optimizer to decide what to do next. It applies the damping it is given and never chooses it.

## Example

```julia
using DampedLinearSolve

H = [4.0 1.0; 1.0 3.0]
ls = makehessiansystem(H, :cholesky)    # a dense system that writes into H
getb(ls) .= [1.0, 2.0]                 # the gradient g
dampen!(ls, 0.5)                       # λ = 0.5, D = I; H itself is not changed
solve!(ls, :cholesky)                  # Success: getx(ls) == (H + 0.5I) \ g, the step is -getx(ls)

jac = makejacobiansystem(20, 2, :auto)     # 20 residuals, 2 variables: QR of J, A = JᵀJ never formed
```

The contract every system meets (construction, filling, damping, the solve and its statuses, the
queries, the spare system) is in the documentation, with the solvers each system takes and a guide
to implementing it for your own storage. BlockLinearSystems.jl implements it for its Schur and
QR-eliminated systems; SecondOrderOptimization.jl is built on it.

## Why not LinearSolve.jl?

[LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) is a unified interface to Julia's linear
solvers for `A u = b`. A second-order optimizer asks a linear solve for more than that interface
provides; this package provides exactly that, and little else. Compared with LinearSolve.jl as
documented for its version 5.18.3:

- **Damping inside the solve.** An optimizer solves the same matrix at several dampings `λ`, with
  `D = I` or a diagonal scaling, and must keep the matrix it filled. Here `dampen!(ls, λ)` or
  `dampen!(ls, λ, s)` records the damping and `solve!` applies it to a copy or restores it, so `A` is
  left as filled. LinearSolve.jl documents no damping or regularization parameter; its shifted-system
  support is for the operator `J − M/γ` with a dense `J` (`update_gamma!`), and otherwise a new matrix
  is assigned to the cache, which refactorizes it.
- **A status that tells indefinite from failed.** `solve!` returns `Success`, `NotPositiveDefinite`
  or `Indefinite` (from the pivots of an LDLᵀ or Bunch–Kaufman factorization, or from CG's
  nonpositive curvature), which decides between raising the damping and taking a different step.
  LinearSolve.jl's return codes include `Success` and `Failure`, and document no indefiniteness or
  inertia report.
- **Queries on the undamped system.** The largest diagonal entry, the diagonal (the squared column
  norms of `J` on the Jacobian path), the quadratic form `vᵀ A v`, the change in the quadratic model, and
  the smallest damping the solver resolves: what an optimizer needs to set and update `λ` and to
  judge a step. LinearSolve.jl documents no such queries.
- **Hessian and Jacobian systems behind one contract.** A Hessian system and a Jacobian system
  (solved by QR or LSMR on `J`, which never form `JᵀJ`, or, with `linsolver = :cholesky` or `:ldl`
  on a dense `J`, by the normal equations, formed once per linearization) answer the same calls, so an
  optimizer is written once. LinearSolve.jl solves rectangular systems in the least-squares sense,
  by QR, SVD or LSMR, as a different problem from the square one.
- **A spare system for trial points.** `copy_structure` gives a system of the same structure with its
  own storage, sharing workspaces where that is safe, and `sharex` lets two systems share a solution
  vector, so an optimizer can linearize at a trial point and swap.
- **No allocation per solve.** Workspaces (pivots, LAPACK work arrays, symbolic analyses, Krylov
  vectors) are built once, with the system; the exceptions are inside SuiteSparse. LinearSolve.jl
  caches factorizations and documents allocation-free repeated solves for some of its algorithms.
- **A small dependency footprint.** Five hard dependencies (LinearAlgebra, SparseArrays,
  StaticArrays, Static, LDLFactorizations), with Krylov.jl and Pardiso.jl as weak dependencies.
  LinearSolve.jl 5.18.3 has 28 hard and 48 weak dependencies.

What this package does not do, and LinearSolve.jl does: unsymmetric systems (LU, GMRES, BiCGStab and
others), the wider catalogue of dense and sparse direct solvers (UMFPACK, KLU, MUMPS, HSL and more)
and iterative ones, GPU arrays (CUDA, AMDGPU and Metal), and a general `A u = b` interface for any
caller. The two are complementary: this package is the narrow layer an optimizer's step needs.
