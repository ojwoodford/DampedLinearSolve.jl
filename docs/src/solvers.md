# Solvers

A solver is named by a `Symbol` or a `Static.StaticSymbol` when the system is built
([`makehessiansystem`](@ref), [`makejacobiansystem`](@ref)), which preallocates its workspace, and again
when it is solved ([`solve!`](@ref)); the two must agree. [`validatelinsolver`](@ref) says whether a
name is available, and [`checklinsolver`](@ref) throws, naming the package to load, when it is not.

| System (built from) | `:auto` | `:cholesky` | `:ldl` | `:qr` |
|---|---|---|---|---|
| static (`MMatrix`, `StaticMatrix`, a `StaticInt` size) | Cholesky | Cholesky | Cholesky, then LU (`Indefinite`) | LU |
| dense (`Matrix`, an `Int` size) | Cholesky | Cholesky | Bunch–Kaufman (LAPACK `sytrf`, preallocated) | column-pivoted QR of `A` |
| sparse (`SparseMatrixCSC`) | LDLᵀ (LDLFactorizations, analysed once) | CHOLMOD (analysed once) | LDLᵀ | sparse QR of `A` |
| Jacobian, dense (`Matrix`, an integer `m`) | QR of `[J r]`, then of `[R; √λ D^½]` | Cholesky of `JᵀJ`, formed once per linearization (BLAS `syrk`) | Bunch–Kaufman of `JᵀJ` | as `:auto` |
| Jacobian, static (a `StaticInt` `n`, `m·n ≤ 48`) | QR of `[J r]`, then of `[R; √λ D^½]` | | | as `:auto` |
| Jacobian, sparse (`SparseMatrixCSC`) | SPQR of `[J; √λ D^½]` (analysed once) | | | as `:auto` |

With Krylov.jl loaded, `:cg` and `:minres` solve dense and sparse systems (Jacobi preconditioned on a
sparse `A` whenever the damped diagonal is positive; CG reports nonpositive curvature as
`Indefinite`), and `:lsmr` solves a dense or sparse Jacobian system on `J` (not the static one). With
Pardiso.jl loaded and a Pardiso library available, `:pardiso` solves sparse systems (analysis once,
numeric factorization per solve) and dense ones through a sparse copy.

## How `:auto` chooses

`:auto` depends only on the storage: Cholesky for a static or dense `A`, where a dense factorization
of a positive-definite matrix is the cheapest; LDLᵀ for a sparse `A`, which also reports an
indefinite matrix; and QR of `J` for a Jacobian system, which works with the condition number of `J`
rather than of `JᵀJ`.

`:auto` never chooses the normal equations of a dense Jacobian system (`:cholesky`, `:ldl`): choose
them for a well-conditioned `J` with `m` well above `n`, where forming `JᵀJ` (about `mn²/2`
multiply-adds, once per linearization) and factoring `JᵀJ + λD` (`n³/3` per damping) costs less
than QR (about `2mn²`). Forming `JᵀJ` squares the condition number of `J`, so on an
ill-conditioned `J` they lose accuracy that QR keeps; that is why `:auto` does not. A system built
for them with a static `n` is the dense Jacobian system, never the static one.

## The static Jacobian system

A Jacobian system with a static number of variables `n` (a `StaticInt`), a dense Jacobian and a QR
solver (`:auto` or `:qr`) is static when `m·n ≤ 48`, a threshold measured against the dense Jacobian system on Julia 1.10
and 1.13 (a starting value, not a tuned one). The caller decides whether `n` is static.
