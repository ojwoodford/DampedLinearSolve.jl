# Solvers

A solver is named by a `Symbol` or a `Static.StaticSymbol` when the system is built
([`makelinearsystem`](@ref), [`maketallsystem`](@ref)), which preallocates its workspace, and again
when it is solved ([`solve!`](@ref)); the two must agree. [`validatelinsolver`](@ref) says whether a
name is available, and [`checklinsolver`](@ref) throws, naming the package to load, when it is not.

| System (built from) | `:auto` | `:cholesky` | `:ldl` | `:qr` |
|---|---|---|---|---|
| static (`MMatrix`, `StaticMatrix`, a `StaticInt` size) | Cholesky | Cholesky | Cholesky, then LU (`Indefinite`) | LU |
| dense (`Matrix`, an `Int` size) | Cholesky | Cholesky | Bunch–Kaufman (LAPACK `sytrf`, preallocated) | column-pivoted QR of `A` |
| sparse (`SparseMatrixCSC`) | LDLᵀ (LDLFactorizations, analysed once) | CHOLMOD (analysed once) | LDLᵀ | sparse QR of `A` |
| tall, dense or static (`Matrix`, an integer `m`) | QR of `[J r]`, then of `[R; √λ D^½]` | | | as `:auto` |
| tall, sparse (`SparseMatrixCSC{Float64, Int}`) | SPQR of `[J; √λ D^½]` (analysed once) | | | as `:auto` |

With Krylov.jl loaded, `:cg` and `:minres` solve dense and sparse systems (Jacobi preconditioned on a
sparse `A` whenever the damped diagonal is positive; CG reports nonpositive curvature as
`Indefinite`), and `:lsmr` solves a dense or sparse tall system on `J` (not the static one). With
Pardiso.jl loaded and a Pardiso library available, `:pardiso` solves sparse systems (analysis once,
numeric factorization per solve) and dense ones through a sparse copy.

## How `:auto` chooses

`:auto` depends only on the storage: Cholesky for a static or dense `A`, where a dense factorization
of a positive-definite matrix is the cheapest; LDLᵀ for a sparse `A`, which also reports an
indefinite matrix; and QR of `J` for a tall system, which works with the condition number of `J`
rather than of `JᵀJ`.

## The static tall system

A tall system with a static number of variables `n` (a `StaticInt`), a dense Jacobian and a QR
solver is static when `m·n ≤ 48`, a threshold measured against the dense tall system on Julia 1.10
and 1.13 (a starting value, not a tuned one). The caller decides whether `n` is static.
