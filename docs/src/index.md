# DampedLinearSolve.jl

```@meta
DocTestSetup = :(using DampedLinearSolve, LinearAlgebra, SparseArrays, Static)
```

Solve, without allocating, the damped linear problems a second-order optimizer poses,
`(H + λD) x = −g` and `min ‖J x + r‖² + λ‖x‖²_D`, and report the outcome well enough for the
optimizer to decide what to do next. It applies the damping it is given and never chooses it.

The package holds:

- the [linear-system contract](contract.md): what an optimizer may call on a linear system, and what
  an implementation must do;
- its implementations for static (`SMatrix`), dense (`Matrix`) and sparse (`SparseMatrixCSC`)
  Hessians, and for dense, sparse and static Jacobians (the *tall* systems, which never form `JᵀJ`);
- the [solvers](solvers.md) and their selection (`:auto`, `:cholesky`, `:ldl`, `:qr`; `:cg`,
  `:minres` and `:lsmr` with Krylov.jl, `:pardiso` with Pardiso.jl);
- [`check_linear_system`](@ref), which checks an implementation against the contract.

Not in it: iterators, the choice or update of `λ`, trust radii, termination, residuals, variables,
automatic differentiation, block storage, reductions, unsymmetric systems, LU, GPU arrays.

```jldoctest
julia> ls = makelinearsystem([4.0 1.0; 1.0 3.0], :cholesky);

julia> getb(ls) .= [1.0, 2.0];

julia> dampen!(ls, 1.0);

julia> solve!(ls, :cholesky)
Success::SolveStatus = 0

julia> getx(ls) ≈ ([4.0 1.0; 1.0 3.0] + I) \ [1.0, 2.0]
true
```

SecondOrderOptimization.jl is built on it, and BlockLinearSystems.jl implements the contract for its
Schur and QR-eliminated systems.
