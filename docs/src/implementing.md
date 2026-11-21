# Implementing the contract

```@meta
DocTestSetup = :(using DampedLinearSolve, LinearAlgebra, SparseArrays, Static)
```

A package with its own storage (a block-structured Hessian, a reduced system) implements the
contract by defining a subtype of [`AbstractLinearSystem`](@ref) and methods of DampedLinearSolve's
functions for it. An optimizer written against the contract then runs on it unchanged.

## What to define

For a system type `MySystem`:

1. **Construction**: a method of [`makehessiansystem`](@ref) (or [`makejacobiansystem`](@ref)) for your
   prototype type, and, if the prototype has checks, of [`checkhessian`](@ref) (or
   [`checkjacobian`](@ref), [`jacobian_ncols`](@ref), [`jacobian_nrows`](@ref),
   [`check_jacobian_linsolver`](@ref)). Add [`structurekey`](@ref) for a prototype that is not a
   matrix, so that a caller caching systems can recognise its structure.
2. **Filling**: [`getA`](@ref), [`getb`](@ref), [`getx`](@ref) and [`zero!`](@ref); for Jacobian storage
   also [`getJ`](@ref), [`getr`](@ref), [`gradient!`](@ref) and [`markstale!`](@ref).
3. **Damping and solving**: [`dampen!`](@ref) with both signatures, and [`solve!`](@ref) for each
   solver you accept, returning a [`SolveStatus`](@ref), leaving `A` and `b` as filled.
4. **Queries**: [`max_diag`](@ref), [`getdiag!`](@ref), [`quadform`](@ref) for your storage (or a
   three-argument `LinearAlgebra.dot`), [`dampingfloor`](@ref), [`isfinitesystem`](@ref) and, for a
   system that can be held in `Float32`, [`systemeltype`](@ref) (a system without a method is `Float64`).
   [`quadratic_model_change`](@ref) has a generic method built on `quadform` and `getb`.
5. **The spare system**: [`copy_structure`](@ref) and [`sharex`](@ref).
6. **The conformance hooks**, for storage other than a `Matrix`, `MMatrix` or `SparseMatrixCSC`:
   [`conformance_matrix`](@ref), [`conformance_fill!`](@ref), and where needed
   [`conformance_fillb!`](@ref) and [`conformance_traits`](@ref).

A system that reduces itself to a standard matrix (eliminating some variables, say) can hold a
DampedLinearSolve system of that matrix's type, fill it, solve it through the contract, and
back-substitute, so that it inherits the standard solvers rather than writing its own.

## The conformance check

Run [`check_linear_system`](@ref) in your test suite for each solver you accept. It builds fresh
systems with the function you pass, fills them through the hooks, and checks every rule that can be
checked from outside; it throws naming the first rule broken.

```jldoctest
julia> check_linear_system(s -> makehessiansystem(zeros(4, 4), s); n = 4, linsolvers = (:auto, :cholesky, :ldl, :qr))

julia> check_linear_system(s -> makejacobiansystem(6, 4, s); n = 4, linsolvers = (:auto, :qr))
```
