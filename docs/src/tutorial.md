# Tutorial: solving damped systems

```@meta
DocTestSetup = :(using DampedLinearSolve, LinearAlgebra, SparseArrays, Static)
```

## A dense Hessian

```jldoctest dense
julia> H = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]; g = [1.0, 0.0, -1.0];

julia> ls = makelinearsystem(copy(H), :ldl);    # Bunch–Kaufman workspace preallocated

julia> getA(ls) .= H; getb(ls) .= g;

julia> dampen!(ls, 0.1); solve!(ls, :ldl)
Success::SolveStatus = 0

julia> getx(ls) ≈ (H + 0.1I) \ g                # the damped Newton step is -getx(ls)
true

julia> quadratic_model_change(ls, -getx(ls)) < 0  # a descent step
true
```

An indefinite matrix is reported, not hidden:

```jldoctest dense
julia> getA(ls)[1, 1] = -4.0; dampen!(ls, 0.0); solve!(ls, :ldl)
Indefinite::SolveStatus = 2

julia> solve!(makelinearsystem(copy(getA(ls)), :cholesky), :cholesky)
NotPositiveDefinite::SolveStatus = 1
```

## A sparse Hessian

The sparse system writes into the prototype's storage; its symbolic analysis is done once, when the
system is built.

```jldoctest sparse
julia> H = spdiagm(-1 => fill(-1.0, 4), 0 => fill(2.5, 5), 1 => fill(-1.0, 4));

julia> ls = makelinearsystem(copy(H), :auto); getb(ls) .= ones(5);

julia> dampen!(ls, 1e-3); solve!(ls, :auto)
Success::SolveStatus = 0

julia> getx(ls) ≈ (Matrix(H) + 1e-3I) \ ones(5)
true

julia> max_diag(ls)
2.5
```

## A tall system

A tall system holds `J` and `r`; the caller writes them, forms `b = Jᵀr` with [`gradient!`](@ref),
and marks the old factorization stale with [`markstale!`](@ref).

```jldoctest tall
julia> J = [1.0 0.0; 1.0 1.0; 1.0 2.0; 1.0 3.0]; r = [1.0, 2.0, 2.0, 4.0];

julia> ls = maketallsystem(4, 2, :auto); A = getA(ls);

julia> getJ(A) .= J; getr(A) .= r; gradient!(getb(ls), A); markstale!(A);

julia> getb(ls) ≈ J' * r
true

julia> dampen!(ls, 0.0); solve!(ls, :auto)
Success::SolveStatus = 0

julia> getx(ls) ≈ J \ r                          # the least-squares solution
true

julia> dampen!(ls, 0.5); solve!(ls, :auto); getx(ls) ≈ (J' * J + 0.5I) \ (J' * r)
true
```

The factorization of `J` is kept between the two solves: a new damping refactors only an `n × n`
block. A spare system for a trial point comes from [`copy_structure`](@ref).
