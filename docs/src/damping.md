# Damping

```@meta
DocTestSetup = :(using DampedLinearSolve, LinearAlgebra, SparseArrays, Static)
```

A system solves `(A + λD) x = b` with the damping set by [`dampen!`](@ref):

- `dampen!(ls, λ)`: `D = I`, the damping of Levenberg's method;
- `dampen!(ls, λ, s)`: `D = Diagonal(s)`, with `s` in the units of the diagonal of `A` (not squared).
  With `s` built from the diagonal of `A` this is Marquardt's damping, which makes the step invariant
  to a rescaling of the variables. [`getdiag!`](@ref) writes the undamped diagonal (for a tall system
  the squared column norms of `J`) for building `s`.

The rules:

- **Recorded, not applied.** `dampen!` stores `λ` and `s`; only [`solve!`](@ref) applies them, to a
  scratch copy (dense and static systems), to the diagonal with the original restored afterwards
  (sparse systems), or as extra rows `√λ D^½` of the least-squares problem (tall systems). `A` and `b`
  are left exactly as filled, so the same fill can be solved at many dampings.
- **Replaced, not accumulated.** Each `dampen!` replaces the damping and scaling set before;
  returning to an earlier damping gives the identical `x`. `λ = 0` is undamped.
- **Never chosen.** The system does not pick or update `λ`. [`dampingfloor`](@ref) says how small a
  damping, relative to the diagonal, its solver still resolves: about `eps(Float64)` for a
  factorization of `A + λD`, and `eps(Float64)^2` for QR of `J`. [`max_diag`](@ref) gives the scale.

```jldoctest
julia> ls = makelinearsystem([2.0 0.0; 0.0 8.0], :auto);

julia> getb(ls) .= [1.0, 1.0];

julia> dampen!(ls, 0.5, [2.0, 8.0]); solve!(ls, :auto); getx(ls) ≈ [1/3, 1/12]
true

julia> dampen!(ls, 0.0); solve!(ls, :auto); getx(ls) ≈ [1/2, 1/8]
true

julia> getA(ls)
2×2 Matrix{Float64}:
 2.0  0.0
 0.0  8.0
```
