using DampedLinearSolve, StaticArrays, SparseArrays, LinearAlgebra, Static, Test
using DampedLinearSolve: AbstractLinearSystem, check_linear_system
const DLS = DampedLinearSolve

# A3: the linear-system interface, checked on every core system, and the checker itself checked
# on deliberately broken systems

# A system that forwards to another (a DenseLinearSystem, unless built from one), with one behaviour broken on request
mutable struct BrokenSystem{S} <: AbstractLinearSystem
    inner::S
    broken::Symbol
    calls::Dict{Symbol, Int}            # dampen! calls of each kind, for the :drift_* faults
    cache::Base.RefValue{Any}           # the last solution, for :copy_shares_cache
    skip::Bool                          # the next solve! does nothing, for :zero_skips_solve
end
BrokenSystem(inner, broken, cache=Ref{Any}(nothing)) = BrokenSystem(inner, broken, Dict{Symbol, Int}(), cache, false)
BrokenSystem(n::Integer, broken) = BrokenSystem(DLS.DenseLinearSystem{Float64}(zeros(n, n), zeros(n), zeros(n)), broken)
DLS.getA(w::BrokenSystem) = DLS.getA(w.inner)
DLS.getb(w::BrokenSystem) = DLS.getb(w.inner)
DLS.getx(w::BrokenSystem) = DLS.getx(w.inner)
DLS.systemeltype(w::BrokenSystem) = DLS.systemeltype(w.inner)
DLS.conformance_fillb!(w::BrokenSystem, b::AbstractVector) = (DLS.conformance_fillb!(w.inner, b); w)
function DLS.solve!(w::BrokenSystem, s::StaticSymbol)
    w.skip && (w.skip = false; return Success)   # as if a factorization from before zero! were reused
    w.broken === :copy_shares_cache && !isnothing(w.cache[]) && (DLS.getx(w) .= w.cache[]; return Success)
    status = DLS.solve!(w.inner, s)
    w.broken === :solve_mutates_A && (w.inner.A[1, 1] += w.inner.lambda)   # leaves damping behind
    w.broken === :copy_shares_cache && (w.cache[] = copy(DLS.getx(w)))
    w.broken === :singular_success && return Success
    return status
end
# :drift_<kind>: from the second dampen! call of that kind on, the damping is slightly off
function drift(w::BrokenSystem, kind::Symbol, λ)
    w.calls[kind] = get(w.calls, kind, 0) + 1
    return w.broken === Symbol(:drift_, kind) && w.calls[kind] > 1 ? λ + 1e-8 : λ
end
function DLS.dampen!(w::BrokenSystem, λ::Real)
    w.cache[] = nothing
    λ = drift(w, iszero(λ) ? :undamped : :uniform, λ)
    w.broken === :dampen_keeps_scaling ? (w.inner.lambda = λ) : DLS.dampen!(w.inner, λ)
    return w
end
function DLS.dampen!(w::BrokenSystem, λ::Real, s)
    w.cache[] = nothing
    λ = drift(w, :scaled, λ)
    w.broken === :dampen_ignores_scaling ? DLS.dampen!(w.inner, λ) : DLS.dampen!(w.inner, λ, s)
    return w
end
function DLS.copy_structure(w::BrokenSystem)
    inner = DLS.copy_structure(w.inner)
    if w.broken === :copy_aliases_A
        inner = DLS.DenseLinearSystem{Float64}(w.inner.A, inner.b, inner.x, inner.A_scratch, inner.lambda, inner.scaling)
    end
    return BrokenSystem(inner, w.broken, w.broken === :copy_shares_cache ? w.cache : Ref{Any}(nothing))
end
DLS.sharex(w::BrokenSystem, from) = w.broken === :sharex_copies ? BrokenSystem(DLS.copy_structure(w.inner), w.broken) :
                                                                   BrokenSystem(DLS.sharex(w.inner, from.inner), w.broken)
function DLS.zero!(w::BrokenSystem)
    w.broken === :zero_keeps_b ? fill!(w.inner.A, 0.0) : DLS.zero!(w.inner)
    w.broken === :zero_skips_solve && (fill!(DLS.getx(w), 0.0); w.skip = true)
    return w
end
DLS.dampingfloor(w::BrokenSystem) = w.broken === :floor_zero ? 0.0 : DLS.dampingfloor(w.inner)
DLS.max_diag(w::BrokenSystem) = w.broken === :max_diag_skips_nan ? maximum(x -> isnan(x) ? 0.0 : abs(x), diag(w.inner.A)) :
                                w.broken === :max_diag_is_min ? minimum(abs, diag(w.inner.A)) : DLS.max_diag(w.inner)
DLS.quadratic_model_change(w::BrokenSystem, dx) = DLS.quadratic_model_change(w.inner, dx) + (w.broken === :model_change_offset)

# A system that forwards to another and reports its element type as Float64
struct ClaimsFloat64{S} <: DLS.AbstractLinearSystem
    inner::S
end
for f in (:getA, :getb, :getx, :max_diag, :dampingfloor, :isfinitesystem)
    @eval DLS.$f(w::ClaimsFloat64) = DLS.$f(w.inner)
end
DLS.solve!(w::ClaimsFloat64, s) = DLS.solve!(w.inner, s)
DLS.dampen!(w::ClaimsFloat64, λ::Real) = (DLS.dampen!(w.inner, λ); w)
DLS.dampen!(w::ClaimsFloat64, λ::Real, sc) = (DLS.dampen!(w.inner, λ, sc); w)
DLS.zero!(w::ClaimsFloat64) = (DLS.zero!(w.inner); w)
DLS.copy_structure(w::ClaimsFloat64) = ClaimsFloat64(DLS.copy_structure(w.inner))
DLS.sharex(w::ClaimsFloat64, from) = ClaimsFloat64(DLS.sharex(w.inner, from.inner))
DLS.systemeltype(::ClaimsFloat64) = Float64

checker_message(make_ls; kwargs...) = try check_linear_system(make_ls; kwargs...); nothing catch e; e end

@testset "conformance.jl" begin
    solvers = (:auto, :cholesky, :ldl, :qr)

    @testset "StaticLinearSystem" begin
        for n in (1, 2, 5)
            @test check_linear_system(s -> DLS.makehessiansystem(static(n), s); n, linsolvers=solvers) === nothing
            @test check_linear_system(s -> DLS.makehessiansystem(zeros(MMatrix{n, n}), s); n, linsolvers=solvers) === nothing
            @test DLS.makehessiansystem(static(n), :auto) isa DLS.StaticLinearSystem
        end
    end

    @testset "DenseLinearSystem" begin
        for n in (1, 6)
            @test check_linear_system(s -> DLS.makehessiansystem(zeros(n, n), s); n, linsolvers=solvers) === nothing
            # also without the preallocated :ldl workspace
            @test check_linear_system(() -> DLS.DenseLinearSystem{Float64}(zeros(n, n), zeros(n), zeros(n)); n, linsolvers=solvers) === nothing
            @test DLS.makehessiansystem(zeros(n, n), :auto) isa DLS.DenseLinearSystem
        end
    end

    @testset "SparseLinearSystem" begin
        n = 8
        pattern = spdiagm(-2 => ones(n - 2), -1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1), 2 => ones(n - 2))
        pattern[1, n] = pattern[n, 1] = 1.0
        @test check_linear_system(s -> DLS.makehessiansystem(copy(pattern), s); n, linsolvers=solvers) === nothing
        # a prototype missing diagonal entries (they are inserted) and an Int32 index type
        missing_diag = sparse([1, 2, 3, 1, 4, 5, 6, 7, 8, 8, 7], [2, 1, 1, 3, 4, 5, 6, 7, 8, 7, 8], ones(11), n, n)
        @test check_linear_system(s -> DLS.makehessiansystem(copy(missing_diag), s); n, linsolvers=solvers) === nothing
        @test check_linear_system(s -> DLS.makehessiansystem(SparseMatrixCSC{Float64, Int32}(pattern), s); n, linsolvers=solvers) === nothing
        @test DLS.makehessiansystem(copy(pattern), :auto) isa DLS.SparseLinearSystem
    end

    @testset "Krylov solvers" begin
        using Krylov
        n = 8
        pattern = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
        @test check_linear_system(s -> DLS.makehessiansystem(copy(pattern), s); n, linsolvers=(:cg, :minres), rtol=1e-6) === nothing
        @test check_linear_system(s -> DLS.makehessiansystem(zeros(n, n), s); n, linsolvers=(:cg, :minres), rtol=1e-6) === nothing
    end

    @testset "Pardiso" begin
        using Pardiso
        if Pardiso.mkl_is_available() || Pardiso.panua_is_available()
            n = 8
            pattern = spdiagm(-1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1))
            @test check_linear_system(s -> DLS.makehessiansystem(copy(pattern), s); n, linsolvers=(:pardiso,)) === nothing
            @test check_linear_system(s -> DLS.makehessiansystem(zeros(n, n), s); n, linsolvers=(:pardiso,)) === nothing
        else
            @test_skip "Pardiso library not available"
        end
    end

    # A Float32 system, held to the tolerances in proportion to eps(Float32) (single-precision.md, SP-D6)
    @testset "Float32 systems" begin
        n = 6
        pattern = spdiagm(-2 => ones(n - 2), -1 => ones(n - 1), 0 => ones(n), 1 => ones(n - 1), 2 => ones(n - 2))
        sparse32 = Float32 <: SparseArrays.CHOLMOD.VTypes ? (:auto, :cholesky, :ldl) : (:auto, :ldl)
        @test check_linear_system(s -> DLS.makehessiansystem(zeros(MMatrix{n, n, Float32}), s); n, linsolvers=solvers) === nothing
        @test check_linear_system(s -> DLS.makehessiansystem(zeros(Float32, n, n), s); n, linsolvers=solvers) === nothing
        @test check_linear_system(() -> DLS.DenseLinearSystem{Float32}(zeros(Float32, n, n), zeros(n), zeros(n)); n, linsolvers=solvers) === nothing
        @test check_linear_system(s -> DLS.makehessiansystem(SparseMatrixCSC{Float32, Int}(pattern), s); n, linsolvers=sparse32) === nothing
        for m in (9, 4)
            @test check_linear_system(s -> DLS.makejacobiansystem(zeros(Float32, m, n), n, s); n, linsolvers=(:auto, :qr, :cholesky, :ldl)) === nothing
        end
        @test check_linear_system(s -> DLS.makejacobiansystem(zeros(Float32, 6, 3), static(3), s); n=3, linsolvers=(:auto, :qr)) === nothing
        # The check is not weakened for Float64: a Float32 system that says it is Float64 fails it
        err = checker_message(() -> ClaimsFloat64(DLS.DenseLinearSystem{Float32}(zeros(Float32, n, n), zeros(n), zeros(n))); n, linsolvers=(:cholesky,))
        @test err isa ErrorException && occursin("round-trip", err.msg)
    end

    @testset "The checker catches broken systems" begin
        n = 4
        @test checker_message(() -> BrokenSystem(n, :none); n, linsolvers=solvers) === nothing
        for (broken, rule, detail) in ((:copy_aliases_A, "copy_structure gives its own A", ""),
                                       (:solve_mutates_A, "solve! leaves A as it was written", ""),
                                       (:dampen_keeps_scaling, "dampen!(ls, λ) solves (A + λI) x = b", ""),
                                       (:dampen_ignores_scaling, "dampen!(ls, λ, s) solves (A + λ Diagonal(s)) x = b", ""),
                                       (:drift_scaled, "dampen! replaces the damping set before (no drift)", "returning to dampen!(ls, λ, s)"),
                                       (:drift_uniform, "dampen! replaces the damping set before (no drift)", "returning to dampen!(ls, λ) "),
                                       (:drift_undamped, "dampen! replaces the damping set before (no drift)", "returning to no damping"),
                                       (:copy_shares_cache, "copy_structure shares nothing a solve corrupts", ""),
                                       (:sharex_copies, "sharex(trial, ls) aliases x", ""),
                                       (:model_change_offset, "quadratic_model_change(ls, dx) == dx' A dx / 2 + b' dx", ""),
                                       (:zero_keeps_b, "zero! zeroes b", ""),
                                       (:zero_skips_solve, "solve! solves A * x = b", "after zero! and refilling"),
                                       (:max_diag_is_min, "max_diag is the largest absolute diagonal entry", ""),
                                       (:max_diag_skips_nan, "max_diag is NaN if a diagonal entry is NaN", ""),
                                       (:floor_zero, "dampingfloor is a relative damping in (0, 1e-6]", ""))
            err = checker_message(() -> BrokenSystem(n, broken); n, linsolvers=(:cholesky,))
            @test err isa ErrorException
            @test occursin("rule \"$rule\"", err.msg) && occursin(detail, err.msg)
        end
        # Jacobian storage: not exact, not accumulated into, and of rank min(m, n)
        for m in (2n, n - 1)
            @test checker_message(() -> BrokenSystem(DLS.DenseJacobianLinearSystem(m, n), :none); n, linsolvers=(:qr,)) === nothing
        end
        for (m, broken, rule) in ((n - 1, :singular_success, "an undamped solve of a singular A does not return Success"),
                                  (2n, :zero_skips_solve, "zero! invalidates what was computed from the old A"))
            err = checker_message(() -> BrokenSystem(DLS.DenseJacobianLinearSystem(m, n), broken); n, linsolvers=(:qr,))
            @test err isa ErrorException
            @test occursin("rule \"$rule\"", err.msg)
        end
        # A sparse Jacobian that does not store every entry cannot hold the test matrix
        Jsp = sparse([i >= j ? 1.0 : 0.0 for i in 1:2n, j in 1:n])
        err = checker_message(s -> DLS.makejacobiansystem(copy(Jsp), n, s); n, linsolvers=(:qr,))
        @test err isa ErrorException && occursin("rule \"conformance_fill! and conformance_matrix round-trip\"", err.msg) &&
              occursin("relative", err.msg)
        # the wrong dimension
        err = checker_message(() -> BrokenSystem(n + 1, :none); n, linsolvers=(:auto,))
        @test err isa ErrorException && occursin("getb has length n", err.msg)
    end
end
