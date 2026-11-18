using DampedLinearSolve, SparseArrays, LinearAlgebra, Static, Random, Test
using DampedLinearSolve: dampen!, zero!, copy_structure, Success, NotPositiveDefinite, check_linear_system
using DampedLinearSolve: getA, getb, getx, solve!   # explicit: Krylov, loaded by other test files, also exports solve!
const DLS = DampedLinearSolve

# How an optimizer fills a tall system: the linearization writes J and r, then b = Jᵀr is formed and
# what was computed from the old J and r is invalidated
include("jacobianfill.jl")

# TP1: the sparse tall system (a SparseMatrixCSC{Float64, Int} Jacobian, solved by SPQR) against
# the dense tall system on the same J

# Both systems set to J and r as a linearization leaves them (b = Jᵀr)
function tp1_pair(J::SparseMatrixCSC, r)
    ls, ld = DLS.SparseTallLinearSystem(copy(J)), DLS.DenseTallLinearSystem(Matrix(J))
    for s in (ls, ld)
        zero!(s)
        getA(s).r .= r
        DLS.jtr!(getb(s), getA(s).J, getA(s).r)
    end
    return ls, ld
end

tp1_solve!(ls, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(:qr)))
tp1_allocations(ls, lambda) = (tp1_solve!(ls, lambda, nothing); Base.@allocations tp1_solve!(ls, lambda, nothing))
tp1_bytes(f::F, args::Vararg{Any, N}) where {F, N} = (f(args...); @allocated f(args...))

# An exponential fit, written without allocating, with J's
# stored entries set through `nonzeros` (the prototype stores both full columns)
const TP1_T = collect(range(0, 1; length = 20))
const TP1_Y = @. 2.0 * exp(-1.5 * TP1_T)
function tp1_cost(x)
    total = 0.0
    for i in eachindex(TP1_T)
        total += (x[1] * exp(x[2] * TP1_T[i]) - TP1_Y[i])^2
    end
    return 0.5 * total
end
function tp1_resjac!(r, J, x)
    m = length(TP1_T)
    for i in 1:m
        e = exp(x[2] * TP1_T[i])
        r[i] = x[1] * e - TP1_Y[i]
        if J isa SparseMatrixCSC
            nonzeros(J)[i] = e
            nonzeros(J)[m + i] = x[1] * TP1_T[i] * e
        else
            J[i, 1] = e
            J[i, 2] = x[1] * TP1_T[i] * e
        end
    end
    return tp1_cost(x)
end

@testset "sparsejacobiansystem.jl" begin
    @testset "Steps equal the dense tall system's: ($m, $n)" for (m, n) in ((30, 6), (6, 6), (30, 1), (4, 6))
        rng = Xoshiro(m + 10n)
        J = sprandn(rng, m, n, 0.4) + sparse(1:min(m, n), 1:min(m, n), 1.0, m, n)
        ls, ld = tp1_pair(J, randn(rng, m))
        @test getb(ls) ≈ getb(ld) rtol = 1e-14
        v = randn(rng, n)
        @test DLS.quadform(getA(ls), v) ≈ DLS.quadform(getA(ld), v) rtol = 1e-14
        @test DLS.max_diag(getA(ls)) == DLS.max_diag(getA(ld))
        for (lambda, scaling) in ((0.0, nothing), (0.3, nothing), (0.3, 0.5 .+ rand(rng, n)), (1e-10, nothing))
            a, b = tp1_solve!(ls, lambda, scaling), tp1_solve!(ld, lambda, scaling)
            @test a == b
            # undamped and wide: rank deficient, as for the dense system
            @test a == (m < n && lambda == 0 ? NotPositiveDefinite : Success)
            a == Success && @test getx(ls) ≈ getx(ld) rtol = 1e-12
        end
    end

    @testset "Rank deficiency and non-finite entries" begin
        rng = Xoshiro(5)
        J0 = sparse(randn(rng, 20, 4))
        J = [J0 J0[:, 2]]                  # a duplicated column
        ls, ld = tp1_pair(J, randn(rng, 20))
        @test tp1_solve!(ls, 0.0, nothing) == NotPositiveDefinite == tp1_solve!(ld, 0.0, nothing)
        @test tp1_solve!(ls, 0.1, nothing) == Success == tp1_solve!(ld, 0.1, nothing)
        @test getx(ls) ≈ getx(ld) rtol = 1e-10
        # a structurally empty column
        Jz = sparse([1, 2, 3], [1, 1, 3], [1.0, 2.0, 3.0], 5, 3)
        lz, _ = tp1_pair(Jz, ones(5))
        @test tp1_solve!(lz, 0.0, nothing) == NotPositiveDefinite
        @test tp1_solve!(lz, 1.0, nothing) == Success
        # NaN or Inf in J or r: NotPositiveDefinite, and x is NaN
        for (Jbad, rbad) in ((sparse([1.0 NaN; 0.0 1.0; 1.0 1.0]), ones(3)), (sparse([1.0 0.0; 0.0 1.0; 1.0 1.0]), [1.0, Inf, 0.0]))
            lb, _ = tp1_pair(Jbad, rbad)
            @test tp1_solve!(lb, 1.0, nothing) == NotPositiveDefinite
            @test all(isnan, getx(lb))
        end
        @test_throws ArgumentError solve!(ls, static(:cholesky))
    end

    @testset "Meets the linear-system interface: ($m, $n)" for (m, n) in ((8, 5), (5, 5), (3, 5), (8, 1))
        @test isnothing(check_linear_system(s -> DLS.SparseTallLinearSystem(sparse(ones(m, n))); n, linsolvers = (:auto, :qr)))
    end


    @testset "Allocation: only SPQR's working memory, one fixed set per solve" begin
        ls, _ = tp1_pair(sprandn(Xoshiro(2), 40, 6, 0.4) + sparse(1:6, 1:6, 1.0, 40, 6), randn(Xoshiro(3), 40))
        perSolve = tp1_allocations(ls, 1.0)
        @test perSolve > 0                                   # the stated exception (SPQR)
        @test tp1_allocations(ls, 0.0) == tp1_allocations(ls, 10.0) == perSolve
        # everything on this side of SPQR allocates nothing
        v = ones(6)
        @test tp1_bytes(DLS.quadform, getA(ls), v) == 0
        @test tp1_bytes(DLS.jtr!, getb(ls), getA(ls).J, getA(ls).r) == 0
        @test tp1_bytes(DLS.max_diag, getA(ls)) == 0
        @test tp1_bytes(TallFill(tp1_resjac!), zeros(2), getA(DLS.SparseTallLinearSystem(sparse(ones(20, 2)))), [1.0, 0.0]) == 0
    end

end

# Paths the solve-level tests moved from SecondOrderOptimization did not reach on their own (rules.md 28)
@testset "sparsejacobiansystem.jl: the remaining branches" begin
    J = sparse([1.0 0.0; 3.0 4.0; 0.0 6.0])
    ls = DLS.SparseTallLinearSystem(copy(J))
    A = getA(ls)
    @test size(A) == (2, 2) && size(A, 1) == 2 && size(A, 2) == 2
    # isfinitesystem looks at both J's stored values and r
    @test isfinitesystem(ls)
    getr(A)[2] = NaN
    @test !isfinitesystem(ls)
    getr(A)[2] = 0.0
    nonzeros(getJ(A))[1] = Inf
    @test !isfinitesystem(ls)
end
