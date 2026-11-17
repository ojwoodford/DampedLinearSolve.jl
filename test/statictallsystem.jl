using DampedLinearSolve, StaticArrays, LinearAlgebra, Static, Random, Test
using DampedLinearSolve: dampen!, zero!, Success, NotPositiveDefinite, check_linear_system
using DampedLinearSolve: getA, getb, getx, solve!
const DLS = DampedLinearSolve

# TP4: the static tall system for small problems, against the dense tall system

function tp4_set!(ls, r)
    zero!(ls)
    getA(ls).r .= r
    DLS.jtr!(getb(ls), getA(ls).J, getA(ls).r)
    return ls
end
tp4_solve!(ls, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(:qr)))

# The exponential fit (20 residuals, 2 variables: m·n = 40, under the cap), without allocating
const TP4_T = collect(range(0, 1; length = 20))
const TP4_Y = @. 2.0 * exp(-1.5 * TP4_T)
function tp4_cost(x)
    total = 0.0
    for i in eachindex(TP4_T)
        total += (x[1] * exp(x[2] * TP4_T[i]) - TP4_Y[i])^2
    end
    return 0.5 * total
end
function tp4_resjac!(r, J, x)
    for i in eachindex(TP4_T)
        e = exp(x[2] * TP4_T[i])
        r[i] = x[1] * e - TP4_Y[i]
        J[i, 1] = e
        J[i, 2] = x[1] * TP4_T[i] * e
    end
    return tp4_cost(x)
end

@testset "statictallsystem.jl" begin
    @testset "Steps equal the dense tall system's: ($m, $n)" for (m, n) in ((8, 3), (3, 3), (12, 1), (2, 5), (6, 6), (24, 2))
        rng = Xoshiro(m + 10n)
        J, r = randn(rng, m, n), randn(rng, m)
        ls, ld = tp4_set!(DLS.StaticTallLinearSystem(copy(J)), r), tp4_set!(DLS.DenseTallLinearSystem(copy(J)), r)
        @test getb(ls) ≈ getb(ld) rtol = 1e-14
        v = randn(rng, n)
        @test DLS.quadform(getA(ls), v) ≈ DLS.quadform(getA(ld), v) rtol = 1e-14
        @test DLS.max_diag(getA(ls)) == DLS.max_diag(getA(ld))
        for (lambda, scaling) in ((0.0, nothing), (0.3, nothing), (0.3, 0.5 .+ rand(rng, n)), (1e-10, nothing))
            a, b = tp4_solve!(ls, lambda, scaling), tp4_solve!(ld, lambda, scaling)
            @test a == b == (m < n && lambda == 0 ? NotPositiveDefinite : Success)
            a == Success && @test getx(ls) ≈ getx(ld) rtol = 1e-12
        end
        # a duplicated column is rank deficient undamped, as for the dense system
        Jr = hcat(J, J[:, 1])
        if m * (n + 1) <= 64
            lr, lq = tp4_set!(DLS.StaticTallLinearSystem(copy(Jr)), r), tp4_set!(DLS.DenseTallLinearSystem(copy(Jr)), r)
            @test tp4_solve!(lr, 0.0, nothing) == tp4_solve!(lq, 0.0, nothing)
        end
    end

    @testset "Non-finite J or r" begin
        for (J, r) in (([1.0 NaN; 0.0 1.0; 1.0 1.0], ones(3)), ([1.0 0.0; 0.0 1.0; 1.0 1.0], [1.0, Inf, 0.0]))
            ls = tp4_set!(DLS.StaticTallLinearSystem(copy(J)), r)
            @test tp4_solve!(ls, 1.0, nothing) == NotPositiveDefinite
            @test all(isnan, getx(ls))
        end
        @test_throws ArgumentError solve!(DLS.StaticTallLinearSystem(ones(3, 2)), static(:cholesky))
    end

    @testset "Meets the linear-system interface: ($m, $n)" for (m, n) in ((8, 5), (5, 5), (3, 5), (8, 1))
        @test isnothing(check_linear_system(s -> DLS.StaticTallLinearSystem(zeros(m, n)); n, linsolvers = (:auto, :qr)))
    end



end

# Paths the solve-level tests moved from SecondOrderOptimization did not reach on their own (rules.md 28)
@testset "statictallsystem.jl: the remaining branches" begin
    J = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    ls = DLS.StaticTallLinearSystem(copy(J))
    A = getA(ls)
    getr(A) .= [1.0, -1.0, 2.0]
    @test size(A) == (2, 2) && size(A, 1) == 2 && size(A, 2) == 2
    @test getJ(A) === A.J && getJ(A) == J
    @test getr(A) === A.r
    # J v and Jᵀ r
    @test DLS.jmul!(zeros(3), getJ(A), [1.0, -2.0]) ≈ J * [1.0, -2.0]
    g = zeros(2)
    @test gradient!(g, A) === g
    @test g ≈ J' * [1.0, -1.0, 2.0]
    # isfinitesystem looks at both J and r
    @test isfinitesystem(ls)
    getr(A)[2] = NaN
    @test !isfinitesystem(ls)
    getr(A)[2] = 0.0
    getJ(A)[3, 1] = -Inf
    @test !isfinitesystem(ls)
end
