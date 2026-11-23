# SP-D11: the mixed-precision refinement of the dense Jacobian QR solve (src/qr_refinement.jl)
using DampedLinearSolve, LinearAlgebra, Random, Static, Test
const DLS = DampedLinearSolve

# J (Float32) with singular values from 1 to κ, and r orthogonal to its range but for ‖Jᵀr‖ = ratio·‖J‖·‖r‖: a residual
# at the solution large against the gradient
function qrr_largeres(m, n, κ, seed; ratio = 1e-6)
    rng = Xoshiro(seed)
    U = Matrix(qr(randn(rng, m, m)).Q); V = Matrix(qr(randn(rng, n, n)).Q)
    J32 = Float32.(U[:, 1:n] * Diagonal(exp.(range(0, log(κ); length = n))) * V')
    J = Float64.(J32)
    Q1 = Matrix(qr(J).Q)[:, 1:n]
    rp = randn(rng, m); rp -= Q1 * (Q1' * rp); rp /= norm(rp)
    x0 = randn(rng, n)
    return J32, rp + J * x0 * (ratio * norm(J) / norm(J' * (J * x0)))
end
# A consistent ill-conditioned problem, plus rounding-level noise
function qrr_smallres(m, n, κ, seed)
    rng = Xoshiro(seed)
    U = Matrix(qr(randn(rng, m, m)).Q); V = Matrix(qr(randn(rng, n, n)).Q)
    J32 = Float32.(U[:, 1:n] * Diagonal(exp.(range(0, log(κ); length = n))) * V')
    b = Float64.(J32) * randn(rng, n)
    return J32, b + 1e-7 * norm(b) / sqrt(m) * randn(rng, m)
end
# The damped least-squares solution of the stored J, in BigFloat
function qrr_reference(J32, r, λ)
    n = size(J32, 2)
    setprecision(BigFloat, 256) do
        A = λ > 0 ? vcat(BigFloat.(J32), sqrt(BigFloat(λ)) * Matrix{BigFloat}(I, n, n)) : BigFloat.(J32)
        c = λ > 0 ? vcat(BigFloat.(r), zeros(BigFloat, n)) : BigFloat.(r)
        Float64.(qr(A) \ c)
    end
end
qrr_fill!(ls, J, r) = (getJ(getA(ls)) .= J; getr(getA(ls)) .= r; gradient!(getb(ls), getA(ls)); markstale!(getA(ls)); ls)
# The relative error of the solve after 0, 1 and 2 steps of refinement, and of the Float64 QR solve of the same J and r
function qrr_errors(J32, r, λrel)
    m, n = size(J32)
    ls = makejacobiansystem(copy(J32), n, :qr)
    qrr_fill!(ls, J32, r)
    λ = λrel * max_diag(ls)
    dampen!(ls, λ)
    xr = qrr_reference(J32, r, λ)
    e = map(0:2) do steps
        markstale!(getA(ls))
        @test DLS.solve_qr!(ls, static(:qr), steps) == Success
        norm(getx(ls) - xr) / norm(xr)
    end
    l64 = qrr_fill!(makejacobiansystem(Float64.(J32), n, :qr), Float64.(J32), r)
    dampen!(l64, λ)
    @test solve!(l64, :qr) == Success
    return (e..., norm(getx(l64) - xr) / norm(xr))
end
qrr_median(v) = sort(v)[cld(length(v), 2)]
function qrr_allocs(ls, s)
    solve!(ls, s)
    return @allocated solve!(ls, s)
end

@testset "qr_refinement.jl" begin
    @testset "storage: Float32 only" begin
        m, n = 9, 4
        l32 = makejacobiansystem(zeros(Float32, m, n), n, :qr)
        rf = getA(l32).ws.scratch.refine
        @test rf isa DLS.QRRefinement && length(rf.rho) == m + n && length(rf.f) == m + n && length(rf.h) == n
        @test eltype(rf.rho) === eltype(rf.f) === eltype(rf.h) === Float64
        # Float64 is not refined: no refinement storage, and the solve is the same with any number of steps
        l64 = makejacobiansystem(zeros(Float64, m, n), n, :qr)
        @test getA(l64).ws.scratch.refine === nothing
        J, r = qrr_largeres(m, n, 10.0, 1)
        qrr_fill!(l64, Float64.(J), r)
        for λ in (0.0, 0.3)
            dampen!(l64, λ)
            x = map(steps -> (markstale!(getA(l64)); DLS.solve_qr!(l64, static(:qr), steps); copy(getx(l64))), 0:2)
            @test x[1] == x[2] == x[3]
        end
        # The normal equations are not refined: their (empty) QR scratch holds none
        @test getA(makejacobiansystem(zeros(Float32, m, n), n, :cholesky)).ws.scratch.refine === nothing
        @test DLS.QR_REFINEMENT_STEPS == 1
    end

    # The error falls by about eps(Float32)·κ per step, to the Float64 solve's (Björck; Higham §20.5): before SP-D11 the
    # Float32 QR's error on this problem was the factor's own, about eps(Float32)·κ²·‖r‖/(‖J‖·‖x‖)
    @testset "large residual, κ = $κ" for κ in (10.0, 1e3), (m, n) in ((60, 6), (12, 4)), λrel in (0.0, 1e-4, 0.1)
        e = [qrr_errors(qrr_largeres(m, n, κ, seed)..., λrel) for seed in 1:8]
        e0, e1, e2, e64 = (qrr_median(getindex.(e, i)) for i in 1:4)
        @test e1 <= (κ == 10 ? 1e-4 : 1e-2) * e0
        @test e2 <= 4 * e64
    end

    # Ill-conditioned with a small residual: no worse than the unrefined QR (and not the eps·κ² growth of a
    # seminormal correction)
    @testset "small residual, κ = 1e5" for (m, n) in ((60, 6), (12, 4)), λrel in (0.0, 1e-4, 0.1)
        e = [qrr_errors(qrr_smallres(m, n, 1e5, seed)..., λrel) for seed in 1:8]
        @test all(x -> x[2] <= x[1] && x[3] <= x[1], e)
    end

    @testset "the contract, and allocation" begin
        m, n = 30, 5
        J, r = qrr_largeres(m, n, 10.0, 3)
        for s in (:auto, :qr)
            ls = qrr_fill!(makejacobiansystem(copy(J), n, s), J, r)
            # Returning to a damping gives the identical x, through a recomputation of the scale in between (SP-D9)
            dampen!(ls, 0.0); @test solve!(ls, s) == Success; x0 = copy(getx(ls))
            dampen!(ls, 0.2); @test solve!(ls, s) == Success; x1 = copy(getx(ls))
            dampen!(ls, 1e16); @test solve!(ls, s) == Success
            dampen!(ls, 0.2); @test solve!(ls, s) == Success; @test getx(ls) == x1
            dampen!(ls, 0.0); @test solve!(ls, s) == Success; @test getx(ls) == x0
            # The system as filled
            @test getJ(getA(ls)) == J && getr(getA(ls)) == r
            # Zero allocation, undamped and damped, with a Marquardt scaling
            dampen!(ls, 0.0); @test qrr_allocs(ls, static(s)) == 0
            dampen!(ls, 0.2, fill(0.5, n)); @test qrr_allocs(ls, static(s)) == 0
        end
        # Fewer residuals than variables, damped: refined over the stacked matrix's rows (its zero rows dropped)
        J3, r3 = Float32.(randn(Xoshiro(4), 3, n)), randn(Xoshiro(5), 3)
        ls = qrr_fill!(makejacobiansystem(copy(J3), n, :qr), J3, r3)
        dampen!(ls, 0.1)
        @test solve!(ls, :qr) == Success
        xr = qrr_reference(J3, r3, 0.1)
        @test norm(getx(ls) - xr) / norm(xr) <= 1e-10
    end
end
