# SP-D12: the held factors of a dense Jacobian system's QR solve (src/qrfactors.jl)
using DampedLinearSolve, LinearAlgebra, Random, SparseArrays, Static, Test
const DLS = DampedLinearSolve

qf_fill!(ls, J, r) = (getJ(getA(ls)) .= J; getr(getA(ls)) .= r; gradient!(getb(ls), getA(ls)); markstale!(getA(ls)); ls)
# A system of element type T for J (m × n, with column norms spread over a few decades, so that the scale is not 1)
function qf_system(::Type{T}, m, n, seed; condition = true, linsolver = :qr) where T
    rng = Xoshiro(seed)
    J = T.(randn(rng, m, n) .* exp10.(range(-2, 2; length = n))')
    r = randn(rng, m)
    ls = qf_fill!(DLS.jacobiansystem(copy(J), n, linsolver, condition), J, r)
    return ls, J, r
end
# M as filled, in Float64, at the damping λ·D
qf_M(J, λ, D) = λ > 0 ? vcat(Float64.(J), Diagonal(sqrt.(λ .* D))) : Float64.(J)
qf_rhs(r, n, λ) = λ > 0 ? vcat(r, zeros(n)) : copy(r)
qf_rel(a, b) = norm(a - b) / norm(b)
# One step of SP-D11's refinement written with the public operations alone, in the system's own variables
function qf_refined(Q, R, M, c)
    p, n = size(M)
    d = lmul!(Q', copy(c))
    x = ldiv!(R, d[1:n])
    d[1:n] .= 0
    rho = lmul!(Q, d)
    f1 = c - rho - M * x
    h = ldiv!(R', -(M' * rho))
    d = lmul!(Q', f1)
    dx = ldiv!(R, d[1:n] - h)
    return x + dx
end
function qf_allocs(ls, v, u)
    Q, R = qrfactors(ls)
    a = @allocated qrfactors(ls)
    a += @allocated lmul!(Q', v)
    a += @allocated lmul!(Q, v)
    a += @allocated ldiv!(R, u)
    a += @allocated ldiv!(R', u)
    return a
end

@testset "qrfactors.jl" begin
    @testset "$T, m = $m, $(condition ? "conditioned" : "unconditioned"), $dname" for T in (Float64, Float32),
            (m, n) in ((30, 5), (6, 6)), condition in (true, false),
            (dname, λ, D) in (("undamped", 0.0, nothing), ("λI", 0.3, nothing), ("Marquardt", 0.3, :colnorms))
        ls, J, r = qf_system(T, m, n, 7m + n; condition)
        Dv = D === :colnorms ? vec(sum(abs2, Float64.(J); dims = 1)) : ones(n)
        D === :colnorms ? dampen!(ls, λ, Dv) : dampen!(ls, λ)
        @test solve!(ls, :qr) == Success
        Q, R = qrfactors(ls)
        p = λ > 0 ? m + n : m
        @test size(Q) == size(Q') == (p, p) && size(R) == size(R') == (n, n)
        M = qf_M(J, λ, Dv)
        rng = Xoshiro(1)
        # Q is orthogonal, to the rounding of its T reflectors
        v = randn(rng, p)
        @test qf_rel(lmul!(Q, lmul!(Q', copy(v))), v) <= 10p * eps(T)
        @test abs(norm(lmul!(Q', copy(v))) - norm(v)) <= 10p * eps(T) * norm(v)
        # M = Q [R; 0]: Mᵀu = Rᵀ (Qᵀu)[1:n]
        u = randn(rng, p)
        @test qf_rel(ldiv!(R', M' * u), lmul!(Q', copy(u))[1:n]) <= 100 * eps(T) * cond(M)
        # The solve before refinement
        c = qf_rhs(r, n, λ)
        x0 = ldiv!(R, lmul!(Q', copy(c))[1:n])
        if T === Float64
            # the solve transforms r with LAPACK, the operations in Float64 loops: the same to rounding
            @test qf_rel(x0, getx(ls)) <= 1e-12 * cond(M)
        else
            x1 = copy(getx(ls))
            markstale!(getA(ls)); @test DLS.solve_qr!(ls, static(:qr), 0) == Success
            @test x0 == getx(ls)
            # SP-D11's step written with the public operations gives the internal step's x (to the rounding of the
            # products with M, which this test forms with BLAS and the refinement in its own loops)
            markstale!(getA(ls)); @test solve!(ls, :qr) == Success
            @test getx(ls) == x1
            Q, R = qrfactors(ls)
            @test qf_rel(qf_refined(Q, R, M, c), x1) <= 1e-13
        end
        # R⁻¹ and R⁻ᵀ are inverse to each other's transposes: (R⁻¹a)·b = a·(R⁻ᵀb)
        a, b = randn(rng, n), randn(rng, n)
        @test dot(ldiv!(R, copy(a)), b) ≈ dot(a, ldiv!(R', copy(b))) rtol = 1e3 * eps(T) * cond(M)
        # Wrong lengths
        @test_throws DimensionMismatch lmul!(Q', zeros(p + 1))
        @test_throws DimensionMismatch lmul!(Q, zeros(p - 1))
        @test_throws DimensionMismatch ldiv!(R, zeros(n + 1))
        @test_throws DimensionMismatch ldiv!(R', zeros(n - 1))
        # Zero allocation
        @test qf_allocs(ls, randn(rng, p), randn(rng, n)) == 0
    end

    @testset "a recomputation of the scale changes neither product (SP-D9), $T" for T in (Float64, Float32)
        ls, J, r = qf_system(T, 20, 4, 3)
        dampen!(ls, 0.0)
        @test solve!(ls, :qr) == Success
        Q, R = qrfactors(ls)
        rng = Xoshiro(2)
        v, a = randn(rng, 20), randn(rng, 4)
        before = (lmul!(Q', copy(v)), lmul!(Q, copy(v)), ldiv!(R, copy(a)), ldiv!(R', copy(a)))
        cs = copy(getA(ls).ws.cs)
        # The held R̂ rescaled to the scale at a heavy damping, as a solve at that damping does
        DLS.rescale_factor!(getA(ls), 1e12 * max_diag(ls), nothing)
        @test getA(ls).ws.cs != cs && getA(ls).ws.nrescale == 1
        Q, R = qrfactors(ls)
        @test (lmul!(Q', copy(v)), lmul!(Q, copy(v)), ldiv!(R, copy(a)), ldiv!(R', copy(a))) == before
    end

    @testset "validity, $T" for T in (Float64, Float32)
        ls, J, r = qf_system(T, 12, 3, 5)
        # Not solved since the fill
        @test_throws ErrorException qrfactors(ls)
        dampen!(ls, 0.1)
        @test solve!(ls, :qr) == Success
        Q, R = qrfactors(ls)
        # A later dampen! does not change them
        dampen!(ls, 0.0)
        Q2, R2 = qrfactors(ls)
        @test size(Q2) == (15, 15)
        v = randn(Xoshiro(3), 15)
        @test lmul!(Q2', copy(v)) == lmul!(Q', copy(v))
        # Using them changes nothing a later solve reads: the same x as a twin system without them
        twin, _, _ = qf_system(T, 12, 3, 5)
        for (l, use) in ((ls, true), (twin, false))
            dampen!(l, 0.1); @test solve!(l, :qr) == Success
            if use
                Q, R = qrfactors(l)
                lmul!(Q', randn(15)); lmul!(Q, randn(15)); ldiv!(R, randn(3)); ldiv!(R', randn(3))
            end
            dampen!(l, 0.7); @test solve!(l, :qr) == Success
            dampen!(l, 0.0); @test solve!(l, :qr) == Success
        end
        @test getx(ls) == getx(twin)
        # A solve of a system sharing the workspace takes the scratch
        trial = qf_fill!(copy_structure(ls), J, r)
        @test solve!(trial, :qr) == Success
        @test_throws ErrorException qrfactors(ls)
        @test size(qrfactors(trial)[1]) == (12, 12)
        @test solve!(ls, :qr) == Success
        @test size(qrfactors(ls)[1]) == (12, 12)
        # markstale! and zero!
        markstale!(getA(ls)); @test_throws ErrorException qrfactors(ls)
        @test solve!(ls, :qr) == Success
        zero!(ls); @test_throws ErrorException qrfactors(ls)
        # A failed solve: J of rank 2
        Jd = copy(J); Jd[:, 3] .= Jd[:, 1]
        qf_fill!(ls, Jd, r); dampen!(ls, 0.0)
        @test solve!(ls, :qr) == NotPositiveDefinite
        @test_throws ErrorException qrfactors(ls)
        # A non-finite J
        Jn = copy(J); Jn[1, 1] = NaN
        qf_fill!(ls, Jn, r)
        @test solve!(ls, :qr) == NotPositiveDefinite
        @test_throws ErrorException qrfactors(ls)
        # After a success, then a failure, of the same fill: none held
        qf_fill!(ls, Jd, r); dampen!(ls, 0.1)
        @test solve!(ls, :qr) == Success
        dampen!(ls, 0.0)
        @test solve!(ls, :qr) == NotPositiveDefinite
        @test_throws ErrorException qrfactors(ls)
    end

    @testset "systems that do not offer it" begin
        # Fewer residuals than variables, damped (a solve succeeds; the factors are refused)
        ls, _, _ = qf_system(Float64, 3, 5, 1)
        dampen!(ls, 0.1); @test solve!(ls, :qr) == Success
        @test_throws ArgumentError qrfactors(ls)
        # Built for the normal equations
        for s in (:cholesky, :ldl)
            l, _, _ = qf_system(Float32, 10, 3, 1; linsolver = s)
            dampen!(l, 0.1); @test solve!(l, s) == Success
            @test_throws ArgumentError qrfactors(l)
        end
        # Static, sparse, Hessian
        @test_throws ArgumentError qrfactors(makejacobiansystem(zeros(6, 2), static(2), :qr))
        @test_throws ArgumentError qrfactors(makejacobiansystem(sprand(Xoshiro(1), 10, 3, 0.5), 3, :qr))
        @test_throws ArgumentError qrfactors(makehessiansystem(zeros(3, 3), :auto))
    end
end
