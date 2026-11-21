using DampedLinearSolve, SparseArrays, LinearAlgebra, Test, Static
const DLS = DampedLinearSolve

# Conditioning inside solve! (src/conditioning.jl). Each conditioned system is built through the internal builders,
# `DLS.hessiansystem(H, linsolver, condition)` and `DLS.jacobiansystem(J, n, linsolver, condition)`, with the
# conditioning on (what makehessiansystem and makejacobiansystem build) and off.
cd_matrix(n) = (M = [sin(1.7 * (i + j) + 0.3 * i * j) for i in 1:n, j in 1:n]; M = (M + M') / 2; M + Diagonal(vec(sum(abs, M; dims = 2)) .+ 1))
cd_jacobian(m, n) = [sin(0.9 * i + 1.1 * j + 0.2 * i * j) + (i == j ? 2.0 : 0.0) for i in 1:m, j in 1:n]
cd_relerr(x, y) = norm(x - y) / norm(y)
cd_relcomp(x, y) = maximum(abs.(x .- y) ./ abs.(y))
cd_allocs(ls, s) = (solve!(ls, s); @allocated solve!(ls, s))
cd_fillj!(ls, J, r) = (getJ(getA(ls)) .= J; getr(getA(ls)) .= r; gradient!(getb(ls), getA(ls)); markstale!(getA(ls)); ls)

const CD_N, CD_M = 8, 14
const CD_SPARSE32 = Float32 <: SparseArrays.CHOLMOD.VTypes
# Every conditioned Hessian system and solver, for each element type: (prototype maker, solver)
cd_hessian_cases(::Type{Float64}) = [(A -> Matrix{Float64}(A), s) for s in (:auto, :cholesky, :ldl, :qr)] ∪
                                    [(A -> sparse(Float64.(A)), s) for s in (:auto, :cholesky, :ldl, :qr)]
cd_hessian_cases(::Type{Float32}) = [(A -> Float32.(A), s) for s in (:auto, :cholesky, :ldl, :qr)] ∪
                                    [(A -> sparse(Float32.(A)), s) for s in (CD_SPARSE32 ? (:auto, :cholesky, :ldl) : (:auto, :ldl))]
# Every conditioned Jacobian system and solver
cd_jacobian_cases(::Type{Float64}) = [(J -> Matrix{Float64}(J), s) for s in (:auto, :qr, :cholesky, :ldl)] ∪ [(J -> sparse(J), s) for s in (:auto, :qr)]
cd_jacobian_cases(::Type{Float32}) = [(J -> Float32.(J), s) for s in (:auto, :qr, :cholesky, :ldl)]
const CD_DAMPINGS = ((0.0, nothing), (0.37, nothing), (2.9, [1.0 + i / CD_N for i in 1:CD_N]))

@testset "conditioning.jl" begin
    @testset "the scale: 1/(1 + √d), the nearest power of two" begin
        for d in (0.0, 0.5, 1.0, 3.0, 8.0, 1e6, 1e-12, 1e12, 3e38)
            s = DLS.conditioning_scale(d, Float64)
            @test ispow2(s) && s <= 1
            @test abs(log2(s) - log2(1 / (1 + sqrt(d)))) <= 0.5
        end
        @test DLS.conditioning_scale(0.0, Float64) == 1.0
        @test DLS.conditioning_scale(8.0, Float64) == 0.25   # 1/(1 + 2.83) = 0.261
        for d in (-1.0, NaN, Inf, -Inf)
            @test DLS.conditioning_scale(d, Float32) == 1.0
        end
        # at least floatmin(T): a normal number in the system's element type
        @test DLS.conditioning_scale(1e300, Float32) == floatmin(Float32)
        @test ispow2(DLS.conditioning_scale(1e300, Float64)) && DLS.conditioning_scale(1e300, Float64) < 1e-149
    end

    A64, b = cd_matrix(CD_N), [cos(1.3 * i) for i in 1:CD_N]
    J64, r = cd_jacobian(CD_M, CD_N), [cos(0.7 * i) for i in 1:CD_M]
    @testset "on and off give the same x, $T" for T in (Float64, Float32)
        for (make, s) in cd_hessian_cases(T)
            on, off = DLS.hessiansystem(make(A64), s, true), DLS.hessiansystem(make(A64), s, false)
            @test on.condition && !off.condition
            getb(on) .= b; getb(off) .= b
            A0, b0 = copy(getA(on)), copy(getb(on))
            for (λ, sc) in CD_DAMPINGS
                dampen!(on, λ, sc); dampen!(off, λ, sc)
                @test solve!(on, s) == Success && solve!(off, s) == Success
                Ad = A64 + λ * (isnothing(sc) ? I : Diagonal(sc))
                @test cd_relerr(getx(on), getx(off)) <= 8 * CD_N * eps(T) * cond(Ad)
                # A and b as filled, bit for bit
                @test getA(on) == A0 && getb(on) == b0
            end
            # Damping replaced, not accumulated: returning to an earlier damping gives the identical x
            dampen!(on, 0.37); solve!(on, s); x1 = copy(getx(on))
            dampen!(on, 2.9, CD_DAMPINGS[3][2]); solve!(on, s)
            dampen!(on, 0.37); solve!(on, s)
            @test getx(on) == x1
            # Nothing allocates (sparse :qr and CHOLMOD allocate inside SuiteSparse; dense :qr inside LinearAlgebra)
            (s in (:auto, :ldl) || (s === :cholesky && getA(on) isa Matrix)) && @test cd_allocs(on, static(s)) == 0
        end
        for (make, s) in cd_jacobian_cases(T)
            on, off = DLS.jacobiansystem(make(J64), CD_N, s, true), DLS.jacobiansystem(make(J64), CD_N, s, false)
            @test on.condition && !off.condition
            cd_fillj!(on, J64, r); cd_fillj!(off, J64, r)
            J0, r0, b0 = copy(getJ(getA(on))), copy(getr(getA(on))), copy(getb(on))
            for (λ, sc) in CD_DAMPINGS
                dampen!(on, λ, sc); dampen!(off, λ, sc)
                @test solve!(on, s) == Success && solve!(off, s) == Success
                Ad = J64' * J64 + λ * (isnothing(sc) ? I : Diagonal(sc))
                @test cd_relerr(getx(on), getx(off)) <= 8 * CD_N * eps(T) * cond(Ad)
                @test getJ(getA(on)) == J0 && getr(getA(on)) == r0 && getb(on) == b0
            end
            dampen!(on, 0.37); solve!(on, s); x1 = copy(getx(on))
            dampen!(on, 2.9, CD_DAMPINGS[3][2]); solve!(on, s)
            dampen!(on, 0.37); solve!(on, s)
            @test getx(on) == x1
            getJ(getA(on)) isa Matrix && @test cd_allocs(on, static(s)) == 0
        end
    end

    # Decision 2: as λ → ∞ with Levenberg damping, λx → b in the original variables. Uniform damping in the scaled
    # variables would give λx → S²b instead, whose direction differs wherever the scales differ: this matrix's diagonal
    # (from 5 to 400) gives several scales, so the same test of such a solve would fail.
    As = Diagonal(1.0:CD_N) * A64 * Diagonal(1.0:CD_N)
    Js = J64 * Diagonal(1.0:CD_N)
    @testset "the small-step limit, $T" for T in (Float64, Float32)
        @test length(unique(DLS.conditioning_scale.(diag(As), T))) > 1
        @test length(unique(DLS.conditioning_scale.(vec(sum(abs2, Js; dims = 1)), T))) > 1
        for (make, s) in cd_hessian_cases(T)
            ls = makehessiansystem(make(As), s)
            getb(ls) .= b
            dampen!(ls, 1e12 * max_diag(ls))
            @test solve!(ls, s) == Success
            @test norm(getx(ls) / norm(getx(ls)) - b / norm(b)) <= 1e-6
        end
        for (make, s) in cd_jacobian_cases(T)
            ls = makejacobiansystem(make(Js), CD_N, s)
            cd_fillj!(ls, Js, r)
            dampen!(ls, 1e12 * max_diag(ls))
            @test solve!(ls, s) == Success
            g = getb(ls)
            dir = norm(getx(ls) / norm(getx(ls)) - g / norm(g))
            if T === Float32 && s in (:auto, :qr)
                # Not met (single-precision.md, Status): the damped QR factorizes the stacked [R; √λ D^½ S], whose
                # rows of R are about 1e-6 of the damping rows here, so the coupling between them is resolved to about
                # eps(Float32)·1e6. Measured 0.12 on a well-scaled J (1.1e-10 in Float64). Independent of the conditioning.
                @test_broken dir <= 1e-6
                @test dir < 0.5
            else
                @test dir <= 1e-6
            end
        end
    end

    # A badly scaled matrix: the diagonal spans 1e-12 to 1e12 (A = D M D, M well conditioned). With power-of-two scales,
    # conditioning changes nothing in a factorization that does not pivot (Cholesky, LDLᵀ, Householder QR), barring
    # overflow and underflow, so Float32 meets the reference with it off as with it on; it changes the pivoted QR's
    # column order.
    @testset "a badly scaled matrix" begin
        D = Diagonal(10.0 .^ range(-6, 6; length = CD_N))
        Abad = D * A64 * D
        ref = Abad \ b
        # componentwise, the error of each x_i relative to itself; in Float64 the reference's own error is of the same size
        tol(T) = 32 * CD_N * eps(T) * cond(A64)
        for T in (Float64, Float32), (make, s) in cd_hessian_cases(T)
            on, off = DLS.hessiansystem(make(Abad), s, true), DLS.hessiansystem(make(Abad), s, false)
            getb(on) .= b; getb(off) .= b
            @test solve!(on, s) == Success && solve!(off, s) == Success
            if s === :qr
                # Pivoted QR: the scale 1/(1 + √d) never scales a weak variable up (s ≤ 1), so the conditioned diagonal
                # still spans 1e-12 to 1, and Float32's pivoted QR does not meet the reference (single-precision.md,
                # Status): measured componentwise error 1.0. In Float64 conditioning is what makes it meet it
                # (2.5e-11 on; 1.0 off, observed).
                if T === Float64
                    @test cd_relcomp(getx(on), ref) <= 1e-9
                    @test cd_relcomp(getx(off), ref) > 0.1
                else
                    @test_broken cd_relcomp(getx(on), ref) <= tol(T)
                    @test cd_relcomp(getx(off), ref) > 0.1
                end
            else
                @test cd_relcomp(getx(on), ref) <= tol(T)
                @test getx(off) == getx(on)   # observed: no difference
            end
        end
        # The Jacobian of the same scaling (columns scaled from 1e-6 to 1e6): every solver meets it, on or off, in either type
        Jbad = J64 * D
        jref = Jbad \ r
        for T in (Float64, Float32), (make, s) in cd_jacobian_cases(T)
            on, off = DLS.jacobiansystem(make(Jbad), CD_N, s, true), DLS.jacobiansystem(make(Jbad), CD_N, s, false)
            cd_fillj!(on, Jbad, r); cd_fillj!(off, Jbad, r)
            @test solve!(on, s) == Success && solve!(off, s) == Success
            @test cd_relcomp(getx(on), jref) <= 8 * CD_N * eps(T) * cond(J64)^2
            @test cd_relcomp(getx(off), jref) <= 8 * CD_N * eps(T) * cond(J64)^2
        end
    end

    @testset "the copies keep the switch" begin
        for on in (false, true)
            for ls in (DLS.hessiansystem(copy(A64), :auto, on), DLS.hessiansystem(sparse(A64), :auto, on),
                       DLS.jacobiansystem(copy(J64), CD_N, :qr, on), DLS.jacobiansystem(copy(J64), CD_N, :cholesky, on),
                       DLS.jacobiansystem(sparse(J64), CD_N, :qr, on))
                @test ls.condition == on && copy_structure(ls).condition == on
                @test sharex(copy_structure(ls), makehessiansystem(copy(A64), :auto)).condition == on
            end
        end
        # The static systems are not conditioned
        @test !hasfield(typeof(makehessiansystem(static(3), :auto)), :condition)
        @test !hasfield(typeof(makejacobiansystem(4, static(2), :qr)), :condition)
    end
end
