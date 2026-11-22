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
    @testset "the scale: 1/(1e-12 + √d), the nearest power of two" begin
        for d in (0.0, 0.5, 1.0, 3.0, 8.0, 1e6, 1e-12, 1e-30, 1e12, 3e38)
            s = DLS.conditioning_scale(d, Float64)
            @test ispow2(s)
            @test abs(log2(s) - log2(1 / (1e-12 + sqrt(d)))) <= 0.5
        end
        @test DLS.conditioning_scale(0.0, Float64) == 2.0^40   # 1/1e-12 = 2^39.86
        @test DLS.conditioning_scale(1.0, Float64) == 1.0
        @test DLS.conditioning_scale(6.0, Float64) == 0.5   # 1/2.45 = 2^-1.29
        @test DLS.conditioning_scale(1e-6, Float32) == 1024.0
        for d in (NaN, Inf, -Inf)
            @test DLS.conditioning_scale(d, Float32) == 1.0
        end
        @test DLS.conditioning_scale(-6.0, Float64) == 0.5 && DLS.conditioning_scale(-1e-6, Float32) == 1024.0   # by |d|
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
                # Pivoted QR: conditioned, the diagonal is about 1 (decision 13's scale scales weak variables up), and
                # the solve meets the reference in either type; off, the column order follows the unscaled diagonal and
                # it does not (componentwise error 1.0, observed)
                @test cd_relcomp(getx(on), ref) <= (T === Float64 ? 1e-9 : tol(T))
                @test cd_relcomp(getx(off), ref) > 0.1
            else
                @test cd_relcomp(getx(on), ref) <= tol(T)
                @test getx(off) == getx(on)   # observed: no difference
            end
        end
        # The Jacobian of the same scaling (columns scaled from 1e-6 to 1e6): every solver meets it with conditioning on, in
        # either type. Off, the Float32 QR's rank test (relative to eps(Float32) times the largest |R_ii|) reads the
        # unscaled R, whose diagonal spans 1e12, as rank deficient (observed); conditioned, R is nearly equilibrated
        jtol(T) = 8 * CD_N * eps(T) * cond(J64)^2
        Jbad = J64 * D
        jref = Jbad \ r
        for T in (Float64, Float32), (make, s) in cd_jacobian_cases(T)
            on, off = DLS.jacobiansystem(make(Jbad), CD_N, s, true), DLS.jacobiansystem(make(Jbad), CD_N, s, false)
            cd_fillj!(on, Jbad, r); cd_fillj!(off, Jbad, r)
            @test solve!(on, s) == Success
            @test cd_relcomp(getx(on), jref) <= jtol(T)
            if T === Float32 && s in (:auto, :qr)
                @test solve!(off, s) == NotPositiveDefinite
            else
                @test solve!(off, s) == Success
                @test cd_relcomp(getx(off), jref) <= jtol(T)
            end
        end
    end

    # Decision 13: the scale is computed at the first solve after a fill, from the damped diagonal, and reused for later
    # dampings. A zero diagonal entry and one of 1e-30, solved undamped first (scales 2^40), then at λ = 1e16: the
    # stored scale would put λ·s² = 1.2e40 beyond floatmax(Float32), so the scale is recomputed at that damping (the
    # writer's proposal, conditioning.jl); then a first solve at λ₀ > 0 followed by 1e20·λ₀. Finite status and x.
    @testset "a zero and a tiny diagonal entry, then a large damping, $T" for T in (Float64, Float32)
        Az = copy(A64)
        Az[1, :] .= 0; Az[:, 1] .= 0; Az[2, :] .= 0; Az[:, 2] .= 0
        Az[2, 2] = 1e-30
        Jz = copy(J64); Jz[:, 1] .= 0; Jz[:, 2] .= 0; Jz[1, 2] = 1e-15
        for (make, s) in cd_hessian_cases(T)
            ls = makehessiansystem(make(Az), s)
            getb(ls) .= b
            n0 = DLS.RESCALES[]
            dampen!(ls, 0.0); solve!(ls, s)
            dampen!(ls, 1e16)
            @test solve!(ls, s) in (Success, Indefinite) && all(isfinite, getx(ls))
            T === Float32 && @test DLS.RESCALES[] > n0   # reached in Float32 only
            T === Float64 && @test DLS.RESCALES[] == n0
            # after the rescale the system keeps the new scale: an earlier damping of the same fill still solves
            dampen!(ls, 1e16); x1 = copy(getx(ls)); solve!(ls, s)
            @test getx(ls) == x1
            for λ0 in (1e-6, 1e-20)
                ls = makehessiansystem(make(Az), s)
                getb(ls) .= b
                dampen!(ls, λ0)
                @test solve!(ls, s) in (Success, Indefinite) && all(isfinite, getx(ls))
                dampen!(ls, 1e20 * λ0)
                @test solve!(ls, s) in (Success, Indefinite) && all(isfinite, getx(ls))
            end
            (s in (:auto, :ldl) || (s === :cholesky && getA(ls) isa Matrix)) && @test cd_allocs(ls, static(s)) == 0
        end
        for (make, s) in cd_jacobian_cases(T)
            ls = makejacobiansystem(make(Jz), CD_N, s)
            # Not met for the Float32 QR (single-precision.md, Status of the continuation): no entry overflows, so the
            # scale is not recomputed, but with the stored scale the damping rows of the two weak columns are about 1e12
            # times the others', and the rank test (relative to eps(Float32) times the largest |R_ii|) reads the damped
            # system as rank deficient: NotPositiveDefinite, observed, where λ = 1e16 and λ = 1 (after λ₀ = 1e-20) are
            # 1e16 and 1e20 times the damping the scale was computed at. Float64's rank test passes.
            qr32 = T === Float32 && s in (:auto, :qr)
            cd_fillj!(ls, Jz, r)
            dampen!(ls, 0.0); solve!(ls, s)
            dampen!(ls, 1e16)
            st = solve!(ls, s)
            qr32 ? (@test_broken st == Success) : (@test st == Success)
            @test all(isfinite, getx(ls))
            for λ0 in (1e-6, 1e-20)
                cd_fillj!(ls, Jz, r)
                dampen!(ls, λ0)
                @test solve!(ls, s) == Success && all(isfinite, getx(ls))
                dampen!(ls, 1e20 * λ0)
                st = solve!(ls, s)
                qr32 && λ0 == 1e-20 ? (@test_broken st == Success) : (@test st == Success)
                @test all(isfinite, getx(ls))
            end
            getJ(getA(ls)) isa Matrix && @test cd_allocs(ls, static(s)) == 0
        end
    end

    @testset "the scale is computed once per fill" begin
        ls = DLS.hessiansystem(copy(A64), :cholesky, true)
        getb(ls) .= b
        dampen!(ls, 0.0); solve!(ls, :cholesky); s0 = copy(ls.cs)
        dampen!(ls, 1e3 * max_diag(ls)); solve!(ls, :cholesky)
        @test ls.cs == s0   # reused for a later damping
        zero!(ls); getA(ls) .= A64; getb(ls) .= b
        solve!(ls, :cholesky)   # first solve of the new fill, at λ₀ = 1e3·max_diag
        @test ls.cs != s0 && ls.cs == DLS.conditioning_scale.(diag(A64) .+ 1e3 * maximum(diag(A64)), Float64)
        J = DLS.jacobiansystem(copy(J64), CD_N, :qr, true)
        cd_fillj!(J, J64, r); dampen!(J, 0.5); solve!(J, :qr); s1 = copy(getA(J).ws.cs)
        dampen!(J, 50.0); solve!(J, :qr)
        @test getA(J).ws.cs == s1 && getA(J).ws.nfactor == 1   # not refactorized per damping
        @test s1 == DLS.conditioning_scale.(vec(sum(abs2, J64; dims = 1)) .+ 0.5, Float64)
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
