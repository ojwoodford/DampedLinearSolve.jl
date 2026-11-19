using DampedLinearSolve, LinearAlgebra, SparseArrays, Static, Random, Test
using DampedLinearSolve: dampen!, zero!, copy_structure, sharex, Success, NotPositiveDefinite, Indefinite, check_linear_system
const DLS = DampedLinearSolve
using DampedLinearSolve: getA, getb, getx, solve!   # explicit: Krylov, loaded by other test files, also exports solve!

# The normal equations of the dense Jacobian system (`:cholesky`, `:ldl`; jacobiannormalsystem.jl). The
# helpers `jacobian_system`'s counterpart, `reference_solve`, `conditioned_jacobian`, `allocations` and
# `JacobianFill` are those of jacobiansystem.jl, which runtests.jl includes first.

# A dense Jacobian system built for `linsolver`, holding J and r, with b = Jᵀr, ready to solve
function normal_system(J, r, linsolver)
    ls = makejacobiansystem(copy(J), size(J, 2), linsolver)
    getA(ls).r .= r
    mul!(getb(ls), J', r)
    DLS.markstale!(getA(ls))
    return ls
end

# The product the solve forms, the upper triangle of JᵀJ (the lower one zero), for a dense Hessian
# system given exactly the same matrix
syrk_upper(J) = LinearAlgebra.BLAS.syrk!('U', 'T', 1.0, J, 0.0, zeros(size(J, 2), size(J, 2)))

# Measured in functions: at testset top level Julia 1.10 boxes the captured variables
normal_damped(ls, s) = (dampen!(ls, 0.7); solve!(ls, s))
normal_scaled(ls, s, sc) = (dampen!(ls, 0.7, sc); solve!(ls, s))
normal_undamped(ls, s) = (dampen!(ls, 0.0); solve!(ls, s))
normal_relinearize(ls, s, w, x) = (zero!(ls); w(getb(ls), getA(ls), x); dampen!(ls, 0.2); solve!(ls, s))
nfactor(ls) = getA(ls).ws.nfactor
normal_mismatch_message(ls, s) = try solve!(ls, s); "" catch e; e isa ArgumentError ? e.msg : "" end

@testset "jacobiannormalsystem.jl" begin
    @testset "check_linear_system with :$s, for m = n, m > n and m < n" for s in (:cholesky, :ldl)
        for (m, n) in ((6, 6), (11, 6), (4, 6), (1, 1), (5, 1), (1, 3))
            @test check_linear_system(t -> makejacobiansystem(m, n, t); n, linsolvers=(s,)) === nothing
            @test check_linear_system(t -> makejacobiansystem(zeros(m, n), n, t); n, linsolvers=(s,)) === nothing
        end
    end

    @testset "makejacobiansystem builds the dense system, with no copy of [J r]" begin
        J = zeros(7, 3)
        for s in (:cholesky, :ldl, static(:cholesky), static(:ldl))
            ls = makejacobiansystem(J, 3, s)
            @test ls isa DLS.DenseJacobianLinearSystem{<:DLS.JacobianNormalWorkspace} && getA(ls).J === J
            @test size(getA(ls).ws.scratch.Jr) == (0, 1) && size(getA(ls).ws.scratch.S) == (0, 1)
            @test size(ls.solver_ws.hs.A) == (3, 3) && all(iszero, ls.solver_ws.hs.A)
        end
        @test makejacobiansystem(7, 3, :ldl).solver_ws.hs.solver_ws isa DLS.DenseLDLWorkspace
        @test makejacobiansystem(7, 3, :cholesky).solver_ws.hs.solver_ws === nothing
        # a static number of variables with m·n within the static system's size: still the dense system
        for s in (:cholesky, :ldl)
            ls = makejacobiansystem(8, static(6), s)
            @test 8 * 6 <= DLS.MAX_STATIC_JACOBIAN && ls isa DLS.DenseJacobianLinearSystem
            @test makejacobiansystem(zeros(2, 3), static(3), s) isa DLS.DenseJacobianLinearSystem
        end
        @test makejacobiansystem(8, static(6), :qr) isa DLS.StaticJacobianLinearSystem
    end

    @testset "Against QR and the extended-precision reference, :$s" for s in (:cholesky, :ldl)
        J = conditioned_jacobian(40, 7, 1e3; seed=11)
        r = J * collect(range(1.0, 2.0; length=7)) + 1e-2 * sin.(1:40)
        sc = collect(range(0.5, 3.0; length=7))
        ls = normal_system(J, r, s)
        qrs = normal_system(J, r, :qr)
        for (λ, d) in ((0.0, nothing), (1e-3, nothing), (0.5, nothing), (1e-3, sc), (2.0, sc))
            isnothing(d) ? (dampen!(ls, λ); dampen!(qrs, λ)) : (dampen!(ls, λ, d); dampen!(qrs, λ, d))
            @test solve!(ls, s) == Success
            @test solve!(qrs, :qr) == Success
            @test getx(ls) ≈ getx(qrs) rtol = 1e-8
            @test getx(ls) ≈ reference_solve(J, r, λ, d) rtol = 1e-8
            # J, r and b are left as filled
            @test getA(ls).J == J && getA(ls).r == r && getb(ls) == J' * r
        end
    end

    @testset "One product per linearization, :$s" for s in (:cholesky, :ldl)
        rng = Random.Xoshiro(12)
        J, r = randn(rng, 25, 5), randn(rng, 25)
        ls = normal_system(J, r, s)
        @test nfactor(ls) == 0
        for λ in (0.0, 1e-6, 0.1, 10.0)
            dampen!(ls, λ)
            @test solve!(ls, s) == Success
        end
        dampen!(ls, 0.1, fill(2.0, 5))
        @test solve!(ls, s) == Success
        @test nfactor(ls) == 1
        # zero! and the linearization wrapper (markstale!) each invalidate the product
        J2 = 2 .* J
        zero!(ls)
        JacobianFill((r_, J_, x) -> (r_ .= r; J_ .= J2; 0.0))(getb(ls), getA(ls), nothing)
        dampen!(ls, 0.1)
        @test solve!(ls, s) == Success && nfactor(ls) == 2
        @test getx(ls) ≈ reference_solve(J2, r, 0.1) rtol = 1e-8
        dampen!(ls, 1.0); solve!(ls, s)
        @test nfactor(ls) == 2
        DLS.markstale!(getA(ls))
        solve!(ls, s)
        @test nfactor(ls) == 3
    end

    @testset "copy_structure: alternate solves each give their own solution, :$s" for s in (:cholesky, :ldl)
        rng = Random.Xoshiro(13)
        J1, r1 = randn(rng, 14, 4), randn(rng, 14)
        J2, r2 = randn(rng, 14, 4), randn(rng, 14)
        ls = normal_system(J1, r1, s)
        trial = copy_structure(ls)
        @test trial.solver_ws === ls.solver_ws && getA(trial).ws.scratch === getA(ls).ws.scratch
        @test getA(trial).ws.token != getA(ls).ws.token
        getA(trial).J .= J2; getA(trial).r .= r2; mul!(getb(trial), J2', r2); DLS.markstale!(getA(trial))
        for λ in (0.0, 0.3)
            dampen!(ls, λ); dampen!(trial, λ)
            @test solve!(ls, s) == Success && isapprox(getx(ls), reference_solve(J1, r1, λ); rtol = 1e-8)
            x1 = copy(getx(ls))
            @test solve!(trial, s) == Success && isapprox(getx(trial), reference_solve(J2, r2, λ); rtol = 1e-8)
            @test solve!(ls, s) == Success && getx(ls) == x1
        end
        @test nfactor(ls) == 3 && nfactor(trial) == 2    # each formed again after the other formed its own
        # sharex keeps the workspace
        other = makejacobiansystem(14, 4, s)
        shared = sharex(other, ls)
        @test getx(shared) === getx(ls) && shared.solver_ws === other.solver_ws
    end

    @testset "m < n: undamped NotPositiveDefinite without a product, damped Success, :$s" for s in (:cholesky, :ldl)
        rng = Random.Xoshiro(14)
        J, r = randn(rng, 3, 5), randn(rng, 3)
        ls = normal_system(J, r, s)
        dampen!(ls, 0.0)
        fill!(getx(ls), 7.0)
        @test solve!(ls, s) == NotPositiveDefinite
        @test nfactor(ls) == 0 && getx(ls) == fill(7.0, 5)
        for λ in (1e-4, 1.0)
            dampen!(ls, λ)
            @test solve!(ls, s) == Success
            @test getx(ls) ≈ reference_solve(J, r, λ) rtol = 1e-8
        end
        # undamped again, with the product current: still not solved
        dampen!(ls, 0.0)
        @test solve!(ls, s) == NotPositiveDefinite && nfactor(ls) == 1
    end

    @testset "Rank-deficient J with m ≥ n: damped Success, undamped as the dense Hessian system, :$s" for s in (:cholesky, :ldl)
        rng = Random.Xoshiro(15)
        for (m, n) in ((12, 4), (30, 9), (5, 5), (200, 3))
            J = randn(rng, m, n)
            J[:, n] = J[:, 1]                       # a duplicated column
            r = randn(rng, m)
            ls = normal_system(J, r, s)
            dampen!(ls, 0.0)
            hs = makehessiansystem(syrk_upper(J), s)
            hs.b .= J' * r
            @test solve!(ls, s) == solve!(hs, s)
            dampen!(ls, 1e-6)
            @test solve!(ls, s) == Success
            @test getx(ls) ≈ reference_solve(J, r, 1e-6) rtol = 1e-6
        end
        # an all-zero J (as after zero!): undamped, the factorization fails and x is not written
        ls = normal_system(zeros(4, 2), ones(4), s)
        fill!(getx(ls), 5.0)
        @test solve!(ls, s) == NotPositiveDefinite && getx(ls) == fill(5.0, 2)
        dampen!(ls, 1.0)
        @test solve!(ls, s) == Success && getx(ls) == zeros(2)
    end

    @testset "Non-finite J or r: NotPositiveDefinite, x NaN, nothing formed, :$s" for s in (:cholesky, :ldl)
        J = randn(Random.Xoshiro(16), 6, 3); r = ones(6)
        for (Jbad, rbad) in ((setindex!(copy(J), NaN, 2, 2), r), (setindex!(copy(J), Inf, 6, 1), r),
                             (J, setindex!(copy(r), NaN, 3)), (J, setindex!(copy(r), -Inf, 1)))
            ls = normal_system(Jbad, rbad, s)
            for λ in (0.0, 1.0, 0.0)            # the second and third solves read the recorded state
                dampen!(ls, λ)
                fill!(getx(ls), 1.0)
                @test solve!(ls, s) == NotPositiveDefinite
                @test all(isnan, getx(ls))
                @test nfactor(ls) == 0
                @test all(iszero, ls.solver_ws.hs.A)
            end
        end
        # with fewer residuals than variables too
        ls = normal_system(fill(NaN, 2, 3), ones(2), s)
        @test solve!(ls, s) == NotPositiveDefinite && all(isnan, getx(ls)) && nfactor(ls) == 0
    end

    @testset ":ldl on a full-rank J is Success, never Indefinite" begin
        for seed in 1:5
            J = conditioned_jacobian(20, 6, 1e4; seed)
            r = randn(Random.Xoshiro(seed), 20)
            ls = normal_system(J, r, :ldl)
            for λ in (0.0, 1e-8, 1.0)
                dampen!(ls, λ)
                @test solve!(ls, :ldl) == Success
            end
        end
    end

    @testset "dampingfloor" begin
        for s in (:cholesky, :ldl)
            @test dampingfloor(makejacobiansystem(6, 3, s)) == eps(Float64)
            @test dampingfloor(makejacobiansystem(zeros(6, 3), 3, s)) == eps(Float64)
        end
        @test dampingfloor(makejacobiansystem(6, 3, :qr)) == eps(Float64)^2
        @test dampingfloor(makejacobiansystem(6, 3, :auto)) == eps(Float64)^2
    end

    @testset "A system is solved with the solver it was built for" begin
        J = randn(Random.Xoshiro(17), 6, 3); r = randn(Random.Xoshiro(18), 6)
        for (built, asked) in ((:cholesky, :ldl), (:ldl, :cholesky), (:cholesky, :qr), (:cholesky, :auto),
                               (:ldl, :qr), (:ldl, :auto), (:qr, :cholesky), (:qr, :ldl), (:auto, :cholesky), (:auto, :ldl))
            ls = normal_system(J, r, built)
            msg = normal_mismatch_message(ls, asked)
            builtname = built === :auto ? :qr : built   # a system built for :auto is built for QR
            @test occursin("linsolver = :$asked", msg) && occursin("for linsolver = :$builtname", msg)
        end
        # :lsmr without Krylov.jl loaded (or with it: then a mismatch, tested in lsmr.jl): an ArgumentError naming the solvers
        ls = normal_system(J, r, :cholesky)
        err = try solve!(ls, :pcg); nothing catch e; e end
        @test err isa ArgumentError && occursin(":auto, :qr, :cholesky or :ldl, or :lsmr with Krylov.jl loaded", err.msg)
    end

    @testset "check_jacobian_linsolver for each prototype kind and solver" begin
        Jd = zeros(4, 2)
        Js = sparse([1.0 0.0; 0.0 1.0; 1.0 1.0])
        for s in (:auto, :qr, :lsmr, :cholesky, :ldl)
            @test check_jacobian_linsolver(4, s) === nothing
            @test check_jacobian_linsolver(Int32(4), s) === nothing
            @test check_jacobian_linsolver(Jd, s) === nothing
        end
        for s in (:auto, :qr, :lsmr)
            @test check_jacobian_linsolver(Js, s) === nothing
        end
        for s in (:cholesky, :ldl)
            err = try check_jacobian_linsolver(Js, s); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin("linsolver = :$s is not supported by the Jacobian system of a SparseMatrixCSC{Float64, Int64}", err.msg)
            @test occursin("a Matrix{Float64} prototype or an integer m also takes :cholesky or :ldl", err.msg)
        end
        for (J, s) in ((Jd, :pardiso), (4, :cg), (true, :cholesky), (zeros(Float32, 3, 2), :ldl))
            @test_throws ArgumentError check_jacobian_linsolver(J, s)
        end
    end

    @testset "No allocation after warm-up, :$s" for s in (:cholesky, :ldl)
        J = randn(Random.Xoshiro(19), 50, 7); r = randn(Random.Xoshiro(20), 50)
        ls = normal_system(J, r, s)
        ss = static(s)
        sc = fill(1.5, 7)
        wrapper = JacobianFill((r_, J_, x) -> (copyto!(r_, x[1]); copyto!(J_, x[2]); 0.0))
        @test allocations(normal_damped, ls, ss) == 0
        @test allocations(normal_scaled, ls, ss, sc) == 0
        @test allocations(normal_undamped, ls, ss) == 0
        @test allocations(normal_relinearize, ls, ss, wrapper, (r, J)) == 0
        @test nfactor(ls) >= 3
    end
end
