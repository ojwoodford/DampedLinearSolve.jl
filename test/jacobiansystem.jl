using DampedLinearSolve, LinearAlgebra, Static, Random, Test
using DampedLinearSolve: dampen!, zero!, copy_structure, sharex, quadratic_model_change, Success, NotPositiveDefinite, check_linear_system
const DLS = DampedLinearSolve
using DampedLinearSolve: getA, getb, getx, solve!   # explicit: Krylov, loaded by other test files, also exports solve!

# How an optimizer fills a Jacobian system: the linearization writes J and r, then b = Jᵀr is formed and
# what was computed from the old J and r is invalidated
include("jacobianfill.jl")

# J2, J3: DenseJacobianLinearSystem, A = JᵀJ held implicitly, solved by QR of J

# A Jacobian system holding J and r, with b = Jᵀr, ready to solve
function jacobian_system(J, r)
    ls = DLS.DenseJacobianLinearSystem(copy(J))
    ls.A.r .= r
    mul!(ls.b, J', r)
    DLS.markstale!(ls.A)
    return ls
end

# The damped normal-equations solution, in extended precision, as the reference
function reference_solve(J, r, λ, s=nothing)
    Jb, rb = big.(J), big.(r)
    D = isnothing(s) ? Matrix{BigFloat}(I, size(J, 2), size(J, 2)) : Diagonal(big.(s))
    return Float64.((Jb' * Jb + λ * D) \ (Jb' * rb))
end

# A Jacobian with condition number near `c`, from fixed orthogonal factors
function conditioned_jacobian(m, n, c; seed=1)
    rng = Random.Xoshiro(seed)
    U = Matrix(qr(randn(rng, m, n)).Q)[:, 1:n]
    V = Matrix(qr(randn(rng, n, n)).Q)
    return U * Diagonal(exp10.(range(0, -log10(c); length=n))) * V'
end

# Measured in functions: at testset top level Julia 1.10 boxes the captured variables
jacobian_damped(ls) = (dampen!(ls, 0.7); solve!(ls, static(:auto)))
jacobian_scaled(ls, s) = (dampen!(ls, 0.7, s); solve!(ls, static(:qr)))
jacobian_undamped(ls) = (DLS.markstale!(getA(ls)); dampen!(ls, 0.0); solve!(ls, static(:auto)))
jacobian_relinearize(ls, w, x) = (zero!(ls); w(getb(ls), getA(ls), x))
# Measured through a function barrier: at testset scope Julia 1.10 boxes the Float64 result (16 bytes)
jacobian_quadform_allocations(A, v) = @allocated DLS.quadform(A, v)
jacobian_model(ls, v) = DLS.quadform(getA(ls), v) + quadratic_model_change(ls, v) + DLS.max_diag(ls)
allocations(f::F, args::Vararg{Any, N}) where {F, N} = (f(args...); @allocated f(args...))   # specialized: 1.10 does not specialize a plain vararg here

@testset "jacobiansystem.jl" begin
    @testset "check_linear_system for m = n, m > n and m < n" begin
        for (m, n) in ((6, 6), (11, 6), (4, 6), (1, 1), (5, 1), (1, 3))
            @test check_linear_system(s -> DLS.DenseJacobianLinearSystem(m, n); n, linsolvers=(:auto, :qr)) === nothing
            # also built from an objective, with a prototype and with an integer m
            @test check_linear_system(s -> makejacobiansystem(zeros(m, n), n, s); n, linsolvers=(:auto,)) === nothing
            @test check_linear_system(s -> makejacobiansystem(m, n, s); n, linsolvers=(:qr,)) === nothing
        end
        @test DLS.conformance_traits(DLS.getA(DLS.DenseJacobianLinearSystem(4, 6))) == (exact = false, maxrank = 4, indefinite = false, accumulates = false)
    end

    @testset "makejacobiansystem writes into a Matrix prototype, or builds its own J" begin
        J = zeros(7, 3)
        ls = makejacobiansystem(J, 3, :auto)
        @test ls isa DLS.DenseJacobianLinearSystem && DLS.getA(ls).J === J
        ls = makejacobiansystem(7, 3, :auto)
        @test size(DLS.getA(ls).J) == (7, 3) && length(DLS.getA(ls).r) == 7 && length(getb(ls)) == 3 && length(getx(ls)) == 3
    end

    @testset "solve! against (JᵀJ + λD) \\ Jᵀr, well and ill conditioned" begin
        for (m, n, c) in ((20, 5, 10.0), (30, 8, 1e10), (8, 8, 1e6))
            J = conditioned_jacobian(m, n, c)
            r = J * collect(range(1.0, 2.0; length=n)) + 1e-3 * sin.(1:m)   # nonzero residual at the solution
            ls = jacobian_system(J, r)
            g = J' * r
            s = collect(range(0.5, 3.0; length=n))
            for λ in (0.0, 1e-8, 1e-3, 0.5, 30.0)
                for sc in (nothing, s)
                    λ == 0 && !isnothing(sc) && continue
                    isnothing(sc) ? dampen!(ls, λ) : dampen!(ls, λ, sc)
                    @test solve!(ls, static(:auto)) == Success
                    xref = reference_solve(J, r, λ, sc)
                    # backward-stable QR: error bounded by cond(J)·eps for the consistent part, and
                    # cond(J)²·eps times the relative residual for the rest
                    @test norm(getx(ls) - xref) <= 1e-12 * c * (1 + 1e-3 * c) * norm(xref)
                    # J, r and g are left as written
                    @test DLS.getA(ls).J == J && DLS.getA(ls).r == r && getb(ls) == g
                end
            end
            # every solve after the first reused the one QR of [J r]
            @test DLS.getA(ls).ws.nfactor == 1
            @test solve!(ls, :qr) == Success && DLS.getA(ls).ws.nfactor == 1
            # writing J again (zero! or the linearization wrapper) refactors
            DLS.zero!(ls)
            JacobianFill((r_, J_, x) -> (r_ .= r; J_ .= J; 0.0))(getb(ls), DLS.getA(ls), nothing)
            @test getb(ls) ≈ g
            dampen!(ls, 0.5); solve!(ls, :auto)
            @test DLS.getA(ls).ws.nfactor == 2
        end
    end

    @testset "Accuracy on an ill-conditioned Jacobian, against the normal equations" begin
        # cond(J) ≈ 1e10, consistent: the normal equations (cond 1e20) cannot resolve it, QR can
        J = conditioned_jacobian(40, 6, 1e10; seed=3)
        xtrue = collect(1.0:6.0)
        r = J * xtrue
        ls = jacobian_system(J, r)
        dampen!(ls, 0.0)
        @test solve!(ls, :auto) == Success
        err_qr = norm(getx(ls) - xtrue) / norm(xtrue)
        dense = DLS.DenseLinearSystem{Float64}(J' * J, J' * r, zeros(6))
        status = solve!(dense, :cholesky)
        err_ne = status == Success ? norm(dense.x - xtrue) / norm(xtrue) : Inf
        @test err_qr < 1e-4
        @test err_qr < 1e-3 * err_ne
    end

    @testset "Rank deficiency: undamped is NotPositiveDefinite, damped is Success" begin
        rng = Random.Xoshiro(7)
        for (m, n) in ((12, 4), (30, 9), (5, 5), (200, 3))
            J = randn(rng, m, n)
            J[:, n] = J[:, 1]                       # a duplicated column
            r = randn(rng, m)
            ls = jacobian_system(J, r)
            dampen!(ls, 0.0)
            @test solve!(ls, :auto) == NotPositiveDefinite
            dampen!(ls, 1e-6)
            @test solve!(ls, :auto) == Success
            @test norm(getx(ls) - reference_solve(J, r, 1e-6)) <= 1e-6 * norm(reference_solve(J, r, 1e-6))
            @test DLS.getA(ls).J == J && DLS.getA(ls).r == r
        end
        # m < n: rank deficient by construction
        J = randn(rng, 3, 5); r = randn(rng, 3)
        ls = jacobian_system(J, r)
        dampen!(ls, 0.0)
        @test solve!(ls, :auto) == NotPositiveDefinite
        for λ in (1e-4, 1.0)
            dampen!(ls, λ)
            @test solve!(ls, :auto) == Success
            @test getx(ls) ≈ reference_solve(J, r, λ) rtol=1e-8
        end
        # an all-zero J (as after zero!)
        ls = jacobian_system(zeros(4, 2), ones(4))
        @test solve!(ls, :auto) == NotPositiveDefinite
        dampen!(ls, 1.0)
        @test solve!(ls, :auto) == Success && getx(ls) == zeros(2)
    end

    @testset "Non-finite input gives a non-Success status and a non-finite x" begin
        J = randn(Random.Xoshiro(2), 6, 3); r = ones(6)
        for (Jbad, rbad) in ((setindex!(copy(J), NaN, 2, 2), r), (setindex!(copy(J), Inf, 6, 1), r),
                             (J, setindex!(copy(r), NaN, 3)), (J, setindex!(copy(r), -Inf, 1)))
            ls = jacobian_system(Jbad, rbad)
            for λ in (0.0, 1.0)
                dampen!(ls, λ)
                status = solve!(ls, :auto)
                @test status != Success
                @test !all(isfinite, getx(ls))
            end
        end
        ls = jacobian_system(setindex!(copy(J), NaN, 1, 3), r)
        @test isnan(DLS.max_diag(ls))
        @test DLS.max_diag(jacobian_system(J, r)) == maximum(j -> sum(abs2, J[:, j]), 1:3)
    end

    @testset "The rest of the interface" begin
        J = randn(Random.Xoshiro(4), 9, 4); r = randn(Random.Xoshiro(5), 9)
        ls = jacobian_system(J, r)
        A = DLS.getA(ls)
        v = randn(4)
        @test DLS.quadform(A, v) ≈ sum(abs2, J * v)
        @test (@inferred DLS.quadform(A, v)) isa Float64
        @test jacobian_quadform_allocations(A, v) == 0
        @test quadratic_model_change(ls, v) ≈ dot(J' * r, v) + 0.5 * sum(abs2, J * v)
        @test size(A) == (4, 4)
        trial = copy_structure(ls)
        @test getx(trial) === getx(ls)
        @test DLS.getA(trial).J !== A.J && DLS.getA(trial).r !== A.r && getb(trial) !== getb(ls) && DLS.getA(trial).ws !== A.ws
        other = DLS.DenseJacobianLinearSystem(9, 4)
        shared = sharex(other, ls)
        @test getx(shared) === getx(ls) && DLS.getA(shared) === DLS.getA(other) && getb(shared) === getb(other)
        @test sharex(shared, ls) === shared
        # zero! leaves J and r as resjac! wrote them (Z1), zeroes b, and invalidates the caches
        A.ws.stale = false; A.ws.normsstale = false
        zero!(ls)
        @test A.J == J && A.r == r && all(iszero, getb(ls)) && A.ws.stale && A.ws.normsstale
        @test_throws ArgumentError solve!(ls, :cholesky)
        @test_throws ArgumentError solve!(ls, :ldl)
    end

    @testset "J v and Jᵀ r are plain loops that equal mul! to rounding (P4)" begin
        for (m, n) in ((40, 6), (6, 6), (40, 1), (1, 1), (3, 7))   # more rows than columns, square, one column, 1×1, wide
            rng = Random.Xoshiro(m + 100n)
            J = randn(rng, m, n); r = randn(rng, m); v = randn(rng, n)
            ls = jacobian_system(J, r)
            A = getA(ls)
            Jv = mul!(zeros(m), J, v)
            @test DLS.quadform(A, v) ≈ dot(Jv, Jv) rtol = 4eps()
            @test A.Jv ≈ Jv rtol = 4eps()
            @test jacobian_quadform_allocations(A, v) == 0
            g = fill(NaN, n)
            wrapper = JacobianFill((r_, J_, x) -> (copyto!(r_, x[1]); copyto!(J_, x[2]); 0.0))
            wrapper(g, A, (r, J))
            @test g ≈ mul!(zeros(n), transpose(J), r) rtol = 4eps()
            @test allocations(jacobian_relinearize, ls, wrapper, (r, J)) == 0
        end
    end

    @testset "The LAPACK workspace query allocates no matrix (Q2)" begin
        # values from the query before Q2, which passed an m×n matrix and a tau vector
        sizes = [(1, 1), (1, 3), (3, 1), (5, 5), (10, 3), (100, 7), (1000, 50), (10_000, 100), (37, 201), (2, 9)]
        before = [32, 96, 32, 160, 96, 224, 1600, 3200, 6432, 288]
        @test [DLS.geqrf_worksize(m, n) for (m, n) in sizes] == before
        @test DLS.geqrf_worksize(0, 5) == 1 && DLS.geqrf_worksize(5, 0) == 1
        DLS.geqrf_worksize(10_000, 100)
        @test (@allocated DLS.geqrf_worksize(10_000, 100)) < 500
        # and the system's construction allocates its own arrays, not two more of the problem's size
        m, n = 20_000, 50
        DLS.DenseJacobianLinearSystem(m, n)
        @test (@allocated DLS.DenseJacobianLinearSystem(m, n)) < 3.2 * 8 * m * (n + 1)
    end

    @testset "copy_structure shares the factorization scratch (Q3)" begin
        rng = Random.Xoshiro(41)
        J1, r1 = randn(rng, 12, 4), randn(rng, 12)
        J2, r2 = randn(rng, 12, 4), randn(rng, 12)
        ls = jacobian_system(J1, r1)
        trial = copy_structure(ls)
        @test getA(trial).ws.scratch === getA(ls).ws.scratch
        @test getA(trial).ws !== getA(ls).ws && getA(trial).ws.token != getA(ls).ws.token
        @test getA(trial).ws.colnorm2 !== getA(ls).ws.colnorm2
        getA(trial).J .= J2; getA(trial).r .= r2; mul!(getb(trial), J2', r2); DLS.markstale!(getA(trial))
        # solve one, then the other, then the first again: each refactors only when the scratch
        # holds the other's factorization, and every solution is that of its own J
        for λ in (0.0, 0.3)
            dampen!(ls, λ); dampen!(trial, λ)
            @test solve!(ls, :auto) == Success && getx(ls) ≈ reference_solve(J1, r1, λ)
            x1 = copy(getx(ls))
            @test solve!(trial, :auto) == Success && getx(trial) ≈ reference_solve(J2, r2, λ)
            @test solve!(ls, :auto) == Success && getx(ls) == x1
        end
        nf1, nf2 = getA(ls).ws.nfactor, getA(trial).ws.nfactor
        @test nf1 == 3 && nf2 == 2     # ls: first solve, after trial (λ = 0), after trial (λ = 0.3); trial: once per λ
        # repeated solves of the same system reuse its factorization
        dampen!(ls, 0.7); solve!(ls, :auto); dampen!(ls, 0.0); solve!(ls, :auto)
        @test getA(ls).ws.nfactor == nf1
        # the column norms are each system's own
        @test DLS.max_diag(ls) ≈ maximum(sum(abs2, J1; dims=1)) && DLS.max_diag(trial) ≈ maximum(sum(abs2, J2; dims=1))
    end


    @testset "No allocation after warm-up" begin
        J = randn(Random.Xoshiro(6), 50, 7); r = randn(Random.Xoshiro(8), 50)
        ls = jacobian_system(J, r)
        wrapper = JacobianFill((r_, J_, x) -> (copyto!(r_, x[1]); copyto!(J_, x[2]); 0.0))
        s = ones(7); v = randn(7); x = (r, J)
        @test allocations(jacobian_damped, ls) == 0
        @test allocations(jacobian_scaled, ls, s) == 0
        @test allocations(jacobian_undamped, ls) == 0
        @test allocations(jacobian_relinearize, ls, wrapper, x) == 0
        @test allocations(jacobian_model, ls, v) == 0
    end
end

# Paths the solve-level tests moved from SecondOrderOptimization did not reach on their own (rules.md 28)
@testset "jacobiansystem.jl: the remaining branches" begin
    # J v for a J with no columns is the zero vector, written without reading v
    y = fill(NaN, 3)
    @test DLS.jmul!(y, zeros(3, 0), Float64[]) == zeros(3)

    # The convenience constructors of the workspace and the Hessian storage
    ws = DLS.JacobianQRWorkspace(5, 2)
    @test ws isa DLS.JacobianQRWorkspace && length(ws.colnorm2) == 2 && ws.stale && ws.normsstale
    J = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    A = DLS.JacobianStorage(J)
    @test getJ(A) === J && getr(A) == zeros(3)
    @test size(A) == (2, 2) && size(A, 1) == 2 && size(A, 2) == 2

    # isfinitesystem looks at both J and r
    ls = makejacobiansystem(copy(J), 2, :auto)
    @test isfinitesystem(ls)
    getr(getA(ls))[2] = NaN
    @test !isfinitesystem(ls)
    getr(getA(ls))[2] = 0.0
    getJ(getA(ls))[1, 2] = Inf
    @test !isfinitesystem(ls)

    # The diagonal of JᵀJ is the squared column norms of J
    ls = makejacobiansystem(copy(J), 2, :auto)
    d = zeros(2)
    @test getdiag!(d, ls) === d
    @test d ≈ diag(J' * J)

    # The Jacobian prototype checks, one per kind of prototype
    @test checkjacobian(3) === nothing
    @test checkjacobian(J) === nothing
    @test checkjacobian(DLS.SparseArrays.sparse(J)) === nothing
    for (bad, text) in ((0, "at least 1"), (zeros(0, 2), "has no rows"), (DLS.SparseArrays.spzeros(0, 2), "has no rows"),
                        (DLS.SparseArrays.sparse(Int32[1], Int32[1], [1.0]), "sparse Jacobian prototype of type"),
                        (zeros(Float16, 3, 2), "must be Float64 or Float32"), (true, "type Bool is not supported"),
                        ("J", "type String is not supported"))
        err = try checkjacobian(bad); nothing catch e; e end
        @test err isa ArgumentError && occursin(text, err.msg)
    end

    # The prototype's size, and the solvers a Jacobian system accepts
    @test jacobian_nrows(4) == 4 && jacobian_nrows(Int32(4)) === 4 && jacobian_nrows(J) == 3
    @test jacobian_ncols(J) == 2 && jacobian_ncols(4) === nothing
    for s in (:auto, :qr, :lsmr)
        @test check_jacobian_linsolver(J, s) === nothing
    end
    # the normal-equations solvers for a dense prototype only (jacobiannormalsystem.jl)
    err = try check_jacobian_linsolver(DLS.SparseArrays.sparse(J), :cholesky); nothing catch e; e end
    @test err isa ArgumentError && occursin("linsolver = :cholesky is not supported", err.msg)
end
