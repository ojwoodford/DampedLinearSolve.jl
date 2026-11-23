using DampedLinearSolve, SparseArrays, StaticArrays, LinearAlgebra, Test, Static
const DLS = DampedLinearSolve

# A Float32 system: its matrix (and every copy and factorization of it) in Float32; b, x, the damping and every query in
# Float64. Solutions are compared with a Float64 dense solve of the Float64 matrix, to a tolerance in units of
# eps(Float32)·cond: narrowing the matrix and solving in Float32 each contribute about that.
const SP_N = 6
sp_matrix(n) = (M = [sin(1.7 * (i + j) + 0.3 * i * j) for i in 1:n, j in 1:n]; M = (M + M') / 2; M + Diagonal(vec(sum(abs, M; dims = 2)) .+ 1))
sp_rhs(n) = [cos(1.3 * i) for i in 1:n]
sp_tol(A) = 8 * size(A, 1) * eps(Float32) * cond(A)
sp_relerr(x, y) = norm(x - y) / norm(y)
sp_allocs(ls, s) = (solve!(ls, s); @allocated solve!(ls, s))

# The solvers that offer Float32 for each kind of Hessian system, on this Julia version
const SP_SPARSE32 = Float32 <: SparseArrays.CHOLMOD.VTypes
sp_sparse_solvers() = SP_SPARSE32 ? (:auto, :cholesky, :ldl) : (:auto, :ldl)

# The damped references: undamped, Levenberg λI and Marquardt λ·Diagonal(s)
const SP_DAMPINGS = ((0.0, nothing), (0.37, nothing), (2.9, [1.0 + i / SP_N for i in 1:SP_N]))
sp_damped(A, λ, s) = A + λ * (isnothing(s) ? I : Diagonal(s))

function sp_check_solves(ls, s, A64, b)
    for (λ, sc) in SP_DAMPINGS
        dampen!(ls, λ, sc)
        @test solve!(ls, s) == Success
        ref = sp_damped(A64, λ, sc) \ b
        @test sp_relerr(getx(ls), ref) <= sp_tol(sp_damped(A64, λ, sc))
    end
end

@testset "precision.jl: Hessian systems" begin
    A64, b = sp_matrix(SP_N), sp_rhs(SP_N)
    A32 = Float32.(A64)

    @testset "static" begin
        for s in (:auto, :cholesky, :ldl, :qr)
            for H in (MMatrix{SP_N, SP_N}(A32), SMatrix{SP_N, SP_N}(A32))
                ls = makehessiansystem(H, s)
                @test ls isa DLS.StaticLinearSystem{SP_N, SP_N^2, Float32}
                H isa MMatrix && @test getA(ls) === H
                @test eltype(getA(ls)) === Float32 && eltype(getb(ls)) === Float64 && eltype(getx(ls)) === Float64
                @test ls.lambda isa Float64
                getA(ls) .= A32
                getb(ls) .= b
                sp_check_solves(ls, static(s), A64, b)
                @test sp_allocs(ls, static(s)) == 0
                @test max_diag(ls) isa Float64 && max_diag(ls) == maximum(diag(A32))
            end
        end
        # copy_structure and sharex keep the element types
        ls = makehessiansystem(MMatrix{SP_N, SP_N}(A32), :auto)
        t = copy_structure(ls)
        @test t isa DLS.StaticLinearSystem{SP_N, SP_N^2, Float32} && eltype(getb(t)) === Float64
        @test getx(sharex(makehessiansystem(MMatrix{SP_N, SP_N}(A32), :auto), ls)) === getx(ls)
    end

    @testset "dense" begin
        for s in (:auto, :cholesky, :ldl, :qr)
            ls = makehessiansystem(copy(A32), s)
            @test ls isa DLS.DenseLinearSystem{Float32}
            @test eltype(getA(ls)) === Float32 && eltype(ls.A_scratch) === Float32 && eltype(ls.rhs) === Float32
            @test eltype(getb(ls)) === Float64 && eltype(getx(ls)) === Float64 && ls.lambda isa Float64
            s === :ldl && @test ls.solver_ws isa DLS.DenseLDLWorkspace{Float32}
            getb(ls) .= b
            sp_check_solves(ls, static(s), A64, b)
            # :qr (pivoted QR, through LinearAlgebra) allocates; every other dense solve does not
            s === :qr || @test sp_allocs(ls, static(s)) == 0
            @test max_diag(ls) isa Float64 && quadform(getA(ls), b) isa Float64
            @test quadform(getA(ls), b) ≈ dot(b, Float64.(A32) * b) rtol = 1e-12
            @test DLS.quadratic_model_change(ls, b) isa Float64
            t = copy_structure(ls)
            @test t isa typeof(ls) && getA(t) !== getA(ls) && getx(t) === getx(ls)
            @test getx(sharex(makehessiansystem(copy(A32), s), ls)) === getx(ls)
        end
        # Without a preallocated workspace, :ldl is Bunch–Kaufman through LinearAlgebra
        ls = DLS.DenseLinearSystem{Float32}(copy(A32), copy(b), zeros(SP_N))
        @test ls.solver_ws === nothing
        sp_check_solves(ls, static(:ldl), A64, b)
        # A dense prototype of another type is copied into a Matrix of its element type
        ls = makehessiansystem(Symmetric(A32), :auto)
        @test ls isa DLS.DenseLinearSystem{Float32} && getA(ls) == A32
        # The Float32 Bunch–Kaufman pivots tell an indefinite matrix as the Float64 ones do
        Ai = copy(A32); Ai[1, 1] = -Ai[1, 1]
        for ls in (makehessiansystem(copy(Ai), :ldl), DLS.DenseLinearSystem{Float32}(copy(Ai), copy(b), zeros(SP_N)))
            getb(ls) .= b
            @test solve!(ls, :ldl) == Indefinite
            @test sp_relerr(getx(ls), Float64.(Ai) \ b) <= sp_tol(Float64.(Ai))
        end
        ls = makehessiansystem(copy(Ai), :cholesky)
        @test solve!(ls, :cholesky) == NotPositiveDefinite
    end

    @testset "sparse" begin
        S32 = sparse(A32)
        for s in sp_sparse_solvers()
            ls = makehessiansystem(copy(S32), s)
            @test ls isa DLS.SparseLinearSystem{Float32}
            @test eltype(getA(ls)) === Float32 && eltype(ls.rhs) === Float32 && eltype(ls.sol) === Float32
            @test eltype(getb(ls)) === Float64 && eltype(getx(ls)) === Float64 && ls.lambda isa Float64
            s in (:auto, :ldl) && @test ls.solver_ws isa DLS.LDLFactorizations.LDLFactorization{Float32}
            s === :cholesky && @test ls.solver_ws isa SparseArrays.CHOLMOD.Factor{Float32}
            getb(ls) .= b
            A0 = copy(nonzeros(getA(ls)))
            sp_check_solves(ls, static(s), A64, b)
            @test nonzeros(getA(ls)) == A0   # the damping is restored exactly
            # LDLᵀ allocates nothing; CHOLMOD and sparse QR allocate inside SuiteSparse (rules.md 21)
            s in (:auto, :ldl) && @test sp_allocs(ls, static(s)) == 0
            @test max_diag(ls) isa Float64
            @test quadform(getA(ls), b) ≈ dot(b, Float64.(A32) * b) rtol = 1e-12
            t = copy_structure(ls)
            @test t isa typeof(ls) && getA(t) !== getA(ls) && getx(t) === getx(ls)
        end
    end

    @testset "decision 10: what is not offered is an ArgumentError when the system is built" begin
        # Another element type, from every prototype
        for H in (Float16.(A64), BigFloat.(A64), [1 0; 0 1], MMatrix{2, 2}(Float16[1 0; 0 1]), SMatrix{2, 2}(1, 0, 0, 1),
                  sparse(Float16.(A64)), Symmetric(Float16.(A64)))
            err = try makehessiansystem(H, :auto); nothing catch e; e end
            @test err isa ArgumentError && occursin("Float64 or Float32", err.msg)
        end
        # The sparse system's SuiteSparse solvers: CHOLMOD where this SparseArrays has it in Float64 only; SPQR always
        for s in (SP_SPARSE32 ? (:qr,) : (:cholesky, :qr))
            err = try makehessiansystem(sparse(A32), s); nothing catch e; e end
            @test err isa ArgumentError && occursin("does not offer Float32", err.msg) &&
                  occursin(SP_SPARSE32 ? ":auto, :cholesky and :ldl" : ":auto and :ldl", err.msg)
        end
        # A solver name no built-in system takes in Float32 (an extension's, before the extension offers it)
        err = try makehessiansystem(copy(A32), :not_a_solver); nothing catch e; e end
        @test err isa ArgumentError && occursin("does not offer Float32 for a dense Hessian system", err.msg)
        # The forms without a prototype stay Float64
        @test makehessiansystem(3, :auto) isa DLS.DenseLinearSystem{Float64}
        @test makehessiansystem(static(3), :auto) isa DLS.StaticLinearSystem{3, 9, Float64}
    end
end

# A Jacobian system in Float32: J (and [J r], R, JᵀJ and their factorizations) in Float32; r, b = Jᵀr, x and the
# queries in Float64. The reference is the Float64 least-squares solution of the Float64 J and r.
sp_jacobian(m, n) = [sin(0.9 * i + 1.1 * j + 0.2 * i * j) + (i == j ? 2.0 : 0.0) for i in 1:m, j in 1:n]
sp_residual(m) = [cos(0.7 * i) for i in 1:m]
function sp_check_jacobian_solves(ls, s, J64, r)
    n = size(J64, 2)
    for (λ, sc0) in SP_DAMPINGS
        sc = isnothing(sc0) ? nothing : sc0[1:n]
        dampen!(ls, λ, sc)
        @test solve!(ls, s) == Success
        A = J64' * J64 + λ * (isnothing(sc) ? I : Diagonal(sc))
        @test sp_relerr(getx(ls), A \ (J64' * r)) <= 8 * n * eps(Float32) * sqrt(cond(A)) * cond(J64)
    end
end
sp_fill!(ls, J32, r) = (getJ(getA(ls)) .= J32; getr(getA(ls)) .= r; gradient!(getb(ls), getA(ls)); markstale!(getA(ls)); ls)

@testset "precision.jl: Jacobian systems" begin
    m, n = 9, SP_N
    J64, r = sp_jacobian(m, n), sp_residual(m)
    J32 = Float32.(J64)

    @testset "dense" begin
        for s in (:auto, :qr, :cholesky, :ldl)
            ls = makejacobiansystem(zeros(Float32, m, n), n, s)
            @test ls isa DLS.DenseJacobianLinearSystem{<:Any, Float32}
            A = getA(ls)
            @test eltype(getJ(A)) === Float32 && eltype(getr(A)) === Float64 && eltype(getb(ls)) === Float64 && eltype(getx(ls)) === Float64
            if s in (:auto, :qr)
                @test eltype(A.ws.scratch.Jr) === Float32 && eltype(A.ws.scratch.S) === Float32 && eltype(A.ws.scratch.work) === Float32
                # SP-D10: no Float32 copy of r: the factored matrices are J S and [R; √λ D^½ S] alone, and the residual
                # column is carried in Float64
                @test size(A.ws.scratch.Jr) == (m, n) && size(A.ws.scratch.S) == (2n, n)
                @test eltype(A.ws.scratch.qv) === Float64 && length(A.ws.scratch.qv) == m && length(A.ws.scratch.qv2) == 2n
            else
                @test ls.solver_ws.hs isa DLS.DenseLinearSystem{Float32}
            end
            sp_fill!(ls, J32, r)
            @test getb(ls) ≈ Float64.(J32)' * r rtol = 1e-14   # the gradient, accumulated in Float64
            sp_check_jacobian_solves(ls, static(s), J64, r)
            dampen!(ls, 0.37)
            @test sp_allocs(ls, static(s)) == 0
            @test max_diag(ls) isa Float64 && max_diag(ls) == maximum(sum(abs2, Float64.(J32); dims = 1))
            @test quadform(A, r[1:n]) ≈ sum(abs2, Float64.(J32) * r[1:n]) rtol = 1e-13
            t = copy_structure(ls)
            @test t isa typeof(ls) && getJ(getA(t)) == getJ(A) && getJ(getA(t)) !== getJ(A)
        end
        # SP-D10: the step is the least-squares solution for r as computed, not as rounded to Float32: r and the same r
        # rounded to Float32 give different steps (before SP-D10 r was narrowed, and they were identical)
        for s in (:auto, :qr), λ in (0.0, 0.37)
            l1, l2 = makejacobiansystem(copy(J32), n, s), makejacobiansystem(copy(J32), n, s)
            sp_fill!(l1, J32, r); sp_fill!(l2, J32, Float64.(Float32.(r)))
            dampen!(l1, λ); dampen!(l2, λ)
            @test solve!(l1, s) == Success && solve!(l2, s) == Success
            @test r != Float64.(Float32.(r)) && getx(l1) != getx(l2)
        end
        # In Float64 the residual column is the factored matrix's last column, as before SP-D10: the same operations
        ls = makejacobiansystem(copy(J64), n, :qr)
        @test size(getA(ls).ws.scratch.Jr) == (m, n + 1) && isempty(getA(ls).ws.scratch.qv)
        # Fewer residuals than variables, undamped: not solved as definite, in Float32 too
        ls = makejacobiansystem(zeros(Float32, 3, n), n, :qr)
        sp_fill!(ls, Float32.(sp_jacobian(3, n)), sp_residual(3))
        dampen!(ls, 0.0)
        @test solve!(ls, :qr) == NotPositiveDefinite
        # A NaN in J is reported, and x set to NaN
        ls = makejacobiansystem(zeros(Float32, m, n), n, :auto)
        J = copy(J32); J[2, 3] = NaN32
        sp_fill!(ls, J, r)
        @test solve!(ls, :auto) == NotPositiveDefinite && all(isnan, getx(ls))
        @test !isfinitesystem(ls)
    end

    @testset "static" begin
        ms, ns = 6, 4
        Js, rs = Float32.(sp_jacobian(ms, ns)), sp_residual(ms)
        for s in (:auto, :qr)
            ls = makejacobiansystem(zeros(Float32, ms, ns), static(ns), s)
            @test ls isa DLS.StaticJacobianLinearSystem{ms, ns, ms * ns, ns, ns * ns, Float32}
            @test eltype(getJ(getA(ls))) === Float32 && eltype(ls.R) === Float32 && eltype(getx(ls)) === Float64
            @test eltype(ls.q) === Float64   # SP-D10: Qᵀr in Float64
            sp_fill!(ls, Js, rs)
            sp_check_jacobian_solves(ls, static(s), sp_jacobian(ms, ns), rs)
            @test sp_allocs(ls, static(s)) == 0
            t = copy_structure(ls)
            @test t isa typeof(ls)
            # the step is the solution for r as computed (SP-D10)
            l2 = makejacobiansystem(zeros(Float32, ms, ns), static(ns), s)
            sp_fill!(l2, Js, Float64.(Float32.(rs)))
            for λ in (0.0, 0.37)
                dampen!(ls, λ); dampen!(l2, λ); solve!(ls, s); solve!(l2, s)
                @test getx(ls) != getx(l2)
            end
        end
    end

    @testset "sparse" begin
        Jsp = sparse(J32)
        # SPQR is Float64 only, on every Julia version (precision.jl)
        for s in (:auto, :qr)
            err = try makejacobiansystem(copy(Jsp), n, s); nothing catch e; e end
            @test err isa ArgumentError && occursin("does not offer Float32 for a sparse Jacobian system", err.msg) && occursin("SPQR", err.msg)
        end
        @test max_diag(DLS.SparseJacobianLinearSystem(copy(Jsp); linsolver = :not_a_solver)) isa Float64
    end

    @testset "decision 10, and other element types" begin
        for J in (Float16.(J64), BigFloat.(J64))
            err = try makejacobiansystem(J, n, :auto); nothing catch e; e end
            @test err isa ArgumentError && occursin("Float64 or Float32", err.msg)
            err = try checkjacobian(J); nothing catch e; e end
            @test err isa ArgumentError && occursin("Float64 or Float32", err.msg)
        end
        @test checkjacobian(J32) === nothing && checkjacobian(sparse(J32)) === nothing
        err = try makejacobiansystem(zeros(Float32, m, n), n, :not_a_solver); nothing catch e; e end
        @test err isa ArgumentError && occursin("does not offer Float32 for a dense Jacobian system", err.msg)
        @test makejacobiansystem(m, n, :auto) isa DLS.DenseJacobianLinearSystem{<:Any, Float64}
    end
end

@testset "precision.jl: the element type and the constants that depend on it" begin
    A = sp_matrix(4); J = sp_jacobian(6, 4)
    for T in (Float64, Float32)
        for ls in (makehessiansystem(T.(A), :auto), makehessiansystem(sparse(T.(A)), :auto), makehessiansystem(MMatrix{4, 4}(T.(A)), :auto),
                   makejacobiansystem(T.(J), 4, :qr), makejacobiansystem(T.(J), 4, :cholesky), makejacobiansystem(T.(J), static(4), :qr))
            @test systemeltype(ls) === T
            @test systemeltype(copy_structure(ls)) === T
        end
        # dampingfloor: eps(T) for a factorization of A + λD, eps(T)² for QR of J; always a Float64
        @test dampingfloor(makehessiansystem(T.(A), :ldl)) === Float64(eps(T))
        @test dampingfloor(makehessiansystem(sparse(T.(A)), :ldl)) === Float64(eps(T))
        @test dampingfloor(makejacobiansystem(T.(J), 4, :cholesky)) === Float64(eps(T))
        @test dampingfloor(makejacobiansystem(T.(J), 4, :qr)) === Float64(eps(T))^2
        @test dampingfloor(makejacobiansystem(T.(J), static(4), :qr)) === Float64(eps(T))^2
    end
    @test systemeltype(makehessiansystem(3, :auto)) === Float64 && systemeltype(makejacobiansystem(6, 3, :qr)) === Float64
    @test systemeltype(nothing) === Float64   # a system that does not say
    @test systemeltype(DLS.SparseJacobianLinearSystem(sparse(Float32.(J)); linsolver = :not_a_solver)) === Float32
    # The indefiniteness tolerance: 1e-6 in Float64, eps(Float32)^(3/8) in Float32 (a starting value)
    @test DLS.indefinite_rtol(Float64) === 1e-6
    @test DLS.indefinite_rtol(Float32) ≈ Float64(eps(Float32))^(3 / 8)
    @test DLS.precision_literal(1e-8, Float32) ≈ Float64(eps(Float32))^(1 / 2)
    # A pivot between the two tolerances: negative to Float64's test, rounding to Float32's
    Ai = Float32[1 0; 0 -1e-4]
    bk = bunchkaufman(Symmetric(Ai))
    @test !DLS.is_indefinite(bk) && DLS.is_indefinite(bunchkaufman(Symmetric(Float64.(Ai))))
    @test DLS.is_indefinite(bk, 1e-6)
    # The rank test of the Jacobian QR: relative to eps of R's element type
    R = Float32[1 0; 0 1e-7]   # below 2·eps(Float32), above 2·eps(Float64)
    @test !DLS.fullrank_upper(R, 2) && DLS.fullrank_upper(Float64.(R), 2)
end
