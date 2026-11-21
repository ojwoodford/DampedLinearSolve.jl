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
sp_sparse_solvers() = SP_SPARSE32 ? (:auto, :cholesky, :ldl, :qr) : (:auto, :ldl)

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
        # The sparse system's SuiteSparse solvers, where this SparseArrays has them in Float64 only
        if !SP_SPARSE32
            for s in (:cholesky, :qr)
                err = try makehessiansystem(sparse(A32), s); nothing catch e; e end
                @test err isa ArgumentError && occursin("does not offer Float32", err.msg) && occursin(":auto and :ldl", err.msg)
            end
        end
        # A solver name no built-in system takes in Float32 (an extension's, before the extension offers it)
        err = try makehessiansystem(copy(A32), :not_a_solver); nothing catch e; e end
        @test err isa ArgumentError && occursin("does not offer Float32 for a dense Hessian system", err.msg)
        # The forms without a prototype stay Float64
        @test makehessiansystem(3, :auto) isa DLS.DenseLinearSystem{Float64}
        @test makehessiansystem(static(3), :auto) isa DLS.StaticLinearSystem{3, 9, Float64}
    end
end

