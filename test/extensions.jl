!in(@__DIR__, LOAD_PATH) && push!(LOAD_PATH, @__DIR__)

using DampedLinearSolve, SparseArrays, LinearAlgebra, Test, Static
using DampedLinearSolve: dampen!, Success, NotPositiveDefinite, Indefinite

# P2b: a system solved with a solver it was not built for is an internal inconsistency (no user
# can reach it: a workspace reused with another linsolver is rebuilt). The message says so and
# names the solver asked for and the one the system was built for.
function mismatch_message(ls, asked, builtfor)
    err = try DampedLinearSolve.solve!(ls, static(asked)); nothing catch e; e end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("Internal inconsistency", msg)
    @test occursin("asked for linsolver = :$asked", msg)
    @test occursin("built by makehessiansystem or makejacobiansystem for linsolver = $builtfor", msg)
    return msg
end

@testset "extensions.jl" begin
    # 1. Validation before loading extensions
    # A2: there are no Schur solver symbols (the type of H selects elimination)
    for s in (:schur, :dense_schur, :sparse_schur)
        @test !DampedLinearSolve.validatelinsolver(s)
    end

    # 2. KrylovExt testing
    using Krylov

    @test DampedLinearSolve.validatelinsolver(:cg)
    @test DampedLinearSolve.validatelinsolver(:minres)
    # M10: LSMR is for a tall system: on the Hessian it would work with H'H, squaring its condition
    # number again (test/lsmr.jl)
    @test DampedLinearSolve.validatelinsolver(:lsmr)
    # X2: :krylov was an alias of :cg and is gone; it fails validation like any unknown solver
    @test !DampedLinearSolve.validatelinsolver(:krylov)
    @test_throws ArgumentError DampedLinearSolve.checklinsolver(:krylov)

    # Quadratic test problem: f(x) = 0.5 * x' A x - b' x
    A_spd = sparse([5.0 1.0 0.0; 1.0 6.0 1.0; 0.0 1.0 7.0])
    b_vec = [1.0, 2.0, 3.0]
    x_exact = A_spd \ b_vec

    quad_cost(x) = 0.5 * dot(x, A_spd * x) - dot(b_vec, x)
    function quad_fdf!(g, H, x)
        g .= A_spd * x .- b_vec
        H .= A_spd
        return 0.5 * dot(x, A_spd * x) - dot(b_vec, x)
    end


    # B1: a sparse system built for another solver throws an informative error rather than
    # trying to replace the `const` workspace field
    for solver in (:cg, :minres)
        ls_wrong = DampedLinearSolve.makehessiansystem(copy(A_spd), static(:auto))
        ls_wrong.b .= b_vec
        @test_throws ArgumentError DampedLinearSolve.solve!(ls_wrong, static(solver))
        mismatch_message(ls_wrong, solver, ":auto or :ldl")
        ls_right = DampedLinearSolve.makehessiansystem(copy(A_spd), static(solver))
        ls_right.b .= b_vec
        @test DampedLinearSolve.solve!(ls_right, static(solver)) == Success
        @test isapprox(ls_right.x, x_exact; rtol=1e-5)
    end
    # P2b: each workspace type names the solver it was built for
    for (built, asked, text) in ((:cg, :minres, ":cg"), (:minres, :cg, ":minres"),
                                 (:cholesky, :cg, ":cholesky"), (:qr, :minres, ":qr"))
        ls_other = DampedLinearSolve.makehessiansystem(copy(A_spd), static(built))
        mismatch_message(ls_other, asked, text)
    end
    ls_odd = DampedLinearSolve.SparseLinearSystem(copy(A_spd), zeros(3), zeros(3), [1, 4, 7], :not_a_workspace)
    mismatch_message(ls_odd, :cg, "an unknown solver (workspace type Symbol)")

    # M11: sparse CG and MINRES are Jacobi preconditioned, which helps on a badly scaled matrix
    n_sc = 400
    T_sc = spdiagm(-1 => fill(-1.0, n_sc - 1), 0 => fill(2.1, n_sc), 1 => fill(-1.0, n_sc - 1))
    D_sc = Diagonal(10.0 .^ range(-3, 3; length=n_sc))
    A_sc = sparse(D_sc * T_sc * D_sc)
    b_sc = A_sc * ones(n_sc)
    for solver in (:cg, :minres)
        ls_sc = DampedLinearSolve.makehessiansystem(copy(A_sc), static(solver))
        ls_sc.b .= b_sc
        @test DampedLinearSolve.solve!(ls_sc, static(solver)) == Success
        @test norm(ls_sc.x .- 1) <= 1e-5 * sqrt(n_sc)
        # Unpreconditioned, CG doesn't converge and MINRES stops on a residual dominated by the
        # large rows, both far from the solution
        ws_plain = solver === :cg ? Krylov.CgWorkspace(A_sc, b_sc) : Krylov.MinresWorkspace(A_sc, b_sc)
        solver === :cg ? Krylov.cg!(ws_plain, A_sc, b_sc) : Krylov.minres!(ws_plain, A_sc, b_sc)
        @test norm(ws_plain.x .- 1) > 1e-2 * sqrt(n_sc)
        sparse_krylov_allocs(ls, s) = @allocated DampedLinearSolve.solve!(ls, s)
        sparse_krylov_allocs(ls_sc, static(solver))
        @test sparse_krylov_allocs(ls_sc, static(solver)) == 0
    end

    # Test Krylov on DenseLinearSystem
    A_dense = Matrix(A_spd)
    quad_dense_cost(x) = 0.5 * dot(x, A_dense * x) - dot(b_vec, x)
    function quad_dense_fdf!(g, H, x)
        g .= A_dense * x .- b_vec
        H .= A_dense
        return 0.5 * dot(x, A_dense * x) - dot(b_vec, x)
    end

    for solver in (:cg, :minres)
        # M9: dense Krylov solves use a preallocated workspace
        ls_dk = DampedLinearSolve.makehessiansystem(copy(A_dense), static(solver))
        @test ls_dk.solver_ws isa Krylov.KrylovWorkspace
        ls_dk.b .= b_vec
        @test DampedLinearSolve.solve!(ls_dk, static(solver)) == Success
        @test isapprox(ls_dk.x, x_exact; rtol=1e-5)
        dense_krylov_allocs(ls, s) = @allocated DampedLinearSolve.solve!(ls, s)
        dense_krylov_allocs(ls_dk, static(solver))
        @test dense_krylov_allocs(ls_dk, static(solver)) == 0
    end

    # Test Krylov negative curvature handling
    A_indef = sparse([-5.0 1.0 0.0; 1.0 6.0 1.0; 0.0 1.0 7.0])
    indef_cost(x) = 0.5 * dot(x, A_indef * x)
    function indef_fdf!(g, H, x)
        g .= A_indef * x
        H .= A_indef
        return 0.5 * dot(x, A_indef * x)
    end
    # B4: CG reports nonpositive curvature as Indefinite, on both sparse and dense systems
    for H_indef in (copy(A_indef), Matrix(A_indef))
        ls_indef = DampedLinearSolve.makehessiansystem(H_indef, static(:cg))
        ls_indef.b .= [1.0, 0.0, 0.0]
        @test DampedLinearSolve.solve!(ls_indef, static(:cg)) == Indefinite
        # Damping enough to make it positive definite gives Success
        dampen!(ls_indef, 10.0)
        @test DampedLinearSolve.solve!(ls_indef, static(:cg)) == Success
        @test isapprox(ls_indef.x, (Matrix(A_indef) + 10I) \ [1.0, 0.0, 0.0]; rtol=1e-6)
    end

    # 3. PardisoExt testing
    using Pardiso

    @test DampedLinearSolve.validatelinsolver(:pardiso)

    if Pardiso.mkl_is_available() || Pardiso.panua_is_available()
        # B1: repeated solves on one system must not try to reassign the const workspace field
        ls = DampedLinearSolve.makehessiansystem(copy(A_spd), static(:pardiso))
        ls.b .= b_vec
        for _ in 1:3
            @test DampedLinearSolve.solve!(ls, static(:pardiso)) == Success
            @test isapprox(ls.x, x_exact; rtol=1e-8)
        end
        ls_wrong = DampedLinearSolve.makehessiansystem(copy(A_spd), static(:auto))
        @test_throws ArgumentError DampedLinearSolve.solve!(ls_wrong, static(:pardiso))
        mismatch_message(ls_wrong, :pardiso, ":auto or :ldl")
        mismatch_message(ls, :cg, ":pardiso")
        # B8: damping changes values only, so the analysed pattern is reused
        dampen!(ls, 2.0)
        @test DampedLinearSolve.solve!(ls, static(:pardiso)) == Success
        @test isapprox(ls.x, (Matrix(A_spd) + 2I) \ b_vec; rtol=1e-8)
        @test ls.A == A_spd   # damping undone
        dampen!(ls, 0.0)
        # B8: an indefinite matrix is reported, not silently solved as indefinite
        for H_indef in (copy(A_indef), Matrix(A_indef))
            ls_indef = DampedLinearSolve.makehessiansystem(H_indef, static(:pardiso))
            ls_indef.b .= b_vec
            @test DampedLinearSolve.solve!(ls_indef, static(:pardiso)) == NotPositiveDefinite
            dampen!(ls_indef, 10.0)
            @test DampedLinearSolve.solve!(ls_indef, static(:pardiso)) == Success
            @test isapprox(ls_indef.x, (Matrix(A_indef) + 10I) \ b_vec; rtol=1e-8)
        end
    else
        # When shared library is not installed, attempting to solve throws an informative error
        ls = DampedLinearSolve.makehessiansystem(copy(A_spd), static(:pardiso))
        @test_throws ErrorException DampedLinearSolve.solve!(ls, static(:pardiso))
        ls_dense = DampedLinearSolve.makehessiansystem(Matrix(A_spd), static(:pardiso))
        @test_throws ErrorException DampedLinearSolve.solve!(ls_dense, static(:pardiso))
    end

    # B8: the lower triangle handed to Pardiso, and its map back into the full matrix
    PExt = Base.get_extension(DampedLinearSolve, :PardisoExt)
    A_full = sparse([4.0 1.0 0.0 2.0; 1.0 5.0 3.0 0.0; 0.0 3.0 6.0 0.0; 2.0 0.0 0.0 7.0])
    L, lmap = PExt.lower_triangle_with_map(A_full, Int32)
    @test L isa SparseMatrixCSC{Float64, Int32}
    @test L == tril(A_full)
    @test L.nzval == A_full.nzval[lmap]
end

# AS4: the Krylov solves neither throw nor report Success with a non-finite step on a non-finite matrix
@testset "Krylov solves on a non-finite or singular matrix (AS4)" begin
    using Krylov
    base = [4.0 1 0 0; 1 3 1 0; 0 1 2 0.5; 0 0 0.5 1]
    variants = (nan_diag = (A -> A[2, 2] = NaN), nan_off = (A -> A[1, 2] = A[2, 1] = NaN), inf_diag = (A -> A[2, 2] = Inf),
                singular = (A -> A .= [1.0 1 0 0; 1 1 0 0; 0 0 2 0; 0 0 0 1]))
    for storage in (:dense, :sparse), s in (:cg, :minres), (kind, f!) in pairs(variants), lambda in (0.0, 1e-3)
        @testset "$storage, $s, $kind, λ = $lambda" begin
        A = copy(base)
        f!(A)
        ls = storage === :dense ? DampedLinearSolve.makehessiansystem(zeros(4, 4), s) :
                                  DampedLinearSolve.makehessiansystem(sparse(ones(4, 4)), s)
        if storage === :sparse
            ls.A.nzval .= vec(A)
            ls.diag_cache .= diag(A)
        else
            ls.A .= A
        end
        ls.b .= [1.0, 2, 3, 4]
        dampen!(ls, lambda)
        st = DampedLinearSolve.solve!(ls, static(s))   # does not throw
        @test st != Success || all(isfinite, ls.x)
        @test DampedLinearSolve.isfinitesystem(ls) == (kind === :singular)
      end
    end
end

# Paths the solve-level tests moved from SecondOrderOptimization did not reach on their own (rules.md 28)
@testset "extensions.jl: the remaining branches" begin
    using Krylov
    DLS = DampedLinearSolve
    A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
    b = [1.0, 2.0, 3.0]
    # A dense system built for another solver has no Krylov workspace: the solve allocates its own
    for s in (:cg, :minres)
        ls = makehessiansystem(copy(A), :auto)
        getb(ls) .= b
        @test ls.solver_ws === nothing
        @test solve!(ls, static(s)) == Success
        @test getx(ls) ≈ A \ b rtol = 1e-6
    end
    # A tall system built for QR is not solved by LSMR
    for J in (ones(4, 2), sparse(ones(4, 2)))
        ls = makejacobiansystem(J, 2, :auto)
        @test_throws ArgumentError solve!(ls, static(:lsmr))
    end
end
