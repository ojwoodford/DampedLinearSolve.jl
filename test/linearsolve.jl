using DampedLinearSolve, SparseArrays, StaticArrays, LinearAlgebra, Test, Static
using DampedLinearSolve: quadratic_model_change

@testset "linearsolve.jl" begin
    # 1. Symmetric positive definite system
    A_mat = randn(5, 5)
    A_spd = A_mat' * A_mat + 5.0 * I
    x_true = randn(5)
    b_vec = A_spd * x_true

    # DenseLinearSystem
    for solver in (:auto, :cholesky, :ldl, :qr)
        ls = DampedLinearSolve.DenseLinearSystem{Float64}(copy(A_spd), copy(b_vec), zeros(5))
        status = DampedLinearSolve.solve!(ls, static(solver))
        @test status == DampedLinearSolve.Success
        @test isapprox(ls.x, x_true; rtol=1e-8)
    end

    # StaticLinearSystem
    for solver in (:auto, :cholesky, :ldl, :qr)
        ls = DampedLinearSolve.StaticLinearSystem{5, 25, Float64}(MMatrix{5, 5}(A_spd), MVector{5}(b_vec), zeros(MVector{5, Float64}))
        status = DampedLinearSolve.solve!(ls, static(solver))
        @test status == DampedLinearSolve.Success
        @test isapprox(ls.x, x_true; rtol=1e-8)
    end

    # SparseLinearSystem
    A_sp = sparse(A_spd)
    for solver in (:auto, :cholesky, :ldl, :qr)
        ls = DampedLinearSolve.makehessiansystem(copy(A_sp), static(solver))
        ls.b .= b_vec
        status = DampedLinearSolve.solve!(ls, static(solver))
        @test status == DampedLinearSolve.Success
        @test isapprox(ls.x, x_true; rtol=1e-8)
    end

    # 2. Indefinite symmetric system
    A_indef = randn(5, 5)
    A_indef = 0.5 * (A_indef + A_indef')
    # Make sure it's not positive definite
    A_indef[1, 1] -= 20.0
    A_indef[5, 5] += 20.0
    b_indef = A_indef * x_true

    # LDL on Dense (returns Indefinite, still solves)
    ls_dense = DampedLinearSolve.DenseLinearSystem{Float64}(copy(A_indef), copy(b_indef), zeros(5))
    status_dense_ldl = DampedLinearSolve.solve!(ls_dense, static(:ldl))
    @test status_dense_ldl == DampedLinearSolve.Indefinite
    @test isapprox(ls_dense.x, x_true; rtol=1e-8)

    # Cholesky on Dense with indefinite system fails cleanly without fallback
    ls_dense_chol = DampedLinearSolve.DenseLinearSystem{Float64}(copy(A_indef), copy(b_indef), zeros(5))
    status_dense_chol = DampedLinearSolve.solve!(ls_dense_chol, static(:cholesky))
    @test status_dense_chol == DampedLinearSolve.NotPositiveDefinite

    # Cholesky on Static with indefinite system fails cleanly without fallback
    ls_static_chol = DampedLinearSolve.StaticLinearSystem{5, 25, Float64}(MMatrix{5, 5}(A_indef), MVector{5}(b_indef), zeros(MVector{5, Float64}))
    status_static_chol = DampedLinearSolve.solve!(ls_static_chol, static(:cholesky))
    @test status_static_chol == DampedLinearSolve.NotPositiveDefinite

    # QR on Dense
    ls_dense_qr = DampedLinearSolve.DenseLinearSystem{Float64}(copy(A_indef), copy(b_indef), zeros(5))
    status_dense_qr = DampedLinearSolve.solve!(ls_dense_qr, static(:qr))
    @test status_dense_qr == DampedLinearSolve.Success
    @test isapprox(ls_dense_qr.x, x_true; rtol=1e-8)

    # LDL on Sparse (returns Indefinite, still solves)
    A_sp_indef = sparse(A_indef)
    ls_sp = DampedLinearSolve.makehessiansystem(copy(A_sp_indef), static(:ldl))
    ls_sp.b .= b_indef
    status_sp_ldl = DampedLinearSolve.solve!(ls_sp, static(:ldl))
    @test status_sp_ldl == DampedLinearSolve.Indefinite
    @test isapprox(ls_sp.x, x_true; rtol=1e-8)

    # Cholesky on Sparse with indefinite system fails cleanly without fallback
    ls_sp_chol = DampedLinearSolve.makehessiansystem(copy(A_sp_indef), static(:cholesky))
    ls_sp_chol.b .= b_indef
    status_sp_chol = DampedLinearSolve.solve!(ls_sp_chol, static(:cholesky))
    @test status_sp_chol == DampedLinearSolve.NotPositiveDefinite

    # QR on Sparse
    ls_sp_qr = DampedLinearSolve.makehessiansystem(copy(A_sp_indef), static(:qr))
    ls_sp_qr.b .= b_indef
    status_sp_qr = DampedLinearSolve.solve!(ls_sp_qr, static(:qr))
    @test status_sp_qr == DampedLinearSolve.Success
    @test isapprox(ls_sp_qr.x, x_true; rtol=1e-8)

    # 3. invsym (dense and static symmetric matrices)
    for N in (2, 3)
        M = randn(N, N)
        S = M' * M + 2.0 * I
        @test isapprox(DampedLinearSolve.invsym(S), inv(S); rtol=1e-12)
        S_inv = DampedLinearSolve.invsym(SMatrix{N, N, Float64}(S))
        @test S_inv isa SMatrix{N, N, Float64}
        @test isapprox(Matrix(S_inv), inv(S); rtol=1e-12)
    end

    # 4. try_cholesky!, max_diag, damp_marquardt!, uniformscaling!
    x_out = zeros(5)
    @test DampedLinearSolve.try_cholesky!(x_out, copy(A_spd), b_vec) == DampedLinearSolve.Success
    @test isapprox(x_out, x_true; rtol=1e-8)
    @test DampedLinearSolve.try_cholesky!(x_out, copy(A_indef), b_indef) == DampedLinearSolve.NotPositiveDefinite

    @test DampedLinearSolve.max_diag(A_spd) ≈ maximum(diag(A_spd))
    ls_sp_test = DampedLinearSolve.makehessiansystem(copy(A_sp), static(:ldl))
    @test DampedLinearSolve.max_diag(ls_sp_test) ≈ maximum(diag(A_spd))

    A_damp = copy(A_spd)
    d = [A_damp[j, j] for j in 1:5]
    DampedLinearSolve.damp_marquardt!(A_damp, d, 0.5)
    for j in 1:5
        @test A_damp[j, j] ≈ d[j] * 1.5
    end

    # 5. Damping contract: zero mutation on dense/static, exact restoration on sparse, no drift
    ls_d = DampedLinearSolve.DenseLinearSystem{Float64}(copy(A_spd), copy(b_vec), zeros(5))
    DampedLinearSolve.dampen!(ls_d, 3.0)
    @test ls_d.lambda == 3.0
    DampedLinearSolve.solve!(ls_d, static(:cholesky))
    x_damped_true = (A_spd + 3.0 * I) \ b_vec
    @test isapprox(ls_d.x, x_damped_true; rtol=1e-8)
    @test ls_d.A == A_spd # base matrix never mutated!

    ls_s = DampedLinearSolve.StaticLinearSystem{5, 25, Float64}(MMatrix{5, 5}(A_spd), MVector{5}(b_vec), zeros(MVector{5, Float64}))
    DampedLinearSolve.dampen!(ls_s, 3.0)
    DampedLinearSolve.solve!(ls_s, static(:cholesky))
    @test isapprox(ls_s.x, x_damped_true; rtol=1e-8)
    @test Matrix(ls_s.A) == A_spd # base matrix never mutated!

    ls_sparse = DampedLinearSolve.makehessiansystem(copy(A_sp), static(:ldl))
    ls_sparse.b .= b_vec
    DampedLinearSolve.dampen!(ls_sparse, 3.0)
    DampedLinearSolve.solve!(ls_sparse, static(:ldl))
    @test isapprox(ls_sparse.x, x_damped_true; rtol=1e-8)
    @test ls_sparse.A.nzval[ls_sparse.diag_inds] == diag(A_spd)

    # Sparse CHOLMOD native shift test
    ls_chol_sp = DampedLinearSolve.makehessiansystem(copy(A_sp), static(:cholesky))
    ls_chol_sp.b .= b_vec
    DampedLinearSolve.dampen!(ls_chol_sp, 3.0)
    DampedLinearSolve.solve!(ls_chol_sp, static(:cholesky))
    @test isapprox(ls_chol_sp.x, x_damped_true; rtol=1e-8)
    @test ls_chol_sp.A.nzval[ls_chol_sp.diag_inds] == diag(A_spd)

    # B6: a sparse prototype with missing diagonal entries keeps its index type
    for Ti in (Int32, Int64)
        H_missing = SparseMatrixCSC{Float64, Ti}(sparse(Ti[1, 2, 3, 1], Ti[1, 2, 1, 3], [4.0, 5.0, 1.0, 1.0], 3, 3))
        @test nnz(H_missing) == 4   # (3, 3) is missing
        A_fixed, dinds = DampedLinearSolve.ensure_diag_and_inds(H_missing)
        @test A_fixed isa SparseMatrixCSC{Float64, Ti}
        @test eltype(dinds) == Ti
        @test all(A_fixed.rowval[dinds[c]] == c for c in 1:3)
        @test A_fixed == H_missing
        ls = DampedLinearSolve.makehessiansystem(H_missing, static(:auto))
        @test ls isa DampedLinearSolve.SparseLinearSystem{Float64, Ti}
        ls.A[3, 3] = 6.0
        ls.b .= [1.0, 2.0, 3.0]
        @test DampedLinearSolve.solve!(ls, static(:auto)) == DampedLinearSolve.Success
        @test isapprox(ls.x, Matrix(ls.A) \ [1.0, 2.0, 3.0]; rtol=1e-10)
    end

    # M2: static :ldl solves an indefinite system and reports it, like dense and sparse :ldl
    ls_static_ldl = DampedLinearSolve.StaticLinearSystem{5, 25, Float64}(MMatrix{5, 5}(A_indef), MVector{5}(b_indef), zeros(MVector{5, Float64}))
    @test DampedLinearSolve.solve!(ls_static_ldl, static(:ldl)) == DampedLinearSolve.Indefinite
    @test isapprox(ls_static_ldl.x, x_true; rtol=1e-8)
    ls_static_ldl_pd = DampedLinearSolve.StaticLinearSystem{5, 25, Float64}(MMatrix{5, 5}(A_spd), MVector{5}(b_vec), zeros(MVector{5, Float64}))
    @test DampedLinearSolve.solve!(ls_static_ldl_pd, static(:ldl)) == DampedLinearSolve.Success
    @test isapprox(ls_static_ldl_pd.x, x_true; rtol=1e-8)
    ls_static_sing = DampedLinearSolve.StaticLinearSystem{2, 4, Float64}(MMatrix{2, 2}([1.0 1.0; 1.0 1.0]), MVector(1.0, 0.0), zeros(MVector{2, Float64}))
    @test DampedLinearSolve.solve!(ls_static_sing, static(:ldl)) == DampedLinearSolve.NotPositiveDefinite
    static_ldl_allocs(ls) = @allocated DampedLinearSolve.solve!(ls, static(:ldl))
    static_ldl_allocs(ls_static_ldl)
    @test static_ldl_allocs(ls_static_ldl) == 0

    # M3: indefiniteness is detected relative to the scale of the matrix
    for scale in (1e-9, 1.0, 1e9)
        A_sc = scale .* A_indef
        b_sc = A_sc * x_true
        ls_d = DampedLinearSolve.DenseLinearSystem{Float64}(copy(A_sc), copy(b_sc), zeros(5))
        @test DampedLinearSolve.solve!(ls_d, static(:ldl)) == DampedLinearSolve.Indefinite
        ls_s = DampedLinearSolve.makehessiansystem(sparse(A_sc), static(:ldl))
        ls_s.b .= b_sc
        @test DampedLinearSolve.solve!(ls_s, static(:ldl)) == DampedLinearSolve.Indefinite
        ls_st = DampedLinearSolve.StaticLinearSystem{5, 25, Float64}(MMatrix{5, 5}(A_sc), MVector{5}(b_sc), zeros(MVector{5, Float64}))
        @test DampedLinearSolve.solve!(ls_st, static(:ldl)) == DampedLinearSolve.Indefinite
        # and a positive definite matrix at the same scale is not flagged
        ls_p = DampedLinearSolve.DenseLinearSystem{Float64}(scale .* A_spd, scale .* b_vec, zeros(5))
        @test DampedLinearSolve.solve!(ls_p, static(:ldl)) == DampedLinearSolve.Success
    end
    # A 2x2 Bunch-Kaufman pivot block with one negative eigenvalue is flagged
    bk2 = bunchkaufman(Symmetric([0.0 1.0; 1.0 0.0]))
    @test DampedLinearSolve.is_indefinite(bk2)
    @test !DampedLinearSolve.is_indefinite(bunchkaufman(Symmetric([2.0 1.0; 1.0 2.0])))

    # B2/M21: the indefiniteness check on a dense :ldl factorization costs O(n), not O(n^2), memory
    n_big = 300
    M_big = randn(n_big, n_big)
    A_big = M_big + M_big'
    bk_big = bunchkaufman(Symmetric(A_big))
    DampedLinearSolve.is_indefinite(bk_big)
    @test (@allocated DampedLinearSolve.is_indefinite(bk_big)) < 8 * n_big * n_big ÷ 10
    @test DampedLinearSolve.is_indefinite(bk_big) == (eigmin(Symmetric(A_big)) < 0)

    # M9: dense :ldl with the preallocated workspace matches Bunch-Kaufman and doesn't allocate
    for A_t in (A_spd, A_indef, [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 -2.0])
        n_t = size(A_t, 1)
        x_t = randn(n_t)
        b_t = A_t * x_t
        ls_ws = DampedLinearSolve.makehessiansystem(copy(A_t), static(:ldl))
        @test ls_ws.solver_ws isa DampedLinearSolve.DenseLDLWorkspace
        ls_ws.b .= b_t
        ls_ref = DampedLinearSolve.DenseLinearSystem{Float64}(copy(A_t), copy(b_t), zeros(n_t))
        @test ls_ref.solver_ws === nothing
        st_ws = DampedLinearSolve.solve!(ls_ws, static(:ldl))
        @test st_ws == DampedLinearSolve.solve!(ls_ref, static(:ldl))
        @test st_ws == (eigmin(Symmetric(A_t)) < 0 ? DampedLinearSolve.Indefinite : DampedLinearSolve.Success)
        @test isapprox(ls_ws.x, x_t; rtol=1e-8)
        dense_ldl_allocs(ls) = @allocated DampedLinearSolve.solve!(ls, static(:ldl))
        dense_ldl_allocs(ls_ws)
        @test dense_ldl_allocs(ls_ws) == 0
        # copy_structure and sharex keep the workspace
        @test DampedLinearSolve.copy_structure(ls_ws).solver_ws === ls_ws.solver_ws
    end
    # An exactly singular matrix fails
    ls_sing = DampedLinearSolve.makehessiansystem([1.0 1.0; 1.0 1.0], static(:ldl))
    ls_sing.b .= [1.0, 0.0]
    @test DampedLinearSolve.solve!(ls_sing, static(:ldl)) == DampedLinearSolve.NotPositiveDefinite
    # The packed-factor check gives the inertia's answer, for both triangles
    for trial in 1:20
        M_t = randn(8, 8)
        A_t = M_t + M_t'
        for uplo in (:U, :L)
            bk_t = bunchkaufman(Symmetric(A_t, uplo))
            @test DampedLinearSolve.is_indefinite(bk_t) == (eigmin(Symmetric(A_t)) < 0)
        end
    end

    # M9: sparse :cholesky solves in place on Julia 1.12 and later
    ls_spc = DampedLinearSolve.makehessiansystem(sparse(A_spd), static(:cholesky))
    ls_spc.b .= b_vec
    x_before = ls_spc.x
    @test DampedLinearSolve.solve!(ls_spc, static(:cholesky)) == DampedLinearSolve.Success
    @test ls_spc.x === x_before
    @test isapprox(ls_spc.x, x_true; rtol=1e-8)

    # M17: the quadratic model change, 0.5 dx'H dx + g'dx
    ls_q = DampedLinearSolve.DenseLinearSystem{Float64}(copy(A_spd), copy(b_vec), zeros(5))
    dx_q = randn(5)
    @test quadratic_model_change(ls_q, dx_q) ≈ 0.5 * dot(dx_q, A_spd * dx_q) + dot(b_vec, dx_q)
end

# AS4: no solve reports Success with a non-finite step, and none throws, on a non-finite or singular matrix
const AS4_BASE = [4.0 1 0 0; 1 3 1 0; 0 1 2 0.5; 0 0 0.5 1]
function as4_matrix(kind)
    A = copy(AS4_BASE)
    kind === :nan_diag && (A[2, 2] = NaN)
    kind === :nan_off && (A[1, 2] = A[2, 1] = NaN)
    kind === :inf_diag && (A[2, 2] = Inf)
    kind === :singular && (A = [1.0 1 0 0; 1 1 0 0; 0 0 2 0; 0 0 0 1])
    return A
end
function as4_system(storage, A, s)
    b = [1.0, 2, 3, 4]
    if storage === :static
        ls = DampedLinearSolve.makehessiansystem(static(4), s)
    elseif storage === :dense
        ls = DampedLinearSolve.makehessiansystem(zeros(4, 4), s)
    else
        ls = DampedLinearSolve.makehessiansystem(sparse(ones(4, 4)), s)
    end
    if storage === :sparse
        ls.A.nzval .= vec(A)
        ls.diag_cache .= diag(A)
    else
        ls.A .= A
    end
    ls.b .= b
    return ls
end
as4_ok(st, x) = st != DampedLinearSolve.Success || all(isfinite, x)

@testset "No solve reports Success with a non-finite step (AS4)" begin
    for storage in (:static, :dense, :sparse), s in (:auto, :cholesky, :ldl, :qr), kind in (:nan_diag, :nan_off, :inf_diag, :singular), lambda in (0.0, 1e-3)
        @testset "$storage, $s, $kind, λ = $lambda" begin
            ls = as4_system(storage, as4_matrix(kind), s)
            DampedLinearSolve.dampen!(ls, lambda)
            st = DampedLinearSolve.solve!(ls, static(s))   # does not throw
            @test as4_ok(st, DampedLinearSolve.getx(ls))
            @test DampedLinearSolve.isfinitesystem(ls) == !(kind in (:nan_diag, :nan_off, :inf_diag))
            # A finite, positive definite matrix still solves
            kind === :singular && lambda > 0 && @test st == DampedLinearSolve.Success
        end
    end
end

# Paths the solve-level tests moved from SecondOrderOptimization did not reach on their own (rules.md 28)
@testset "linearsolve.jl: the remaining branches" begin
    DLS = DampedLinearSolve
    # The indefiniteness test of a rook-pivoted Bunch-Kaufman factorization reads its block-diagonal factor
    A_rook = [4.0 1.0 2.0; 1.0 -3.0 0.5; 2.0 0.5 3.0]
    bk = bunchkaufman(Symmetric(A_rook), true)
    @test bk.rook
    @test DLS.is_indefinite(bk) == any(<(0), eigvals(A_rook))
    @test !DLS.is_indefinite(bunchkaufman(Symmetric([4.0 1.0; 1.0 3.0]), true))
    bk2 = bunchkaufman(Symmetric([1e-3 1.0 0.0; 1.0 1e-3 0.0; 0.0 0.0 2.0]), true; check = false)   # a 2×2 pivot block
    @test any(<(0), bk2.ipiv) && DLS.is_indefinite(bk2)
    # Dense :ldl on a system built without a workspace: a singular matrix fails the factorization
    ls = DLS.DenseLinearSystem{Float64}(zeros(2, 2), ones(2), zeros(2))
    @test solve!(ls, static(:ldl)) == NotPositiveDefinite
    # Before an extension is loaded, its solver names are rejected with the package to load
    if isnothing(Base.get_extension(DampedLinearSolve, :KrylovExt))
        for s in (:cg, :minres, :lsmr)
            err = try checklinsolver(s); nothing catch e; e end
            @test err isa ArgumentError && occursin("requires Krylov.jl", err.msg)
        end
    end
    if isnothing(Base.get_extension(DampedLinearSolve, :PardisoExt))
        err = try checklinsolver(:pardiso); nothing catch e; e end
        @test err isa ArgumentError && occursin("requires Pardiso.jl", err.msg)
    end
    err = try checklinsolver(static(:not_a_solver)); nothing catch e; e end
    @test err isa ArgumentError && occursin("Unknown linear solver", err.msg)
    @test checklinsolver(:auto) && checklinsolver(static(:qr))
end
