using DampedLinearSolve, SparseArrays, LinearAlgebra, Static, Random, Test
using Krylov
using DampedLinearSolve: dampen!, zero!, Success, NotPositiveDefinite, check_linear_system
using DampedLinearSolve: getA, getb, getx, solve!
const DLS = DampedLinearSolve

# TP2: LSMR on the Jacobian of the dense and sparse tall systems (KrylovExt), against their QR solve

function tp2_set!(ls, r)
    zero!(ls)
    getA(ls).r .= r
    DLS.jtr!(getb(ls), getA(ls).J, getA(ls).r)
    return ls
end
tp2_solve!(ls, s, lambda, scaling) = (dampen!(ls, lambda, scaling); solve!(ls, static(s)))
tp2_bytes(ls, lambda) = (tp2_solve!(ls, :lsmr, lambda, nothing); @allocated tp2_solve!(ls, :lsmr, lambda, nothing))

# The exponential fit, without allocating; a sparse J written through `nonzeros` (both columns
# stored)
const TP2_T = collect(range(0, 1; length = 20))
const TP2_Y = @. 2.0 * exp(-1.5 * TP2_T)
function tp2_cost(x)
    total = 0.0
    for i in eachindex(TP2_T)
        total += (x[1] * exp(x[2] * TP2_T[i]) - TP2_Y[i])^2
    end
    return 0.5 * total
end
function tp2_resjac!(r, J, x)
    m = length(TP2_T)
    for i in 1:m
        e = exp(x[2] * TP2_T[i])
        r[i] = x[1] * e - TP2_Y[i]
        if J isa SparseMatrixCSC
            nonzeros(J)[i] = e
            nonzeros(J)[m + i] = x[1] * TP2_T[i] * e
        else
            J[i, 1] = e
            J[i, 2] = x[1] * TP2_T[i] * e
        end
    end
    return tp2_cost(x)
end

@testset "lsmr.jl" begin
    @testset "Steps match QR: $kind ($m, $n)" for kind in (:dense, :sparse), (m, n) in ((40, 6), (6, 6), (40, 1), (4, 6))
        rng = Xoshiro(m + 7n)
        J = sprandn(rng, m, n, 0.5) + sparse(1:min(m, n), 1:min(m, n), 1.0, m, n)
        r = randn(rng, m)
        ll = kind === :dense ? DLS.DenseJacobianLinearSystem(Matrix(J); linsolver = :lsmr) : DLS.SparseJacobianLinearSystem(copy(J); linsolver = :lsmr)
        lq = DLS.DenseJacobianLinearSystem(Matrix(J))
        tp2_set!(ll, r); tp2_set!(lq, r)
        for (lambda, scaling) in ((0.0, nothing), (0.3, nothing), (0.3, 0.5 .+ rand(rng, n)), (1e-8, nothing))
            a, b = tp2_solve!(ll, :lsmr, lambda, scaling), tp2_solve!(lq, :qr, lambda, scaling)
            @test a == b
            # to LSMR's tolerance (√eps), on these well-conditioned J
            a == Success && @test getx(ll) ≈ getx(lq) rtol = 1e-7
        end
        @test tp2_bytes(ll, 0.3) == 0
        @test tp2_bytes(ll, 0.0) == 0
        # an LSMR system holds no QR factorization, and a QR system no LSMR workspace
        @test_throws ArgumentError tp2_solve!(ll, :qr, 0.3, nothing)
        @test_throws ArgumentError tp2_solve!(lq, :lsmr, 0.3, nothing)
    end

    @testset "Non-finite J or r" begin
        for (J, r) in (([1.0 NaN; 0.0 1.0; 1.0 1.0], ones(3)), ([1.0 0.0; 0.0 1.0; 1.0 1.0], [1.0, Inf, 0.0]))
            for ll in (DLS.DenseJacobianLinearSystem(copy(J); linsolver = :lsmr), DLS.SparseJacobianLinearSystem(sparse(J); linsolver = :lsmr))
                tp2_set!(ll, r)
                @test tp2_solve!(ll, :lsmr, 1.0, nothing) == NotPositiveDefinite
                @test all(isnan, getx(ll))
            end
        end
    end

    @testset "Meets the linear-system interface: $kind ($m, $n)" for kind in (:dense, :sparse), (m, n) in ((8, 5), (5, 5), (3, 5), (8, 1))
        make = kind === :dense ? (s -> DLS.DenseJacobianLinearSystem(m, n; linsolver = :lsmr)) : (s -> DLS.SparseJacobianLinearSystem(sparse(ones(m, n)); linsolver = :lsmr))
        @test isnothing(check_linear_system(make; n, linsolvers = (:lsmr,), rtol = 1e-6))
    end


end
