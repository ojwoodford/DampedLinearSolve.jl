using DampedLinearSolve, Test

# The public API is exactly what is exported, and every exported name is documented (rules.md 1, 10).
# Adding or dropping an export is a change to this list.
const PUBLIC_API = sort([
    :AbstractLinearSystem,
    :DampedLinearSolve,
    :Indefinite,
    :NotPositiveDefinite,
    :SolveStatus,
    :Success,
    :check_jacobian_linsolver,
    :check_linear_system,
    :checkhessian,
    :checkjacobian,
    :checklinsolver,
    :conformance_fill!,
    :conformance_fillb!,
    :conformance_matrix,
    :conformance_traits,
    :copy_structure,
    :dampen!,
    :dampingfloor,
    :getA,
    :getJ,
    :getb,
    :getdiag!,
    :getr,
    :getx,
    :gradient!,
    :isfinitesystem,
    :jacobian_ncols,
    :jacobian_nrows,
    :makehessiansystem,
    :makejacobiansystem,
    :markstale!,
    :max_diag,
    :quadform,
    :quadratic_model_change,
    :sharex,
    :solve!,
    :structurekey,
    :validatelinsolver,
    :zero!,
])

@testset "Public API" begin
    @test issorted(PUBLIC_API)
    @test sort(names(DampedLinearSolve)) == PUBLIC_API
    # (Base.Docs.hasdoc is not in Julia 1.10, so look the binding up in the module's docstrings)
    docs = Base.Docs.meta(DampedLinearSolve)
    @testset "$name has a docstring" for name in PUBLIC_API
        @test haskey(docs, Base.Docs.Binding(DampedLinearSolve, name))
    end
    # Internal names stay defined, under the module
    @testset "$name is defined but internal" for name in (
            :StaticLinearSystem, :DenseLinearSystem, :SparseLinearSystem,
            :DenseJacobianLinearSystem, :SparseJacobianLinearSystem, :StaticJacobianLinearSystem,
            :AbstractJacobianLinearSystem, :JacobianStorage, :SparseJacobianStorage, :StaticJacobianStorage,
            :MAX_STATIC_JACOBIAN, :fast_dot, :linsolver_symbol, :uniformscaling!, :invsym)
        @test isdefined(DampedLinearSolve, name)
        @test !(name in names(DampedLinearSolve))
    end
end
