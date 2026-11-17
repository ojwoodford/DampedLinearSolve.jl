!in(@__DIR__, LOAD_PATH) && push!(LOAD_PATH, @__DIR__)

using Test

@testset "DampedLinearSolve Tests" begin
    include("publicapi.jl")
    include("imports.jl")
    include("aqua.jl")
    include("utils.jl")
    include("linearsolve.jl")
    include("conformance.jl")
    include("tallsystem.jl")
    include("sparsetallsystem.jl")
    include("statictallsystem.jl")
    include("extensions.jl")
    include("lsmr.jl")
end
