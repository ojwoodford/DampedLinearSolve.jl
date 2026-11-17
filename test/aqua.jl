using DampedLinearSolve, Aqua, Test

@testset "Aqua" begin
    Aqua.test_all(DampedLinearSolve)
end
