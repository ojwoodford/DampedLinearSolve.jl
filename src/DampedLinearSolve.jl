"""
    DampedLinearSolve

Solve, without allocating, the damped linear problems a second-order optimizer poses,
`(H + λD) x = −g` and `min ‖J x + r‖² + λ‖x‖²_D`, and report the outcome well enough for the
optimizer to decide what to do next.
"""
module DampedLinearSolve

end # module DampedLinearSolve
