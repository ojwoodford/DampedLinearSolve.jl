using Documenter
using DampedLinearSolve

DocMeta.setdocmeta!(DampedLinearSolve, :DocTestSetup, :(using DampedLinearSolve, LinearAlgebra, SparseArrays, Static); recursive=true)

makedocs(
    sitename = "DampedLinearSolve.jl",
    modules = [DampedLinearSolve],
    checkdocs = :exports,
    remotes = nothing,
    doctest = true,
    warnonly = false,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        sidebar_sitename = true,
    ),
    pages = [
        "Home" => "index.md",
        "The linear-system contract" => "contract.md",
        "Solvers" => "solvers.md",
        "Damping" => "damping.md",
        "Tutorial: solving damped systems" => "tutorial.md",
        "Implementing the contract" => "implementing.md",
        "API Reference" => "api.md",
    ]
)

# Publishes to the gh-pages branch when run on GitHub Actions (see .github/workflows/Documentation.yml);
# does nothing when run locally.
deploydocs(
    repo = "github.com/ojwoodford/DampedLinearSolve.jl.git",
    devbranch = "main",
    push_preview = false,
)
