using Documenter
using TrackPad
using StaticArrays

# Build documentation
makedocs(
    sitename = "TrackPad.jl",
    modules  = [TrackPad],
    format   = Documenter.HTML(
        prettyurls       = get(ENV, "CI", nothing) == "true",
        canonical        = "https://MSU-Beam-Dynamics.github.io/TrackPad.jl",
        edit_link        = "main",
        assets           = String[],
        size_threshold   = nothing,
    ),
    pages = [
        "Home"        => "index.md",
        "User Guide"  => [
            "Getting Started" => "guide.md",
            "Elements"        => "elements.md",
        ],
        "API Reference" => "api.md",
    ],
    # Use :none during initial development; change to :exports once all
    # exported symbols have docstrings.
    checkdocs = :none,
    doctest   = true,
    warnonly  = Documenter.except(:missing_docs),
)

deploydocs(
    repo         = "github.com/MSU-Beam-Dynamics/TrackPad.jl",
    devbranch    = "main",
    push_preview = true,
)
