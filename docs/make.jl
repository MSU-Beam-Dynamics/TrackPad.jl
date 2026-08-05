# Restrict load path to only the docs project + stdlib.
# This prevents optional extensions (e.g. TrackPadPolySeriesExt) from
# being triggered by packages installed in the user's global environment.
empty!(Base.LOAD_PATH)
push!(Base.LOAD_PATH, "@")
push!(Base.LOAD_PATH, "@stdlib")

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
            "Conventions"     => "conventions.md",
            "Elements"        => "elements.md",
            "File I/O"        => "io.md",
            "GPU Acceleration" => "gpu.md",
        ],
        "AI Agent Guide" => "agent-guide.md",
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
