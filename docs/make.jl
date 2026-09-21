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
        # api.md collects every docstring and is ~100 KB; that is intended.
        size_threshold      = nothing,
        size_threshold_warn = 256 * 2^10,
    ),
    pages = [
        "Home"        => "index.md",
        "User Guide"  => [
            "Getting Started" => "guide.md",
            "Conventions"     => "conventions.md",
            "Elements"        => "elements.md",
            "File I/O"        => "io.md",
            "GPU Acceleration" => "gpu.md",
            "Performance"     => "performance.md",
            "Examples"        => "examples.md",
        ],
        "AI Agent Guide" => "agent-guide.md",
        "API Reference" => "api.md",
    ],
    # Every exported symbol must carry a docstring and appear in a page.
    checkdocs = :exports,
    doctest   = true,
)

deploydocs(
    repo         = "github.com/MSU-Beam-Dynamics/TrackPad.jl",
    devbranch    = "main",
    push_preview = true,
)
