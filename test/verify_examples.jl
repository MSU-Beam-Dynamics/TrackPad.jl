using Test
using LinearAlgebra
using StaticArrays
using TrackPad

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")

Base.include(
    @__MODULE__,
    joinpath(EXAMPLES_DIR, "advanced", "fodo_cell_derivations.jl"),
)
using .FODOCellWithDipolesExample

Base.include(@__MODULE__, joinpath(EXAMPLES_DIR, "common.jl"))
using .TrackPadExamples

@testset "Numbered notebook curriculum" begin
    notebooks = [
        "00_fodo_quickstart.ipynb",
        "01_madx_import.ipynb",
        "02_tpsa_map.ipynb",
        "03_ad_tracking.ipynb",
        "04_ad_tpsa.ipynb",
        "05_impedance_wake.ipynb",
        "06_lattice_matching.ipynb",
        "07_orbit_correction.ipynb",
        "08_optics_correction.ipynb",
        "09_gpu_and_parameter_sweeps.ipynb",
    ]
    required_calls = [
        "periodic_twiss", "read_madx", "tpsa_map", "batch_jacobian!",
        "Enzyme.autodiff", "LongitudinalRLCWake", "gettune",
        "find_closed_orbit_4d", "periodic_twiss", "ParamSweepLattice",
    ]

    for (filename, required_call) in zip(notebooks, required_calls)
        path = joinpath(EXAMPLES_DIR, filename)
        @test isfile(path)
        notebook = read(path, String)
        @test occursin("\"nbformat\": 4", notebook)
        @test occursin(required_call, notebook)
        @test !occursin(r"\"outputs\": \[[^\]]", notebook)
        @test !occursin(r"\"execution_count\": [0-9]", notebook)
    end

    for filename in notebooks[3:end]
        @test occursin(
            "madx_fodo", read(joinpath(EXAMPLES_DIR, filename), String),
        )
    end

    for filename in notebooks[3:5]
        @test occursin(
            "Pkg.instantiate()", read(joinpath(EXAMPLES_DIR, filename), String),
        )
    end

    ad_notebook = read(joinpath(EXAMPLES_DIR, "03_ad_tracking.ipynb"), String)
    @test occursin("finite_difference", ad_notebook)
    @test occursin("max_abs_error", ad_notebook)
    @test occursin("max_rel_error", ad_notebook)

    ad_tpsa_notebook = read(joinpath(EXAMPLES_DIR, "04_ad_tpsa.ipynb"), String)
    @test occursin("finite_difference", ad_tpsa_notebook)
    @test occursin("abs_error", ad_tpsa_notebook)
    @test occursin("relative_error", ad_tpsa_notebook)

    @test occursin(
        "mode=:cartesian",
        read(joinpath(EXAMPLES_DIR, "09_gpu_and_parameter_sweeps.ipynb"), String),
    )
    @test occursin(
        "mode=:aligned",
        read(joinpath(EXAMPLES_DIR, "09_gpu_and_parameter_sweeps.ipynb"), String),
    )
end

@testset "Concise example fixtures" begin
    manifest = read(joinpath(EXAMPLES_DIR, "Manifest.toml"), String)
    trackpad_entry = match(r"\[\[deps\.TrackPad\]\][\s\S]*?(?=\n\[\[|\z)", manifest)
    @test !isnothing(trackpad_entry)
    isnothing(trackpad_entry) || @test occursin("\"Random\"", trackpad_entry.match)

    beam = default_beam()
    ring = fodo_ring()
    twiss = periodic_twiss(ring, beam)
    @test isperiodic(ring)
    @test length(ring) == 5
    @test all(isfinite, (twiss.tunex, twiss.tuney))

    imported, imported_beam = read_madx(
        joinpath(EXAMPLES_DIR, "data", "fodo.madx"); sequence=:FODO,
    )
    @test isperiodic(imported)
    @test length(imported) == 17
    @test total_length(imported) == 10.0
    @test count(e -> e isa SBend, imported) == 2
    @test count(e -> e isa Sextupole, imported) == 4
    @test count(e -> e isa Quadrupole, imported) == 3
    @test unique(e.L for e in imported if e.name == :dqb) == [0.5]
    @test unique(e.L for e in imported if e.name == :dqs) == [0.05]
    @test imported_beam.energy ≈ 3.0e9 atol=1.0
    imported_twiss = periodic_twiss(
        imported, imported_beam;
        sample_integrator_steps=true, max_step=0.1,
    )
    @test length(imported_twiss.s) > length(imported) + 1
    @test all(isfinite, (imported_twiss.tunex, imported_twiss.tuney))
    @test maximum(abs, imported_twiss.dx) > 0
    @test isapprox(imported_twiss.dx[1], imported_twiss.dx[end]; atol=1e-12)
    @test isapprox(imported_twiss.dy[1], imported_twiss.dy[end]; atol=1e-12)

    madx_notebook = read(
        joinpath(EXAMPLES_DIR, "01_madx_import.ipynb"), String,
    )
    for calculation in (
        "periodic_twiss", "sample_integrator_steps=true", "max_step=0.10",
        "getchrom", "plot_lattice!",
        "tw.dx", "tw.dy",
        "tw.betax", "tw.betay", "tw.alphax", "tw.alphay",
    )
        @test occursin(calculation, madx_notebook)
    end
    @test !occursin("cyclic_map", madx_notebook)

    instrumented = instrumented_fodo()
    closed = find_closed_orbit_4d(instrumented, beam)
    initial = @SVector [closed[1], closed[2], closed[3], closed[4], 0.0, 0.0]
    orbit = boundary_orbit(instrumented, beam, initial)
    @test length(bpm_values(instrumented, orbit[:, 1])) == 4
end

@testset "Detailed FODO validation remains available" begin
    notebook_path = joinpath(
        EXAMPLES_DIR, "advanced", "fodo_cell_derivations.ipynb",
    )
    notebook = read(notebook_path, String)
    @test occursin("\"nbformat\": 4", notebook)
    @test occursin("FODO cell with finite dipoles", notebook)
    @test occursin("using CairoMakie", notebook)
    @test !occursin("NotebookSVG", notebook)

    result = run_example(; check=true, verbose=false)
    sampled = sample_fodo_optics()
    @test result.lattices.mid_qf.periodic
    @test result.optics.mid_qf.beta > result.optics.mid_qd.beta > 0
    @test result.chrom_momentum[1] < 0
    @test result.chrom_momentum[2] < 0
    @test all(sampled.betax .> 0)
    @test all(sampled.betay .> 0)
    @test isapprox(sampled.betax[1], sampled.betax[end]; atol=1e-11)
    @test isapprox(sampled.betay[1], sampled.betay[end]; atol=1e-11)
end

# A section header whose title equals a docstring name in the same page gives
# two headings with the same slug, and Documenter then refuses to resolve any
# `[`Name`](@ref)` written on that page (newer versions make it a hard error).
# Duplicate section titles within a page are ambiguous the same way.
@testset "Documentation headings have unambiguous slugs" begin
    docs_src = joinpath(@__DIR__, "..", "docs", "src")
    for file in sort(filter(f -> endswith(f, ".md"), readdir(docs_src)))
        text = read(joinpath(docs_src, file), String)
        # names listed in @docs blocks become headings with the binding's slug
        documented = Set{String}()
        for block in eachmatch(r"```@docs\n(.*?)```"s, text)
            for line in split(block.captures[1], '\n')
                name = strip(first(split(line, '(')))
                isempty(name) || push!(documented, name)
            end
        end
        # section titles, ignoring fenced code (where `#` starts a comment)
        headings = String[]
        for line in eachmatch(r"^#+\s+(.+?)\s*$"m, replace(text, r"```.*?```"s => ""))
            push!(headings, String(line.captures[1]))
        end
        for h in headings
            @test !(h in documented) ||
                  "$file: section `$h` collides with the docstring of the same name" == ""
            @test count(==(h), headings) == 1 ||
                  "$file: section `$h` appears more than once" == ""
        end
    end
end
