using Test

Base.include(
    @__MODULE__,
    joinpath(@__DIR__, "..", "examples", "fodo_cell_with_dipoles.jl"),
)
using .FODOCellWithDipolesExample

@testset "Executable examples" begin
    notebook_path = joinpath(
        @__DIR__, "..", "examples", "fodo_cell_with_dipoles.ipynb",
    )
    notebook = read(notebook_path, String)
    @test occursin("\"nbformat\": 4", notebook)
    @test occursin("FODO cell with finite dipoles", notebook)
    @test occursin("using CairoMakie", notebook)
    @test occursin("sample_fodo_optics", notebook)
    @test !occursin("NotebookSVG", notebook)

    result = run_example(; check = true, verbose = false)
    sampled = sample_fodo_optics()

    @test result.lattices.mid_qf.periodic
    @test length(result.lattices.mid_qf) == 5
    @test result.optics.mid_qf.beta > result.optics.mid_qd.beta > 0
    @test result.dispersion.mid_qf[1] > result.dispersion.mid_qd[1] > 0
    @test result.chrom_momentum[1] < 0
    @test result.chrom_momentum[2] < 0
    @test result.h.mid_qf > 0
    @test result.h.mid_qd > 0
    @test length(sampled.s) == length(sampled.betax) == length(sampled.betay) ==
        length(sampled.dispersion)
    @test all(sampled.betax .> 0)
    @test all(sampled.betay .> 0)
    @test all(sampled.dispersion .> 0)
    @test isapprox(sampled.betax[1], sampled.betax[end]; atol = 1e-11)
    @test isapprox(sampled.betay[1], sampled.betay[end]; atol = 1e-11)
    @test isapprox(sampled.dispersion[1], sampled.dispersion[end]; atol = 1e-12)
end
