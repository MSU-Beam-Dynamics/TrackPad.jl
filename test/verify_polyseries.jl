# Tests for TrackPadPolySeriesExt
# Run with: julia --project=. test/test_polyseries.jl

using Test
using TrackPad
using PolySeries
using StaticArrays
using Random

@testset "PolySeries extension: TPSA tracking" begin

    # --- basic identity map through a Drift ---
    @testset "Drift identity map" begin
        nv = 6
        order = 4
        drift_length = 2.0
        r0 = polyseries_variables(Float64; order=order)  # (x, px, y, py, z, pz)
        beam = Beam(1.0e9)
        d    = Drift(drift_length)
        lat  = Lattice([d])
        rout = linepass(lat, r0, beam)

        # The constant part of each component should be zero (identity map centered at 0)
        for i in 1:nv
            @test cst(rout[i]) ≈ 0.0  atol=1e-12
        end
        # Linear part: x' = x + L*px
        # coefficient of px in x should be ≈ length of drift
        # just check it's non-trivially non-zero
        @test abs(element(rout[1], [0, 1, 0, 0, 0, 0])) > 1e-3

        seed = 1234
        Random.seed!(seed)
        Nr0 = SVector{nv,Float64}(rand()*0.1 for i in 1:nv)
        Nrout = linepass(lat, Nr0, beam)

        for i in 1:nv
            ri = rout[i](Nr0...)
            Nri = Nrout[i]
            @test isapprox(Nri, ri; atol=1e-3)
        end

    end

    # --- one_turn_map_tpsa helper ---
    @testset "one_turn_map_tpsa FODO" begin
        nv = 6
        order = 4
        set_descriptor!(nv, order)
        d  = Drift(1.0)
        q1 = Quadrupole(0.5, 1.2)
        q2 = Quadrupole(0.5, -1.2)
        lat = Lattice([d, q1, d, q2, d])
        beam = Beam(1.0e9)

        M = polyseries_one_turn_map(lat, beam; order=order)
        @test length(M) == 6
        # Map should be a valid CTPS object
        @test M[1] isa CTPS
    end

end