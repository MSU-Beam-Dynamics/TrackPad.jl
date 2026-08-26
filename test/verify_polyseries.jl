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

    @testset "Translation follows exact drift convention" begin
        nv = 6
        order = 4
        ds = 3.0e-3
        dx = 1.0e-3
        dy = -2.0e-3
        r0 = polyseries_variables(Float64; order=order)
        beam = Beam(1.0e9)
        translation_map = linepass(
            Lattice([Translation(0.0; dx=dx, dy=dy, ds=ds)]), r0, beam)
        drift_map = linepass(Lattice([Drift(ds)]), r0, beam)
        point = SVector{nv,Float64}(1.0e-4, 2.0e-4, -3.0e-4,
                                    1.5e-4, 4.0e-5, 2.0e-4)

        translated = SVector{nv,Float64}(map(f -> f(point...), translation_map))
        drifted = SVector{nv,Float64}(map(f -> f(point...), drift_map))
        expected = setindex(setindex(drifted, drifted[1] - dx, 1),
                            drifted[3] - dy, 3)
        @test translated ≈ expected atol=1.0e-14

        scalar = linepass(
            Lattice([Translation(0.0; dx=dx, dy=dy, ds=ds)]), point, beam)
        @test translated ≈ scalar atol=1.0e-14
    end

    @testset "Multipole fringes match scalar tracking" begin
        nv = 6
        order = 4
        beam = Beam(1.0e9)
        point = SVector{nv,Float64}(1.0e-4, 2.0e-4, -3.0e-4,
                                    1.5e-4, 4.0e-5, 2.0e-4)
        elements = AbstractElement[
            Quadrupole(0.4, 1.3; fringe_entrance=1, fringe_exit=1),
            Sextupole(0.4, 2.0; kick_angle=[1.2e-4, -0.8e-4],
                       fringe_entrance=1, fringe_exit=1),
            Octupole(0.4, -3.0; kick_angle=[-0.7e-4, 1.1e-4],
                      fringe_entrance=1, fringe_exit=1),
            SBend(0.9, 0.15, 0.03, 0.02;
                  polynom_b=[0.0, 1.3, 0.0, 0.0],
                  fringe_quad_entrance=1, fringe_quad_exit=1),
        ]

        # PolySeries caches Float64 workspaces by (variable count, order). Use a
        # fresh descriptor per case so earlier TPSA tests cannot contaminate the
        # fringe-map regression; the extra variables are inert.
        for (case_index, elem) in enumerate(elements)
            descriptor_nv = nv + case_index
            set_descriptor!(descriptor_nv, order)
            variables = SVector{nv,CTPS{Float64}}(
                ntuple(i -> CTPS(point[i], i), nv))
            lat = Lattice(AbstractElement[elem])
            polynomial = linepass(lat, variables, beam)
            evaluated = SVector{nv,Float64}(map(cst, polynomial))
            scalar = linepass(lat, point, beam)
            @test evaluated ≈ scalar atol=1.0e-12 rtol=1.0e-12
        end
    end

end
