using Test
using TrackPad

@testset "Lattice plot data" begin
    elements = AbstractElement[
        Drift(0.4; name=:D),
        SBend(1.0, 0.1; name=:B),
        Quadrupole(0.2, 1.0; name=:QF),
        Quadrupole(0.2, -1.0; name=:QD),
        Sextupole(0.1, 2.0; name=:SF),
        Octupole(0.1, -3.0; name=:OD),
        RFCavity(0.0, 1.0e6, 500.0e6; name=:RF),
        CrabCavity(0.0; name=:CRAB),
        Marker(name=:BPM1),
        Corrector(0.0, -1.0e-6, 0.0; name=:HCOR),
        Solenoid(0.2, 0.1; name=:SOL),
        Wiggler(0.2; name=:WIG, lw=0.1, Bmax=1.0, energy=1.0e9),
    ]
    lat = Lattice(elements)
    glyphs = lattice_plot_data(lat; thin_width=0.02)

    @test length(glyphs) == length(elements) - 1
    @test [glyph.kind for glyph in glyphs] == [
        :bend, :quadrupole, :quadrupole, :sextupole, :octupole,
        :rf_cavity, :crab_cavity, :bpm, :kicker, :solenoid, :wiggler,
    ]
    @test glyphs[2].height > 0
    @test glyphs[3].height < 0
    @test glyphs[5].height < 0
    @test glyphs[9].height < 0
    @test glyphs[1].centered
    @test !glyphs[2].centered
    @test !glyphs[3].centered
    @test glyphs[1].s_start == 0.4
    @test glyphs[1].s_end == 1.4

    rf = only(filter(glyph -> glyph.kind === :rf_cavity, glyphs))
    @test rf.s_start == rf.s_end
    @test rf.plot_end - rf.plot_start ≈ 0.02

    with_drifts = lattice_plot_data(lat; include_drifts=true)
    @test first(with_drifts).kind === :drift
    zero_strength = lattice_plot_data(Lattice(AbstractElement[
        Quadrupole(0.1, 0.0), Sextupole(0.1, 0.0), Octupole(0.1, 0.0),
    ]))
    @test all(glyph -> glyph.centered, zero_strength)
    @test_throws ArgumentError lattice_plot_data(lat; thin_width=0.0)
end
