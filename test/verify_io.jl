using Test
using TrackPad
using StaticArrays
using YAML

function write_fixture(contents)
    path, io = mktemp()
    write(io, contents)
    close(io)
    return path
end

@testset "Consumer-neutral core API" begin
    removed_names = (
        :eicvibe_capabilities,
        :eicvibe_compile_yaml,
        :eicvibe_twiss,
        :eicvibe_twiss_yaml,
        :eicvibe_track,
        :eicvibe_track_bpm,
        :eicvibe_closed_orbit,
    )
    @test all(name -> !isdefined(TrackPad, name), removed_names)
    @test all(
        name -> !(name in names(TrackPad)),
        (:resolve_pals, :resolve_pals_full, :compile_branch, :CompiledBranch),
    )
    @test !isdefined(TrackPad, :resolve_pals_full)
    @test !isdefined(TrackPad, :compile_branch)
    @test !isdefined(TrackPad, :CompiledBranch)

    beam = Beam(1.0e9)
    ring = Lattice(AbstractElement[
        Drift(1.0; name=:D1),
        Quadrupole(0.3, 0.7; name=:QF),
        Drift(1.0; name=:D2),
        Quadrupole(0.3, -0.7; name=:QD),
    ]; periodic=true)
    initial = @SVector [0.0, 1.0e-3, 0.0, 0.0, 0.0, 0.0]
    tracked = linepass(ring, initial, beam)
    @test tracked isa SVector{6,Float64}
    @test all(isfinite, tracked)

    optics = periodic_twiss(ring, beam)
    @test length(optics.s) == length(ring) + 1
    @test all(isfinite, optics.betax)
    @test find_closed_orbit_4d(ring, beam) ≈ zeros(4) atol=1e-12

    path = write_fixture("""
    PALS:
      facility:
        - frame_change:
            kind: Patch
            PatchP:
              x_offset: 0.0309
        - transport:
            kind: BeamLine
            line: [frame_change]
    """)
    patch_line, _ = read_pals(path; beam_energy=beam.energy)
    @test !isperiodic(patch_line)
    @test patch_line[1] isa Patch
    @test patch_line[1].x_offset == 0.0309
    rm(path; force=true)
end

@testset "Canonical PALS branch resolution" begin
    path = write_fixture("""
    PALS:
      facility:
        - beginning:
            kind: BeginningEle
            ReferenceP:
              species_ref: electron
              pc_ref: 3.0e9
        - qbase:
            kind: Quadrupole
            length: 0.5
            MagneticMultipoleP:
              Kn1: 1.2
              Ks1: 0.03
        - sub:
            kind: BeamLine
            line:
              - qbase
              - qneg:
                  inherit: qbase
                  MagneticMultipoleP:
                    Kn1: -1.2
        - ring_line:
            kind: BeamLine
            periodic: true
            line:
              - beginning
              - sub:
                  repeat: 2
              - b1:
                  kind: Bend
                  length: 1.25
                  BendP:
                    angle_ref: 0.125
                    e1: 0.01
                    e2: 0.02
              - rf:
                  kind: RFCavity
                  length: 0.2
                  RFP:
                    voltage: 2.0e6
                    frequency: 5.0e8
                    phase: 0.25
                    harmon: 4
        - collider:
            kind: Lattice
            branches:
              - electron_ring:
                  inherit: ring_line
                  periodic: true
        - use: collider
    """)

    lattice, beam = read_pals(path)
    @test isperiodic(lattice)
    @test lattice.name == :collider
    @test length(lattice) == 6
    @test lattice[1] isa Quadrupole
    @test lattice[1].k1 == 1.2
    @test lattice[1].polynom_a[2] == 0.03
    @test lattice[2].k1 == -1.2
    @test lattice[5] isa SBend
    @test lattice[5].angle == 0.125
    @test lattice[6] isa RFCavity
    @test lattice[6].energy == beam.energy
    @test lattice[6].charge == -1.0
    @test lattice[6].lag ≈ 0.25 * 2.99792458e8 / 5.0e8
    @test beam.mass == M_ELECTRON
    @test beam.charge == -1.0
    @test beam.energy ≈ sqrt((3.0e9)^2 + M_ELECTRON^2) - M_ELECTRON

    rm(path; force=true)
end

@testset "Inline PALS lattice branches" begin
    path = write_fixture("""
    PALS:
      facility:
        - line_machine:
            kind: Lattice
            branches:
              - transport:
                  kind: BeamLine
                  line:
                    - d1:
                        kind: Drift
                        length: 1.5
        - use: line_machine
    """)

    lattice, _ = read_pals(path; branch=:transport)
    @test !isperiodic(lattice)
    @test lattice[1] isa Drift
    @test lattice[1].L == 1.5

    rm(path; force=true)
end

@testset "Strict PALS errors" begin
    path = write_fixture("""
    PALS:
      facility:
        - mystery:
            kind: UnsupportedDevice
            length: 0.4
        - line:
            kind: BeamLine
            line: [mystery]
    """)

    @test_throws ArgumentError read_pals(path)
    lattice, _ = read_pals(path; strict=false)
    @test lattice[1] isa Drift
    @test lattice[1].L == 0.4

    rm(path; force=true)
end

@testset "Canonical PALS round trip" begin
    beam = Beam(2.0e9; mass=M_PROTON, charge=1.0)
    original = Lattice(
        AbstractElement[
            Drift(1.0; name=:D),
            Drift(2.0; name=:D),
            Quadrupole(
                0.4, 0.7;
                name=:Q, polynom_a=[0.0, 0.02, 0.0, 0.0],
            ),
            SBend(1.2, 0.15, 0.01, 0.02; name=:B),
            RFCavity(
                0.1, 1.5e6, 4.0e8, 0.03;
                name=:RF, h=8.0, energy=beam.energy,
            ),
        ];
        name=:machine,
        periodic=true,
    )

    path = tempname() * ".yaml"
    write_pals(
        path, original;
        beam=beam, lattice_name="machine", branch_name="ring",
    )
    raw = YAML.load_file(path; dicttype=Dict{String,Any})
    facility = raw["PALS"]["facility"]
    definitions = Dict{String,Any}()
    for item in facility
        name, value = first(item)
        definitions[string(name)] = value
    end
    @test definitions["B"]["kind"] == "Bend"
    @test definitions["machine"]["kind"] == "Lattice"
    @test definitions["machine"]["branches"] == ["ring"]
    @test haskey(definitions, "D__2")
    @test definitions["use"] == "machine"

    restored, restored_beam = read_pals(path)
    @test isperiodic(restored)
    @test length(restored) == length(original)
    @test restored[1].L == 1.0
    @test restored[2].L == 2.0
    @test restored[3].k1 == 0.7
    @test restored[3].polynom_a[2] == 0.02
    @test restored[4].angle == 0.15
    @test restored[5].volt == 1.5e6
    @test restored[5].lag ≈ 0.03
    @test restored[5].charge == 1.0
    @test restored_beam.mass == M_PROTON
    @test restored_beam.charge == 1.0
    @test restored_beam.energy ≈ beam.energy

    rm(path; force=true)
end

@testset "MAD-X beam and case conventions" begin
    path = write_fixture("""
    beam, particle=proton, pc=3.0;
    d: drift, l=1.0;
    rf: rfcavity, l=0.1, volt=2.0, freq=500.0, lag=0.25, harmon=4;
    ring: line=(d, rf);
    use, period=ring;
    """)

    lattice, beam = read_madx(path; sequence="RING")
    @test isperiodic(lattice)
    @test lattice.name == :ring
    @test length(lattice) == 2
    @test lattice[1] isa Drift
    @test lattice[2] isa RFCavity
    @test lattice[2].volt == 2.0e6
    @test lattice[2].freq == 5.0e8
    @test lattice[2].lag ≈ 0.25 * 2.99792458e8 / 5.0e8
    @test lattice[2].charge == 1.0
    @test lattice[2].energy == beam.energy
    @test beam.mass == M_PROTON
    @test beam.charge == 1.0
    @test beam.energy ≈ sqrt((3.0e9)^2 + M_PROTON^2) - M_PROTON

    rm(path; force=true)
end

@testset "MAD-X RF phase uses reference charge" begin
    path = write_fixture("""
    beam, particle=electron, energy=9.0;
    rf: rfcavity, volt=30.12, freq=591.0, lag=0.5, harmon=2971;
    ring: line=(rf);
    use, period=ring;
    """)

    lattice, beam = read_madx(path)
    cavity = lattice[1]
    @test cavity isa RFCavity
    @test cavity.charge == -1.0
    @test cavity.lag ≈ 0.5 * 2.99792458e8 / cavity.freq

    dz = 1.0e-6
    plus = pass!(
        cavity, SVector(0.0, 0.0, 0.0, 0.0, dz, 0.0), inv(beam.beta),
    )
    minus = pass!(
        cavity, SVector(0.0, 0.0, 0.0, 0.0, -dz, 0.0), inv(beam.beta),
    )
    rf_slope = (plus[6] - minus[6]) / (2dz)
    p0c = beam.beta * (beam.energy + beam.mass)
    expected = -cavity.charge * cavity.volt / p0c *
               (2π * cavity.freq / 2.99792458e8)
    @test rf_slope > 0
    @test rf_slope ≈ expected rtol=1.0e-6

    # For z=-c*Δt, a positive-slip ring has negative R56 and requires a
    # positive RF slope for stability.
    r56 = -2.5
    @test abs(2 + r56 * rf_slope) < 2

    rm(path; force=true)
end

@testset "MAD-X labeled beam and dotted identifiers" begin
    path = write_fixture("""
    beam_def: beam, particle=electron, energy=9.0;
    lpipe = 1.25;
    mar.sep: marker;
    es.pipe01: drift, l:=lpipe;
    b.rect: rbend, l=1.816, angle=-0.0392699;
    ring: line=(mar.sep, es.pipe01, b.rect);
    use, period=ring;
    """)

    lattice, beam = read_madx(path; num_int_steps=24)
    @test isperiodic(lattice)
    @test length(lattice) == 3
    @test lattice[1] isa Marker
    @test lattice[1].name == Symbol("mar.sep")
    @test lattice[2] isa Drift
    @test lattice[2].name == Symbol("es.pipe01")
    @test lattice[2].L == 1.25
    @test lattice[3] isa SBend
    @test lattice[3].L ≈ 1.816 * -0.0392699 / (2sin(-0.0392699 / 2))
    @test lattice[3].num_int_steps == 24
    @test beam.energy ≈ 9.0e9 - M_ELECTRON
    @test_throws ArgumentError read_madx(path; num_int_steps=0)

    rm(path; force=true)
end

@testset "MAD-X skew multipole slots and open boundary" begin
    path = write_fixture("""
    beam, particle=electron, energy=3.0;
    q: quadrupole, l=0.2, k1=0.7, k1s=0.11;
    s: sextupole, l=0.2, k2=1.2, k2s=0.22;
    o: octupole, l=0.2, k3=1.4, k3s=0.33;
    b: sbend, l=0.4, angle=0.02, k1s=0.44;
    transport: line=(q, s, o, b);
    """)

    lattice, _ = read_madx(path)
    @test !isperiodic(lattice)
    @test lattice[1].polynom_a == @SVector [0.0, 0.11, 0.0, 0.0]
    @test lattice[2].polynom_a == @SVector [0.0, 0.0, 0.22, 0.0]
    @test lattice[3].polynom_a == @SVector [0.0, 0.0, 0.0, 0.33]
    @test lattice[4].polynom_a == @SVector [0.0, 0.44, 0.0, 0.0]

    rm(path; force=true)
end
