using Test
using TrackPad
using YAML

function write_fixture(contents)
    path, io = mktemp()
    write(io, contents)
    close(io)
    return path
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

    resolved = resolve_pals(path)
    @test resolved.lattice_name == "collider"
    @test resolved.name == "electron_ring"
    @test resolved.periodic
    @test length(resolved.elements) == 7
    @test [e.name for e in resolved.elements[2:5]] ==
          ["qbase", "qneg", "qbase", "qneg"]
    @test [e.occurrence for e in resolved.elements[2:5]] == [1, 1, 2, 2]
    @test resolved.reference["pc_ref"] == 3.0e9

    lattice, beam = read_pals(path)
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

    resolved = resolve_pals(path; branch=:transport)
    @test resolved.name == "transport"
    @test length(resolved.elements) == 1
    lattice, _ = read_pals(path; branch=:transport)
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
    )

    path = tempname() * ".yaml"
    write_pals(
        path, original;
        beam=beam, lattice_name="machine", branch_name="ring", periodic=true,
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
    @test length(restored) == length(original)
    @test restored[1].L == 1.0
    @test restored[2].L == 2.0
    @test restored[3].k1 == 0.7
    @test restored[3].polynom_a[2] == 0.02
    @test restored[4].angle == 0.15
    @test restored[5].volt == 1.5e6
    @test restored[5].lag ≈ 0.03
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
    @test lattice.name == :ring
    @test length(lattice) == 2
    @test lattice[1] isa Drift
    @test lattice[2] isa RFCavity
    @test lattice[2].volt == 2.0e6
    @test lattice[2].freq == 5.0e8
    @test lattice[2].lag ≈ 0.25 * 2.99792458e8 / 5.0e8
    @test lattice[2].energy == beam.energy
    @test beam.mass == M_PROTON
    @test beam.charge == 1.0
    @test beam.energy ≈ sqrt((3.0e9)^2 + M_PROTON^2) - M_PROTON

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

@testset "EICViBE in-memory branch interchange" begin
    elements = Any[
        Dict(
            "name" => "d",
            "kind" => "Drift",
            "length" => 1.0,
        ),
        Dict(
            "name" => "q",
            "source_name" => "q_prototype",
            "kind" => "Quadrupole",
            "length" => 0.4,
            "MagneticMultipoleP" => Dict("Kn1" => 0.8),
        ),
        Dict(
            "name" => "d",
            "kind" => "Drift",
            "length" => 2.0,
        ),
    ]
    reference = Dict(
        "species_ref" => "electron",
        "E_tot_ref" => 1.5e9,
    )

    compiled = compile_branch(
        elements;
        machine_name=:collider,
        branch_name=:electron_ring,
        periodic=true,
        reference=reference,
    )
    @test compiled.lattice.name == :collider
    @test compiled.branch_name == "electron_ring"
    @test compiled.periodic
    @test length(compiled.lattice) == 3
    @test compiled.lattice[2] isa Quadrupole
    @test compiled.lattice[2].k1 == 0.8
    @test compiled.source_index[("d", 1)] == 1
    @test compiled.source_index[("q", 1)] == 2
    @test compiled.source_index[("d", 2)] == 3
    @test compiled.beam.energy ≈ 1.5e9 - M_ELECTRON
end
