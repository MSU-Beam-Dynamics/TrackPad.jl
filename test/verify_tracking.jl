using Test
using TrackPad
using StaticArrays
using LinearAlgebra

# The unified tracking interface: `track`/`track!` for one particle or a bunch,
# one pass or many turns, serial, threaded or on a packed GPULattice. The
# JuTrack-compatible `linepass`/`linepass!` are the single-pass spellings and
# keep their own loss convention.

const beam = Beam(3.0e9)
const cell = AbstractElement[
    Quadrupole(0.5, 0.9; num_int_steps=4), Drift(0.6),
    SBend(1.2, 2π/40; num_int_steps=4), Drift(0.6),
    Sextupole(0.2, 2.0; num_int_steps=2),
    Quadrupole(0.5, -0.9; num_int_steps=4), Drift(0.6),
    SBend(1.2, 2π/40; num_int_steps=4), Drift(0.6),
]
const ring = Lattice(vcat([copy(cell) for _ in 1:20]...); periodic=true)
const line = Lattice(copy(cell))
const r0 = @SVector [1.0e-3, 0.0, 0.5e-3, 0.0, 0.0, 1.0e-4]

@testset "track: one particle, one pass or many turns" begin
    # one pass is exactly the JuTrack-compatible linepass
    @test track(ring, r0, beam) === linepass(ring, r0, beam)
    @test track(line, r0, beam) === linepass(line, r0, beam)
    @test track(ring, r0, beam; nturns=0) === r0

    # n turns is n single passes
    manual = r0
    for _ in 1:7
        manual = linepass(ring, manual, beam)
    end
    @test track(ring, r0, beam; nturns=7) === manual

    # an open line has no turns to repeat
    @test track(line, r0, beam; nturns=1) === linepass(line, r0, beam)
    @test_throws ArgumentError track(line, r0, beam; nturns=2)
    @test_throws ArgumentError track(ring, r0, beam; nturns=-1)

    # a lost particle comes back as NaN and stops the turn loop
    big = @SVector [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    @test all(isnan, track(ring, big, beam; nturns=1000))

    # the default beam is the same one `linepass` uses
    @test track(ring, r0) === linepass(ring, r0)
end

@testset "track!: a bunch, serial and threaded" begin
    n = 64
    coords0 = zeros(n, 6)
    for i in 1:n
        coords0[i, 1] = 1.0e-4 * i
        coords0[i, 3] = 0.5e-4 * i
        coords0[i, 6] = 1.0e-5 * (i - 32)
    end

    # every row is the single-particle result
    serial = copy(coords0)
    track!(serial, ring, beam; nturns=25)
    for i in (1, 7, 33, 64)
        ri = SVector{6,Float64}(coords0[i, :])
        want = track(ring, ri, beam; nturns=25)
        @test isequal(collect(serial[i, :]), collect(want))
    end

    # ... and the threaded path agrees with the serial one bit for bit
    threaded = copy(coords0)
    track!(threaded, ring, beam; nturns=25, threaded=true)
    @test isequal(serial, threaded)
    # as does the positional spelling
    batched = copy(coords0)
    cpu_batch_linepass!(batched, ring, beam, 25)
    @test isequal(batched, threaded)

    # losses: NaN in the coordinates, flags on request, both paths
    lossy = copy(coords0); lossy[3, 1] = 1.0; lossy[9, 3] = 1.0
    fs = zeros(Int, n); fp = fill(false, n)
    cs = copy(lossy); track!(cs, ring, beam; nturns=50, lost=fs)
    cp = copy(lossy); track!(cp, ring, beam; nturns=50, lost=fp, threaded=true)
    @test isequal(cs, cp)
    @test fs[3] != 0 && fs[9] != 0
    @test Bool.(fs) == fp
    @test all(isnan, cs[3, :]) && all(isnan, cs[9, :])
    @test !any(isnan, cs[1, :])
    # a BitVector is accepted on the threaded path too (copied in and out)
    bits = falses(n); cb = copy(lossy)
    track!(cb, ring, beam; nturns=50, lost=bits, threaded=true)
    @test isequal(cb, cp) && Vector(bits) == fp

    # non-mutating matrix form leaves the input alone
    keep = copy(lossy)
    out = track(ring, lossy, beam; nturns=50)
    @test lossy == keep
    @test isequal(out, cs)

    # argument checks
    @test_throws ArgumentError track!(copy(coords0), line, beam; nturns=3)
    @test_throws ArgumentError track!(zeros(4, 5), ring, beam)
    @test_throws ArgumentError track!(copy(coords0), ring, beam; lost=zeros(Int, 3))
end

@testset "track! matches the packed GPULattice path" begin
    n = 16
    coords0 = zeros(n, 6)
    for i in 1:n
        coords0[i, 1] = 2.0e-4 * (i - 8)
        coords0[i, 3] = 1.0e-4 * (i - 8)
    end
    gl = GPULattice(ring, beam; dtype=Float64)
    packed = copy(coords0); track!(packed, gl; nturns=4)
    scalar = copy(coords0); track!(scalar, ring, beam; nturns=4)
    @test isapprox(packed, scalar; rtol=1e-11, atol=1e-13)
    @test_throws ArgumentError track!(copy(coords0), GPULattice(line, beam; dtype=Float64); nturns=2)
end

@testset "linepass! keeps the JuTrack loss convention" begin
    n = 8
    coords = zeros(n, 6)
    coords[2, 1] = 0.05        # inside the aperture below, then outside it
    ap = Lattice(AbstractElement[Drift(1.0; r_apertures=[-0.01, 0.01, -0.01, 0.01, 0.0, 0.0])])
    flags = zeros(Int, n)
    linepass!(coords, ap, beam, flags)
    @test flags[2] == 1
    @test !any(isnan, coords)                 # evolved coordinates are kept
    # track! reports the same loss as NaN
    coords2 = zeros(n, 6); coords2[2, 1] = 0.05
    flags2 = zeros(Int, n)
    track!(coords2, ap, beam; lost=flags2)
    @test flags2 == flags
    @test all(isnan, coords2[2, :]) && !any(isnan, coords2[1, :])
end

@testset "time-varying elements advance once per turn" begin
    nturn = Turn()
    lat = Lattice(AbstractElement[
        timed(Quadrupole(0.5, 0.6); k1 = 0.6 + 0.5 * Time() + 0.01 * nturn),
        Drift(1.0),
        Quadrupole(0.5, -0.6), Drift(1.0),
    ]; periodic=true)
    manual = r0
    for k in 0:2
        manual = linepass(lat, manual, beam; time=0.1k, turn=4 + k)
    end
    @test track(lat, r0, beam; nturns=3, time=0.0, dt_turn=0.1, turn=4) === manual

    coords = Matrix(reshape(collect(r0), 1, 6))
    track!(coords, lat, beam; nturns=3, time=0.0, dt_turn=0.1, turn=4)
    @test isapprox(collect(coords[1, :]), collect(manual); atol=1e-15)
    threaded = Matrix(reshape(collect(r0), 1, 6))
    track!(threaded, lat, beam; nturns=3, time=0.0, dt_turn=0.1, turn=4, threaded=true)
    @test isequal(threaded, coords)
end

@testset "Lattice narrows an untyped element vector" begin
    d = Drift(1.0)
    q = Quadrupole(0.5, 0.6)
    # `Any[...]` is what a vector of boxed locals (or a REPL `Any` literal)
    # looks like; it used to send the fallback constructor into itself.
    homogeneous = Lattice(Any[d, d])
    @test eltype(homogeneous.elements) === typeof(d)
    mixed = Lattice(Any[d, q, d]; periodic=true)
    @test eltype(mixed.elements) <: AbstractElement
    @test length(mixed.elements) == 3
    @test isperiodic(mixed)
    @test linepass(mixed, r0, beam) === linepass(Lattice([d, q, d]), r0, beam)
    @test_throws ArgumentError Lattice(Any[d, 1.0])
end
