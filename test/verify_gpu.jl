using Test
using TrackPad
using StaticArrays

function initial_coordinates(::Type{T}, n::Int) where T
    coords = zeros(T, n, 6)
    for i in 1:n
        scale = T(i) / T(n)
        coords[i, 1] = T(8e-4) * (T(2) * scale - one(T))
        coords[i, 2] = T(2e-4) * sin(T(i))
        coords[i, 3] = T(6e-4) * cos(T(i))
        coords[i, 4] = T(-1.5e-4) * scale
        coords[i, 5] = T(3e-4) * (scale - T(0.5))
        coords[i, 6] = T(4e-4) * (T(0.5) - scale)
    end
    return coords
end

function scalar_reference(lat::Lattice, beam::Beam{T}, coords::Matrix{T}, nturns::Int=1) where T
    result = copy(coords)
    for i in axes(result, 1)
        r = SVector{6,T}(result[i, :]...)
        r = ringpass(lat, r, beam, nturns)
        result[i, :] .= r
    end
    return result
end

function supported_gpu_elements(::Type{T}, energy::T) where T
    z4 = zeros(T, 4)
    return [
        Marker(),
        Drift(T(0.35)),
        Quadrupole(T(0.2), T(0.7); num_int_steps=4),
        Sextupole(T(0.15), T(1.1); num_int_steps=4),
        Octupole(T(0.1), T(-1.3); num_int_steps=4),
        SBend(T(0.4), T(0.03), T(0.01), T(0.012); num_int_steps=4),
        RFCavity(T(0.1), T(2e5), T(80e6), T(2e-4); energy=energy),
        Corrector(T(0.12), T(2e-5), T(-3e-5)),
        Solenoid(T(0.18), T(0.25)),
        ThinMultipole(zero(T), z4, T[0, 0.02, -0.03, 0.01]; max_order=3),
    ]
end

@testset "GPU encoding and CPU backend parity" begin
    T = Float64
    beam = Beam(T(3e9))
    coords0 = initial_coordinates(T, 32)

    for elem in supported_gpu_elements(T, beam.energy)
        lat = Lattice([elem])
        expected = scalar_reference(lat, beam, coords0)
        actual = copy(coords0)
        batch_linepass!(actual, GPULattice(lat, beam; dtype=T))
        @test isapprox(actual, expected; rtol=2e-13, atol=2e-14)
    end

    ring = Lattice(supported_gpu_elements(T, beam.energy))
    expected = scalar_reference(ring, beam, coords0, 3)
    actual = copy(coords0)
    batch_ringpass!(actual, GPULattice(ring, beam; dtype=T), 3)
    @test isapprox(actual, expected; rtol=3e-12, atol=3e-13)
    @test_throws ArgumentError batch_linepass!(actual, GPULattice(ring, beam; dtype=T), -1)
end

@testset "GPU parameter sweep parity" begin
    T = Float64
    beam = Beam(T(3e9))
    strengths = collect(T, range(T(0.4), T(0.8); length=12))
    base = Lattice([Quadrupole(T(0.3), strengths[1]; num_int_steps=4), Drift(T(0.7))])
    gl = GPULattice(base, beam; dtype=T)
    sweep = ParamSweepLattice(gl, [(1, 2, strengths)])
    input = initial_coordinates(T, length(strengths))
    actual = similar(input)
    param_sweep_linepass!(actual, input, sweep)

    expected = similar(input)
    for i in eachindex(strengths)
        lat = Lattice([Quadrupole(T(0.3), strengths[i]; num_int_steps=4), Drift(T(0.7))])
        r = linepass(lat, SVector{6,T}(input[i, :]...), beam)
        expected[i, :] .= r
    end
    @test isapprox(actual, expected; rtol=2e-13, atol=2e-14)
end

@testset "Unsupported GPU elements are rejected" begin
    beam = Beam(3e9)
    @test_throws ArgumentError GPULattice(Lattice([Wiggler(0.4; lw=0.1, Bmax=0.2)]), beam)
    @test_throws ArgumentError GPULattice(Lattice([ExactSBend(0.4, 0.03)]), beam)
    @test_throws ArgumentError GPULattice(Lattice([DriftSC(0.4)]), beam)
    @test_throws ArgumentError GPULattice(Lattice([Drift(0.4; t1=ones(6))]), beam)
end
