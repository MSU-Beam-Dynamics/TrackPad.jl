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
        for _ in 1:nturns
            r = linepass(lat, r, beam)
        end
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
        RFCavity(T(0.1), T(2e5), T(80e6), T(2e-4); energy=energy, charge=T(-1)),
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

    ring = Lattice(supported_gpu_elements(T, beam.energy); periodic=true)
    expected = scalar_reference(ring, beam, coords0, 3)
    actual = copy(coords0)
    batch_ringpass!(actual, GPULattice(ring, beam; dtype=T), 3)
    @test isapprox(actual, expected; rtol=3e-12, atol=3e-13)
    @test_throws ArgumentError batch_ringpass!(
        copy(coords0), GPULattice(Lattice([Drift(T(0.1))]), beam; dtype=T), 1,
    )
    @test_throws ArgumentError batch_linepass!(
        copy(coords0), GPULattice(Lattice([Drift(T(0.1))]), beam; dtype=T), 2,
    )
    @test_throws ArgumentError cpu_batch_linepass!(
        copy(coords0), Lattice([Drift(T(0.1))]), beam, 2,
    )
    @test_throws ArgumentError batch_linepass!(actual, GPULattice(ring, beam; dtype=T), -1)
end

@testset "GPU parameter sweep parity" begin
    T = Float64
    beam = Beam(T(3e9))
    strengths = collect(T, range(T(0.4), T(0.8); length=12))
    xkicks = collect(T, range(T(-2e-5), T(3e-5); length=12))
    ykicks = collect(T, range(T(1e-5), T(-4e-5); length=12))
    base = Lattice([
        Quadrupole(T(0.3), strengths[1]; num_int_steps=4),
        Corrector(T(0.1), xkicks[1], ykicks[1]),
        Drift(T(0.7)),
    ])
    gl = GPULattice(base, beam; dtype=T)
    sweep = ParamSweepLattice(gl, [
        (1, 2, strengths),
        (2, 2, xkicks),
        (2, 3, ykicks),
    ])
    input = initial_coordinates(T, length(strengths))
    actual = similar(input)
    param_sweep_linepass!(actual, input, sweep)

    @test sweep.fparams === gl.fparams
    @test ndims(sweep.fparams) == 2
    @test size(sweep.variation_index) == size(gl.fparams)
    @test sweep.varied_elements == Int32[1, 1, 0]
    @test size(sweep.variation_values) == (length(strengths), 3)
    @test sweep.n_variations == 3
    @test sweep.variation_index[1, 1] == 0
    @test sweep.variation_index[2, 1] == 1
    @test sweep.variation_index[2, 2] == 2
    @test sweep.variation_index[3, 2] == 3

    expected = similar(input)
    for i in eachindex(strengths)
        lat = Lattice([
            Quadrupole(T(0.3), strengths[i]; num_int_steps=4),
            Corrector(T(0.1), xkicks[i], ykicks[i]),
            Drift(T(0.7)),
        ])
        r = linepass(lat, SVector{6,T}(input[i, :]...), beam)
        expected[i, :] .= r
    end
    @test isapprox(actual, expected; rtol=2e-13, atol=2e-14)

    @test_throws DimensionMismatch param_sweep_linepass!(
        zeros(T, length(strengths) - 1, 6), input, sweep)
    @test_throws DimensionMismatch param_sweep_linepass!(
        actual, zeros(T, length(strengths), 5), sweep)
    @test_throws ArgumentError ParamSweepLattice(gl, [])
    @test_throws ArgumentError ParamSweepLattice(gl, [(1, 2, strengths)]; mode=:invalid)
    @test_throws DimensionMismatch ParamSweepLattice(
        gl, [(1, 2, strengths), (2, 2, xkicks[1:end-1])])
    @test_throws ArgumentError ParamSweepLattice(
        gl, [(1, 2, strengths), (1, 2, strengths)])
    @test_throws BoundsError ParamSweepLattice(gl, [(0, 2, strengths)])
    @test_throws BoundsError ParamSweepLattice(gl, [(1, 19, strengths)])
end

@testset "GPU Cartesian parameter sweep parity" begin
    T = Float64
    beam = Beam(T(3e9))
    qf_values = T[0.4, 0.6]
    qd_values = T[-0.7, -0.5, -0.3]
    base = Lattice([
        Quadrupole(T(0.2), first(qf_values); num_int_steps=4),
        Drift(T(0.4)),
        Quadrupole(T(0.2), first(qd_values); num_int_steps=4),
    ])
    gl = GPULattice(base, beam; dtype=T)
    sweep = ParamSweepLattice(
        gl,
        [(1, 2, qf_values), (3, 2, qd_values)];
        mode=:cartesian,
    )
    combinations = collect(Iterators.product(qf_values, qd_values))
    n = length(combinations)
    input = initial_coordinates(T, n)
    actual = similar(input)
    param_sweep_linepass!(actual, input, sweep)

    @test sweep.n_configs == n
    @test size(sweep.variation_values) == (n, 2)
    @test vec(sweep.variation_values[:, 1]) ==
        vec([values[1] for values in combinations])
    @test vec(sweep.variation_values[:, 2]) ==
        vec([values[2] for values in combinations])

    expected = similar(input)
    for (i, (qf, qd)) in enumerate(combinations)
        lat = Lattice([
            Quadrupole(T(0.2), qf; num_int_steps=4),
            Drift(T(0.4)),
            Quadrupole(T(0.2), qd; num_int_steps=4),
        ])
        expected[i, :] .= linepass(lat, SVector{6,T}(input[i, :]...), beam)
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
