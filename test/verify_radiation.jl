using Test
using Statistics
using StaticArrays
using LinearAlgebra
using TrackPad

# Independent of TrackPad's own constants: ħc and Sands' photon second moment
# N<u²> = (55/(24√3)) u_c P_γ, the coefficient of AT's diffusion matrix.
const HBARC_EV_M = 1.973269804593025e-7
const SANDS_CU = 55 / (24 * sqrt(3))
const CQ_EV_M = 55 / (32 * sqrt(3)) * HBARC_EV_M        # Cq * mc²

# Per-pass variance of the stored δE through a uniform bend on the reference
# orbit. At β0 = 1 it is AT's ∫ CU r_e λ̄ γ⁵ |h|³ ds; the factor β0 is the
# documented finite-β conversion (loss ∝ β0², times E0/(P0c) = 1/β0).
bend_variance(beam, h, L) = SANDS_CU * classical_radius(beam) *
    (HBARC_EV_M / beam.mass) * beam.beta * beam.gamma^5 * abs(h)^3 * L

@testset "Synchrotron-radiation damping" begin
    beam = Beam(3.0e9)
    L = 1.0
    angle = 0.1
    h = angle / L
    bend = SBend(L, angle; num_int_steps=16, rad=1)
    lat = Lattice(AbstractElement[bend])
    r0 = @SVector zeros(6)
    out = linepass(lat, r0, beam)

    expected = (2 / 3) * classical_radius(beam) * beam.beta^2 * beam.gamma^3 * h^2 * L
    # The analytic expression is first order in the loss; tracking updates the
    # energy and trajectory after every slice.
    @test out[6] ≈ -expected rtol=1e-5
    @test out[2] != 0                    # dispersion response after losing energy

    conservative = linepass(Lattice(AbstractElement[
        SBend(L, angle; num_int_steps=16, rad=0)]), r0, beam)
    @test conservative[6] == 0
    @test_throws ArgumentError pass!(bend, r0, inv(beam.beta))
    @test_throws ArgumentError linepass(
        Lattice(AbstractElement[SBend(L, angle; rad=3)]), r0, beam)
end

@testset "Quantum excitation and reproducibility" begin
    beam = Beam(3.0e9)
    L = 1.0
    h = 0.1
    bend = SBend(L, h * L; num_int_steps=8, rad=2)
    lat = Lattice(AbstractElement[bend])
    n = 8192
    serial = zeros(n, 6)
    threaded = copy(serial)
    track!(serial, lat, beam; radiation_seed=0x1234)
    track!(threaded, lat, beam; radiation_seed=0x1234, threaded=true)
    @test isequal(serial, threaded)

    mean_loss = (2 / 3) * classical_radius(beam) * beam.beta^2 * beam.gamma^3 * h^2 * L
    @test mean(@view serial[:, 6]) ≈ -mean_loss rtol=0.03
    @test var(@view serial[:, 6]) ≈ bend_variance(beam, h, L) rtol=0.05

    other = zeros(n, 6)
    track!(other, lat, beam; radiation_seed=0x1235)
    @test other != serial

    # At finite beta, photon-energy variance must be converted from ΔE/E to
    # TrackPad's stored ΔE/(P0*c) coordinate.  A high charge keeps this
    # deliberately low-gamma test well above floating-point noise.
    slow = Beam(2.0e8; mass=1.0e8, charge=1.0e4)
    slow_bend = SBend(L, h * L; num_int_steps=8, rad=2)
    slow_coords = zeros(4096, 6)
    track!(slow_coords, Lattice(AbstractElement[slow_bend]), slow;
           radiation_seed=0x2468)
    slow_loss = (2 / 3) * classical_radius(slow) * slow.beta^2 *
                slow.gamma^3 * h^2 * L
    @test mean(@view slow_coords[:, 6]) ≈ -slow_loss rtol=1e-3
    @test var(@view slow_coords[:, 6]) ≈ bend_variance(slow, h, L) rtol=0.08
end

@testset "Radiating multipoles and SC magnets" begin
    beam = Beam(3.0e9)
    r0 = @SVector [1e-3, 2e-4, -7e-4, 1e-4, 0.0, 2e-3]
    elems = AbstractElement[
        Quadrupole(0.3, 1.2; rad=1),
        Sextupole(0.2, 20.0; rad=1),
        Octupole(0.1, 100.0; rad=1),
        ExactSBend(0.8, 0.08; num_int_steps=4, rad=1),
        QuadrupoleSC(0.3; k1=1.2, rad=1),
        SextupoleSC(0.2; k2=20.0, rad=1),
        OctupoleSC(0.1; k3=100.0, rad=1),
        SBendSC(0.8, 0.08; rad=1),
    ]
    out = linepass(Lattice(elems), r0, beam)
    @test all(isfinite, out)
    @test out[6] < r0[6]
end

@testset "LBend radiates over its closed-form body" begin
    beam = Beam(3.0e9)
    L, h = 1.0, 0.1
    r0 = @SVector zeros(6)
    expected = (2 / 3) * classical_radius(beam) * beam.beta^2 * beam.gamma^3 * h^2 * L
    # One slice: the energy taken once, at the body's mean, as Xsuite takes a thick magnet's.
    @test linepass(Lattice(AbstractElement[LBend(L, h * L; rad=1)]), r0, beam)[6] ≈ -expected rtol=1e-3
    @test linepass(Lattice(AbstractElement[LBend(L, h * L)]), r0, beam)[6] == 0
    coords = zeros(8192, 6)
    track!(coords, Lattice(AbstractElement[LBend(L, h * L; rad=2)]), beam; radiation_seed=0x1357)
    @test var(@view coords[:, 6]) ≈ bend_variance(beam, h, L) rtol=0.05

    # A thick quadrupole (angle 0) radiates off the axis only, as its many-step
    # integration does to within the one slice's approximation.
    k1 = 0.5
    off = SVector(1e-3, 0.0, 0.0, 0.0, 0.0, 0.0)
    quad = Lattice(AbstractElement[LBend(L, 0.0; K=k1, rad=1)])
    fine = Lattice(AbstractElement[Quadrupole(L, k1; num_int_steps=64, rad=1)])
    @test linepass(quad, r0, beam)[6] == 0
    @test linepass(quad, off, beam)[6] < 0
    @test linepass(quad, off, beam)[6] ≈ linepass(fine, off, beam)[6] rtol=0.1
end

@testset "Wiggler radiation uses the tracked beam" begin
    beam = Beam(3.0e9)
    r0 = @SVector zeros(6)
    w = Wiggler(1.2; lw=0.2, Bmax=1.2, Nsteps=8, rad=1,
                energy=beam.energy, mass=beam.mass, charge=beam.charge)
    out = linepass(Lattice(AbstractElement[w]), r0, beam)
    @test out[6] < 0
    @test_throws ArgumentError linepass(Lattice(AbstractElement[
        Wiggler(1.2; lw=0.2, Bmax=1.2, rad=1)]), r0, beam)

    # Midpoint sampling has one radiation weight per physical slice and
    # converges without the former extra full entrance kick.
    fine = Wiggler(1.2; lw=0.2, Bmax=1.2, Nsteps=16, rad=1,
                   energy=beam.energy, mass=beam.mass, charge=beam.charge)
    outf = linepass(Lattice(AbstractElement[fine]), r0, beam)
    @test out[6] ≈ outf[6] rtol=5e-4

    @test linepass(Lattice(AbstractElement[
        Wiggler(0.0; lw=0.2, Bmax=1.2, rad=1,
                energy=beam.energy)]), r0, beam) == r0
    @test_throws ArgumentError Wiggler(1.1; lw=0.2, Bmax=1.2, rad=1,
                                       energy=beam.energy)

    ion1 = Beam(3.0e9; mass=M_PROTON, charge=1.0)
    ion2 = Beam(3.0e9; mass=M_PROTON, charge=2.0)
    weak1 = Wiggler(1.0; lw=0.2, Bmax=1e-3, Nsteps=8, rad=1,
                    energy=ion1.energy, mass=ion1.mass, charge=ion1.charge)
    weak2 = Wiggler(1.0; lw=0.2, Bmax=1e-3, Nsteps=8, rad=1,
                    energy=ion2.energy, mass=ion2.mass, charge=ion2.charge)
    d1 = -linepass(Lattice(AbstractElement[weak1]), r0, ion1)[6]
    d2 = -linepass(Lattice(AbstractElement[weak2]), r0, ion2)[6]
    @test d2 / d1 ≈ 16 rtol=2e-3
end

# A compact 6 GeV ring whose bends carry k1 = -h²/2: that cancels I4, so
# Jx = 1, JE = 2 and the damping times are 200 and 100 turns. Only the bends
# radiate; a quadrupole on the closed orbit does not.
const RING_CELLS = 6
function ring_cells(rad)
    h = 2pi / (2RING_CELLS)
    bend = SBend(1.0, h; polynom_b=[0.0, -h^2 / 2, 0.0, 0.0], max_order=1, rad=rad)
    cell = AbstractElement[
        Quadrupole(0.15, 2.0), Drift(0.4), bend, Drift(0.4),
        Quadrupole(0.3, -2.0), Drift(0.4), bend, Drift(0.4),
        Quadrupole(0.15, 2.0)]
    return reduce(vcat, fill(cell, RING_CELLS))
end

@testset "Ring damping partition and quantum equilibrium" begin
    beam = Beam(6.0e9)
    bare = Lattice(ring_cells(0); periodic=true)
    I = twiss(bare, beam; radiation_integrals=true).radiation
    Jx = 1 - I.I4 / I.I2
    JE = 2 + I.I4 / I.I2
    @test Jx ≈ 1 atol=0.02
    # U0/E0 from I2, with the exact (finite-β) Larmor power.
    U0_E0 = (2 / 3) * classical_radius(beam) * beam.beta^3 * beam.gamma^3 * I.I2

    # RF at three times the loss, harmonic 4, on the stable side of the crest.
    volt = 3 * U0_E0 * beam.energy
    frf = 4 * beam.beta * 299792458.0 / total_length(bare)
    lag = (pi - asin(1 / 3)) / 2pi * 299792458.0 / frf
    ring(rad) = Lattice(vcat(ring_cells(rad), AbstractElement[
        RFCavity(0.0, volt, frf, lag; energy=beam.energy, charge=beam.charge)]); periodic=true)

    # Damping rates: eigenvalues of the radiating one-turn map about its closed
    # orbit, found by tracking (optics functions deliberately ignore `rad`).
    damped = ring(1)
    co = @SVector zeros(6)
    for _ in 1:3000
        co = linepass(damped, co, beam)
    end
    @test co[6] > 0                         # the energy offset the RF restores
    step = 1e-9
    M = reduce(hcat, map(1:6) do i
        e = SVector{6}(ntuple(j -> j == i ? step : 0.0, 6))
        (linepass(damped, co + e, beam) - linepass(damped, co - e, beam)) / 2step
    end)
    F = eigen(M)
    decrement = zeros(3)
    for k in 1:6
        imag(F.values[k]) > 0 || continue
        v = abs2.(F.vectors[:, k])
        decrement[argmax((v[1] + v[2], v[3] + v[4], v[5] + v[6]))] = -log(abs(F.values[k]))
    end
    @test decrement[1] ≈ Jx * U0_E0 / 2 rtol=0.01
    @test decrement[2] ≈ U0_E0 / 2 rtol=0.01
    @test decrement[3] ≈ JE * U0_E0 / 2 rtol=0.01
    @test sum(decrement) ≈ 2 * U0_E0 rtol=0.005      # Robinson: ΣJ = 4

    # Equilibrium: a cold bunch tracked two horizontal damping times, compared
    # with Cq γ² I3/(JE I2) and Cq γ² I5/(Jx I2) times the residual transient.
    # Across seeds the ratios scatter by ~3 % about a few-percent bias from the
    # 1 % loss per turn; a diffusion missing Sands' factor 2 reads 0.5.
    excited = ring(2)
    n = 400
    X = repeat(collect(co)', n)
    τx = 2 / (Jx * U0_E0)
    τE = 2 / (JE * U0_E0)
    warm = round(Int, 2τx)
    track!(X, excited, beam; nturns=warm, threaded=true, radiation_seed=2026)
    σδ2_eq = CQ_EV_M / beam.mass * beam.gamma^2 * I.I3 / (JE * I.I2)
    εx_eq = CQ_EV_M / beam.mass * beam.gamma^2 * I.I5 / (Jx * I.I2)
    rδ = Float64[]; rx = Float64[]
    for k in 1:4
        track!(X, excited, beam; nturns=25, turn=warm + 25(k - 1), threaded=true,
               radiation_seed=2026)
        t = warm + 25k
        ε = sort(eigenemittances(beam_covariance(X)))    # (y ≡ 0, x, z)
        @test ε[1] < 1e-9 * ε[2] < ε[3]
        push!(rδ, var(@view X[:, 6]) * beam.beta^2 / (σδ2_eq * (1 - exp(-2t / τE))))
        push!(rx, ε[2] / (εx_eq * (1 - exp(-2t / τx))))
    end
    @test !any(isnan, X)
    @test mean(rδ) ≈ 1 rtol=0.15
    @test mean(rx) ≈ 1 rtol=0.15
end

@testset "Optics ignore radiation" begin
    beam = Beam(3.0e9)
    h = 2pi / 12
    bend(rad) = SBend(1.0, h; polynom_b=[0.0, -h^2 / 2, 0.0, 0.0], max_order=1, rad=rad)
    function lattice(rad)
        cell = AbstractElement[
            Quadrupole(0.15, 2.0; rad=rad), Drift(0.4), bend(rad), Drift(0.2),
            Sextupole(0.1, 5.0; rad=rad), Drift(0.1), Quadrupole(0.3, -2.0; rad=rad),
            Drift(0.4), bend(rad), Drift(0.4), Quadrupole(0.15, 2.0; rad=rad)]
        elements = reduce(vcat, fill(cell, 6))
        push!(elements, RFCavity(0.0, 1.0e6, 500.0e6; energy=beam.energy, charge=beam.charge))
        return Lattice(elements; periodic=true)
    end
    # Element kinds whose TPSA compilation would dominate this file's run time
    # are checked on a short line through the finite-difference path alone.
    line(rad) = Lattice(AbstractElement[
        SBendSC(1.0, h; polynom_b=[0.0, -h^2 / 2, 0.0, 0.0], max_order=1, rad=rad),
        timed(Quadrupole(0.3, -2.0; rad=rad); k1=-2.0),
        OctupoleSC(0.1; k3=100.0, rad=rad),
        Wiggler(0.4; lw=0.1, Bmax=0.3, Nsteps=8, rad=rad,
                energy=beam.energy, mass=beam.mass, charge=beam.charge)])
    same_twiss(a, b) = all(isequal(getfield(a, f), getfield(b, f)) for f in fieldnames(typeof(a)))
    coefficients(m) = [getfield(t, :c) for t in m]     # CTPS has no value ==

    plain = lattice(0)
    r0 = SVector(1e-4, 0.0, -1e-4, 0.0, 0.0, 1e-4)
    for rad in (1, 2)
        lat = lattice(rad)
        @test linepass(lat, r0, beam) != linepass(plain, r0, beam)   # tracking radiates
        for method in (:tpsa, :fd)
            @test same_twiss(twiss(lat, beam; method, radiation_integrals=true),
                             twiss(plain, beam; method, radiation_integrals=true))
            @test getchrom(lat, beam; method) == getchrom(plain, beam; method)
        end
        @test same_twiss(twissring(lat, 0.0, 0; E0=beam.energy, m0=beam.mass),
                         twissring(plain, 0.0, 0; E0=beam.energy, m0=beam.mass))
        @test gettune(lat, beam) == gettune(plain, beam)
        @test one_turn_map(lat, beam) == one_turn_map(plain, beam)
        @test transfer_map(lat, beam; reference=r0) == transfer_map(plain, beam; reference=r0)
        @test findm66(lat, 1e-3, 0; E0=beam.energy, m0=beam.mass) ==
              findm66(plain, 1e-3, 0; E0=beam.energy, m0=beam.mass)
        @test find_closed_orbit_6d(lat, beam) == find_closed_orbit_6d(plain, beam)
        @test find_closed_orbit_4d(lat, beam; dp=1e-3) == find_closed_orbit_4d(plain, beam; dp=1e-3)
        @test coefficients(tpsa_map(lat, beam; order=2)) ==
              coefficients(tpsa_map(plain, beam; order=2))
        @test linepass(line(rad), r0, beam) != linepass(line(0), r0, beam)
        @test transfer_map(line(rad), beam; reference=r0) ==
              transfer_map(line(0), beam; reference=r0)
    end
    # Stripping keeps the lattice type, so optics code stays type-stable.
    @test (@inferred TrackPad._without_radiation(lattice(2))) isa typeof(lattice(2))
    @test TrackPad._without_radiation(plain) === plain
    # A time-dependent radiation flag cannot be cleared statically.
    @test_throws ArgumentError transfer_map(
        Lattice(AbstractElement[timed(Quadrupole(0.2, 1.0); rad=1)]), beam)
end
