using Test
using TrackPad
using StaticArrays
using LinearAlgebra
using Logging

# Regression tests for the 2026-09-04 code review corrections:
#   - kernels are generic in the coordinate type (return type follows the
#     coordinates, lost-particle branches included)
#   - Marker accepts any Real `beti`
#   - Wiggler harmonic tables are float vectors
#   - Twiss/dispersion functions resolve time-varying elements
#   - Patch pitch/tilt composition is unchanged and symplectic
#   - Beam keyword arguments accept any Real

const beam_rf = Beam(3.0e9)
const beti_rf = 1.0 / beam_rf.beta
const r_rf = SVector(1.0e-3, 2.0e-3, -7.0e-4, 8.0e-4, 3.0e-3, 2.0e-2)

const symplectic_form_rf = [
    0.0 1.0 0.0 0.0 0.0 0.0;
   -1.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 1.0 0.0 0.0;
    0.0 0.0 -1.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 1.0;
    0.0 0.0 0.0 0.0 -1.0 0.0;
]

function numerical_map_rf(elem, r0=r_rf; h=1.0e-5)
    M = zeros(6, 6)
    for j in 1:6
        e = SVector{6,Float64}(ntuple(k -> k == j ? 1.0 : 0.0, 6))
        rp = pass!(elem, r0 + h * e, beti_rf)
        rm = pass!(elem, r0 - h * e, beti_rf)
        M[:, j] = (rp - rm) / (2h)
    end
    return M
end

@testset "USE_EXACT_HAMILTONIAN is a constant with an explicit Val switch" begin
    @test isconst(TrackPad, :USE_EXACT_HAMILTONIAN)
    r = SVector(1e-3, 2e-4, -1e-3, 3e-4, 0.0, 1e-3)
    @test drift6(r, 0.5, beti_rf) == drift6(r, 0.5, beti_rf, Val(TrackPad.USE_EXACT_HAMILTONIAN))
    exact = drift6(r, 0.5, beti_rf, Val(true))
    linear = drift6(r, 0.5, beti_rf, Val(false))
    @test exact != linear
    @test exact[1] ≈ linear[1] rtol=1e-6   # agree to the px²/py² order
end

@testset "Lost-particle branch returns the coordinate type" begin
    # px > 1: pz² < 0 in the exact drift → lost. Float32 coordinates with a
    # Float64 element length must come back typed like the regular path.
    r32 = SVector{6,Float32}(0.0f0, 2.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    out = drift6(r32, 1.0, 1.0)
    @test eltype(out) === promote_type(Float32, Float64)
    @test all(isnan, out)
    ok = drift6(SVector{6,Float32}(1f-3, 1f-4, 0f0, 0f0, 0f0, 0f0), 1.0, 1.0)
    @test eltype(ok) === eltype(out)

    yrot = YRotation(0.0; angle=0.03)
    lost = pass!(yrot, SVector(0.0, 2.0, 0.0, 0.0, 0.0, 0.0), beti_rf)
    @test all(isnan, lost)
    @test eltype(lost) === Float64
end

@testset "Marker accepts any Real beti and coordinate type" begin
    m = Marker(name=:M)
    r32 = SVector{6,Float32}(1f-3, 0f0, 0f0, 0f0, 0f0, 0f0)
    @test pass!(m, r32, 1.0) === r32
    @test pass!(m, r_rf, 1.0f0) === r_rf
end

@testset "Element kernels are generic in the coordinate type" begin
    # Float32 coordinates through Float64 elements: no MethodError, and the
    # result is Float64 (the promotion), matching what Float64 coordinates give.
    r32 = SVector{6,Float32}(r_rf)
    elems = AbstractElement[
        Translation(0.0; dx=1.0e-3, dy=-2.0e-3, ds=3.0e-3),
        YRotation(0.0; angle=0.03),
        Patch(x_pitch=0.01, y_pitch=-0.02, tilt=0.1, z_offset=2e-3),
        CrabCavity(0.0; volt=2.0e5, freq=80.0e6, energy=beam_rf.energy, charge=beam_rf.charge),
        AccelCavity(0.0; volt=2.0e5, freq=80.0e6, energy=beam_rf.energy, charge=beam_rf.charge),
        LorentzBoost(0.03),
        InvLorentzBoost(0.03),
        StrongThinGaussianBeam(1e-4, 1e-3, 1e-3),
        LBend(0.9, 0.15),
    ]
    for elem in elems
        out64 = pass!(elem, r_rf, beti_rf)
        out32 = pass!(elem, r32, beti_rf)
        @test eltype(out64) === Float64
        @test eltype(out32) === Float64
        @test isapprox(out32, out64; rtol=1e-5, atol=1e-9)
    end
end

@testset "Patch pitch/tilt composition" begin
    # y_pitch is the same frame rotation as YRotation with the same angle.
    a = 0.02
    @test pass!(Patch(y_pitch=a), r_rf, beti_rf) ≈ pass!(YRotation(0.0; angle=a), r_rf, beti_rf) atol=1e-16
    # x_pitch is the y rotation with x <-> y swapped and the opposite sign.
    swap(r) = SVector(r[3], r[4], r[1], r[2], r[5], r[6])
    @test pass!(Patch(x_pitch=a), r_rf, beti_rf) ≈
          swap(pass!(YRotation(0.0; angle=-a), swap(r_rf), beti_rf)) atol=1e-16
    # Pure tilt rotates (x, y) and (px, py) together and leaves z, delta alone.
    tilted = pass!(Patch(tilt=0.1), r_rf, beti_rf)
    @test hypot(tilted[1], tilted[3]) ≈ hypot(r_rf[1], r_rf[3])
    @test hypot(tilted[2], tilted[4]) ≈ hypot(r_rf[2], r_rf[4])
    @test tilted[5] == r_rf[5] && tilted[6] == r_rf[6]
    # The full patch is symplectic.
    for patch in (Patch(x_pitch=0.01), Patch(y_pitch=-0.02),
                  Patch(x_pitch=0.01, y_pitch=-0.02, tilt=0.1, x_offset=1e-3,
                        z_offset=2e-3, t_offset=1e-12))
        M = numerical_map_rf(patch)
        @test norm(M' * symplectic_form_rf * M - symplectic_form_rf, Inf) < 1e-7
    end
end

@testset "Wiggler harmonic table is a float vector" begin
    w = Wiggler(1.2; lw=0.2, Bmax=0.8, Nsteps=8, By=[1.0, 0.5, 0.0, 1.0, 1.0, 0.3])
    @test eltype(w.By) === Float64
    @test w.By[2] == 0.5 && w.By[6] == 0.3
    out = pass!(w, r_rf, beti_rf)
    @test all(isfinite, out)
    # Integer tables still work and are converted.
    w_int = Wiggler(1.2; lw=0.2, Bmax=0.8, Nsteps=8, By=[1, 1, 0, 1, 1, 0])
    @test eltype(w_int.By) === Float64
    @test_throws ArgumentError Wiggler(1.2; lw=0.2, By=[1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
end

@testset "Twiss and dispersion resolve time-varying elements" begin
    D = Drift(2.0)
    QF = Quadrupole(0.5, 1.5)
    QD = Quadrupole(0.5, -1.5)
    t = Time()
    QF_t = timed(QF; k1=1.5 + 0.1 * sin(t))
    ring_t = Lattice(AbstractElement[QF_t, D, QD, D]; periodic=true)
    ring_0 = materialize_lattice(ring_t; time=0.0)
    tw_t = periodic_twiss(ring_t, beam_rf)
    tw_0 = periodic_twiss(ring_0, beam_rf)
    @test tw_t.betax ≈ tw_0.betax
    @test tw_t.betay ≈ tw_0.betay
    @test tw_t.tunex ≈ tw_0.tunex
    disp_t = periodic_dispersion(ring_t, beam_rf)
    disp_0 = periodic_dispersion(ring_0, beam_rf)
    @test disp_t.dx ≈ disp_0.dx
    entrance = optics4DUC(tw_0.betax[1], tw_0.alphax[1], tw_0.betay[1], tw_0.alphay[1])
    tr_t = transport_twiss(ring_t, beam_rf, entrance)
    @test tr_t.betax ≈ tw_0.betax
end

@testset "Beam keyword arguments accept any Real" begin
    b = Beam(1.0e9; charge=1, mass=M_PROTON)
    @test b.charge === 1.0
    @test b.mass === M_PROTON
    @test b.gamma ≈ (1.0e9 + M_PROTON) / M_PROTON
end

# ── Second review pass: packed-sweep curvature, apertures, turn context,
#    closed-orbit convergence ─────────────────────────────────────────────────

@testset "Bend angle sweeps drive the tracked curvature" begin
    build(angle) = Lattice(AbstractElement[
        Drift(0.5), SBend(0.9, angle; num_int_steps=4), Drift(0.5),
    ])
    gl = GPULattice(build(0.1), beam_rf; dtype=Float64)
    angles = [0.1, 0.2]
    sweep = ParamSweepLattice(gl, [(2, 2, angles)])

    r0 = [1.0e-3 2.0e-4 -5.0e-4 1.0e-4 0.0 1.0e-3]
    out = zeros(Float64, length(angles), 6)
    param_sweep_linepass!(out, repeat(r0, length(angles), 1), sweep)

    # Every configuration must agree with a lattice rebuilt at that angle.
    for (k, angle) in enumerate(angles)
        reference = copy(r0)
        batch_linepass!(reference, GPULattice(build(angle), beam_rf; dtype=Float64))
        @test out[k, :] ≈ vec(reference) atol=1e-14
    end
    # ... and the sweep must actually move the orbit.
    @test !isapprox(out[1, :], out[2, :]; atol=1e-8)

    # Slot 8 holds the derived curvature; sweeping it would do nothing.
    @test_throws ArgumentError ParamSweepLattice(gl, [(2, 8, angles)])
end

@testset "Apertures are honored by every CPU tracking entry point" begin
    aperture = Drift(1.0; r_apertures=[-2.0e-3, 2.0e-3, -2.0e-3, 2.0e-3, 0.0, 0.0])
    lat = Lattice(AbstractElement[aperture, Drift(1.0), Quadrupole(0.3, 1.2)])
    # x = 1.5 mm + 1.0 mrad * 1 m = 2.5 mm, outside the 2 mm half-aperture.
    r_lost = SVector(1.5e-3, 1.0e-3, 0.0, 0.0, 0.0, 0.0)

    coords = reshape(collect(r_lost), 1, 6)
    flags = zeros(Int, 1)
    linepass!(coords, lat, beam_rf, flags)
    @test flags[1] == 1                       # reference behaviour

    @test all(isnan, linepass(lat, r_lost, beam_rf))
    batch = reshape(collect(r_lost), 1, 6)
    cpu_batch_linepass!(batch, lat, beam_rf)
    @test all(isnan, batch)

    # Opting out reproduces the previous aperture-blind behaviour.
    @test all(isfinite, linepass(lat, r_lost, beam_rf; check_apertures=false))

    # A particle inside the aperture is untouched by the check.
    r_ok = SVector(1.0e-4, 1.0e-5, 0.0, 0.0, 0.0, 0.0)
    @test linepass(lat, r_ok, beam_rf) ≈ linepass(lat, r_ok, beam_rf; check_apertures=false)
    inside = reshape(collect(r_ok), 1, 6)
    cpu_batch_linepass!(inside, lat, beam_rf)
    @test all(isfinite, inside)
end

@testset "CPU batch tracking advances the turn context" begin
    kick = 1.0e-5
    corrector = timed(Corrector(0.0, 0.0, 0.0); xkick = kick * Turn())
    ring = Lattice(AbstractElement[corrector, Drift(0.5)]; periodic=true)
    nturns = 3

    coords = zeros(1, 6)
    cpu_batch_linepass!(coords, ring, beam_rf, nturns)

    reference = zeros(1, 6)
    flags = zeros(Int, 1)
    ringpass!(reference, ring, beam_rf, flags, nturns)

    @test coords ≈ reference atol=1e-15
    @test coords[1, 2] ≈ 3 * kick atol=1e-15   # turns 0 + 1 + 2
    # An explicit starting turn is honored as well.
    shifted = zeros(1, 6)
    cpu_batch_linepass!(shifted, ring, beam_rf, 1; turn=5)
    @test shifted[1, 2] ≈ 5 * kick atol=1e-15
end

@testset "Closed-orbit solvers report non-convergence" begin
    # A net steering kick with no focusing has no closed orbit: every turn adds
    # the same amount to px and nothing restores it, so px_out - px is a nonzero
    # constant for every starting point. (A pure frame shift is NOT such a case:
    # a particle at px = dx/L absorbs the offset over the following drift.)
    drifting = Lattice(AbstractElement[Corrector(0.0, 1.0e-5, 0.0), Drift(1.0)];
                       periodic=true)
    @test_throws ErrorException find_closed_orbit_4d(drifting, beam_rf)
    @test_throws ErrorException find_closed_orbit_6d(drifting, beam_rf)

    orbit = with_logger(NullLogger()) do
        find_closed_orbit_4d(drifting, beam_rf; strict=false)
    end
    @test length(orbit) == 4

    # A ring that does have a closed orbit still returns it.
    ring = Lattice(AbstractElement[
        Quadrupole(0.5, 1.5), Drift(2.0), Quadrupole(0.5, -1.5), Drift(2.0),
    ]; periodic=true)
    @test find_closed_orbit_4d(ring, beam_rf) ≈ zeros(4) atol=1e-12
    @test find_closed_orbit_6d(ring, beam_rf) ≈ zeros(6) atol=1e-12
end
