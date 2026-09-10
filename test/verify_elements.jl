using Test
using TrackPad
using LinearAlgebra
using StaticArrays
import JuTrack
Base.include(@__MODULE__, joinpath(@__DIR__, "convention_helpers.jl"))

const PARITY_ATOL = 1e-15
# JuTrack uses mutually inconsistent finite-beta longitudinal normalizations
# across element families. Exact parity is meaningful only in the common
# ultrarelativistic limit; canonical finite-beta behavior is tested below.
const ENERGY_VAL = 1.0e15

# Deterministic initial coordinates shared across element parity checks.
particles_initial = [
    1.0e-4   2.0e-4   3.0e-4  -1.0e-4   5.0e-5   2.0e-4
   -2.2e-4  1.7e-4  -1.1e-4   2.3e-4  -7.0e-5   1.0e-4
    3.5e-4  -2.1e-4  8.0e-5  -1.8e-4   1.4e-4  -2.6e-4
   -4.0e-4  2.9e-4   1.6e-4   9.0e-5  -1.2e-4   3.1e-4
    5.2e-4  -3.3e-4 -2.4e-4   1.1e-4   2.6e-4  -3.7e-4
   -6.1e-4  4.4e-4   2.7e-4  -2.5e-4  -3.0e-4   4.5e-4
    7.0e-4  -5.2e-4 -3.1e-4   3.4e-4   3.3e-4  -5.0e-4
   -8.0e-4  6.0e-4   3.9e-4  -4.2e-4  -3.8e-4   5.8e-4
    9.1e-4  -6.7e-4 -4.6e-4   5.1e-4   4.4e-4  -6.3e-4
   -9.8e-4  7.3e-4   5.3e-4  -5.9e-4  -4.9e-4   7.1e-4
]

function _track_tp(elem; energy::Float64 = ENERGY_VAL, particles = particles_initial)
    coords = copy(particles)
    lost_flags = zeros(Int, size(coords, 1))
    linepass!(coords, Lattice([elem]), Beam(energy), lost_flags)
    return coords
end

function _track_jt(elem; energy::Float64 = ENERGY_VAL, particles = particles_initial, current::Float64 = 0.0)
    beam = JuTrack.Beam(flip_longitudinal_coordinate(particles), energy = energy, mass = JuTrack.m_e, current = current)
    JuTrack.linepass!([elem], beam)
    return flip_longitudinal_coordinate(beam.r)
end

function _assert_parity(name::AbstractString, tp_elem, jt_elem; energy::Float64 = ENERGY_VAL, particles = particles_initial, current::Float64 = 0.0)
    tp = _track_tp(tp_elem; energy = energy, particles = particles)
    jt = _track_jt(jt_elem; energy = energy, particles = particles, current = current)
    diff_norm = norm(tp - jt)
    diff_max = maximum(abs.(tp - jt))
    @testset "$name" begin
        @test diff_norm < PARITY_ATOL
        @test diff_max < PARITY_ATOL
        @test isapprox(tp, jt, atol = PARITY_ATOL)
    end
end

@testset "Element JuTrack Parity" begin
    old_exact_beti = JuTrack.use_exact_beti
    JuTrack.use_exact_beti = 1
    try
        # Core linear/multipole/bend elements.
        _assert_parity("Marker", Marker(), JuTrack.MARKER())
        _assert_parity("Drift", Drift(0.7), JuTrack.DRIFT(len = 0.7))
        _assert_parity("Quadrupole", Quadrupole(0.4, 1.3; num_int_steps = 10), JuTrack.KQUAD(len = 0.4, k1 = 1.3, NumIntSteps = 10))
        _assert_parity("Sextupole", Sextupole(0.4, 2.0; num_int_steps = 10), JuTrack.KSEXT(len = 0.4, k2 = 2.0, NumIntSteps = 10))
        _assert_parity("Octupole", Octupole(0.4, -3.0; num_int_steps = 10), JuTrack.KOCT(len = 0.4, k3 = -3.0, NumIntSteps = 10))
        _assert_parity(
            "ThinMultipole",
            ThinMultipole(0.0, [0.0, 0.0, 0.0, 0.0], [0.0, 0.2, -0.1, 0.05]; max_order = 3),
            JuTrack.thinMULTIPOLE(len = 0.0, PolynomA = [0.0, 0.0, 0.0, 0.0], PolynomB = [0.0, 0.2, -0.1, 0.05], MaxOrder = 3),
        )
        _assert_parity("SBend", SBend(0.9, 0.15, 0.03, 0.02; num_int_steps = 10), JuTrack.SBEND(len = 0.9, angle = 0.15, e1 = 0.03, e2 = 0.02, NumIntSteps = 10))
        _assert_parity("RBend", RBend(0.9, 0.15; num_int_steps = 10), JuTrack.RBEND(len = 0.9, angle = 0.15, NumIntSteps = 10))
        _assert_parity("ExactSBend", ExactSBend(0.9, 0.15, 0.03, 0.02; num_int_steps = 10), JuTrack.ESBEND(len = 0.9, angle = 0.15, e1 = 0.03, e2 = 0.02, NumIntSteps = 10))
        _assert_parity("ERBend", ERBend(0.9, 0.15; num_int_steps = 10), JuTrack.ERBEND(len = 0.9, angle = 0.15, NumIntSteps = 10))

        # RF/cavity and related maps.
        _assert_parity(
            "RFCavity",
            RFCavity(0.3, 2.0e6, 500.0e6, 0.002; h = 1200.0, philag = 0.3, energy = ENERGY_VAL),
            JuTrack.RFCA(len = 0.3, volt = 2.0e6, freq = 500.0e6, lag = 0.002, h = 1200.0, philag = 0.3, energy = ENERGY_VAL),
        )
        _assert_parity(
            "CrabCavity",
            CrabCavity(0.3; volt = 2.0e6, freq = 500.0e6, phi = 0.1, energy = ENERGY_VAL),
            JuTrack.CRABCAVITY(len = 0.3, volt = 2.0e6, freq = 500.0e6, phi = 0.1, energy = ENERGY_VAL),
        )
        _assert_parity(
            "AccelCavity",
            AccelCavity(0.0; volt = 2.5e6, freq = 500.0e6, h = 1200.0, phis = 0.2, energy = ENERGY_VAL),
            JuTrack.AccelCavity(len = 0.0, volt = 2.5e6, freq = 500.0e6, h = 1200.0, phis = 0.2),
        )
        # Auxiliary helpers.
        _assert_parity("Solenoid", Solenoid(0.5, 0.8), JuTrack.SOLENOID(len = 0.5, ks = 0.8))
        _assert_parity("Corrector", Corrector(0.4, 1.5e-4, -2.2e-4), JuTrack.CORRECTOR(len = 0.4, xkick = 1.5e-4, ykick = -2.2e-4))
        _assert_parity("HKicker", HKicker(L = 0.0, xkick = 2.5e-4), JuTrack.HKICKER(len = 0.0, xkick = 2.5e-4))
        _assert_parity("VKicker", VKicker(L = 0.0, ykick = -1.5e-4), JuTrack.VKICKER(len = 0.0, ykick = -1.5e-4))
        _assert_parity("YRotation", YRotation(0.0; angle = 0.02), JuTrack.YROTATION(len = 0.0, angle = 0.02))
        _assert_parity("Wiggler", Wiggler(1.2; lw = 0.2, Bmax = 0.8, Nsteps = 8), JuTrack.WIGGLER(len = 1.2, lw = 0.2, Bmax = 0.8, Nsteps = 8))
        # Vertical wiggler harmonics need kx != 0 (the field decays along x);
        # degenerate all-zero wave-vector blocks produce NaN in both codes.
        _assert_parity(
            "Wiggler vertical harmonics",
            Wiggler(1.2; lw = 0.2, Bmax = 0.8, Nsteps = 8, By = Int[], Bx = [1, 1, 1, 0, 1, 0]),
            JuTrack.WIGGLER(len = 1.2, lw = 0.2, Bmax = 0.8, Nsteps = 8, By = Int[], Bx = [1, 1, 1, 0, 1, 0]),
        )
        _assert_parity(
            "Wiggler mixed harmonics",
            Wiggler(1.2; lw = 0.2, Bmax = 0.8, Nsteps = 8, By = [1, 1, 0, 1, 1, 0], Bx = [1, 2, 1, 0, 1, 0]),
            JuTrack.WIGGLER(len = 1.2, lw = 0.2, Bmax = 0.8, Nsteps = 8, By = [1, 1, 0, 1, 1, 0], Bx = [1, 2, 1, 0, 1, 0]),
        )

        # Translation longitudinal shift follows TrackPad's negated z axis and
        # is drift-consistent; with dx = dy = 0 it must equal a Drift of ds
        # exactly. JuTrack's TRANSLATION mixes an opposing x-sign with its
        # c(t-t0) update and is not drift-consistent, so only the in-package
        # invariant is asserted here (see conventions.md).
        @testset "Translation drift consistency" begin
            ds = 3.0e-3
            beam_tb = Beam(ENERGY_VAL)
            coords_t = copy(particles_initial)
            lost_t = zeros(Int, size(coords_t, 1))
            linepass!(coords_t, Lattice([Translation(0.0; dx = 1.0e-3, dy = -2.0e-3, ds = ds)]),
                      beam_tb, lost_t)
            coords_d = copy(particles_initial)
            linepass!(coords_d, Lattice([Drift(ds)]), Beam(ENERGY_VAL), zeros(Int, size(coords_d, 1)))
            # Remove the lateral origin offsets analytically: x -= dx, y -= dy.
            shifted = copy(coords_d)
            shifted[:, 1] .-= 1.0e-3
            shifted[:, 3] .-= -2.0e-3
            @test coords_t ≈ shifted atol = 1.0e-15
        end

        # Bend multipoles must raise the kick expansion order automatically
        # (JuTrack raises MaxOrder from nonzero PolynomB entries).
        @test SBend(0.9, 0.15; polynom_b = [0.0, 0.3, 0.0, 0.0]).max_order == 1
        @test SBend(0.9, 0.15; polynom_b = [0.0, 0.3, 0.05, 0.0]).max_order == 2
        @test SBend(0.9, 0.15; polynom_b = [0.0, 0.3, 0.0, 0.7]).max_order == 3
        @test SBend(0.9, 0.15; max_order = 2).max_order == 2  # user value kept
        @test ExactSBend(0.9, 0.15; polynom_b = [0.0, 0.0, 0.0, 0.7]).max_order == 3
        @test SBendSC(1.0, 0.2; polynom_b = [0.0, 0.3, 0.0, 0.0]).max_order == 1
        _assert_parity(
            "SBend gradient auto-order",
            SBend(0.9, 0.15, 0.03, 0.02; num_int_steps = 10,
                  polynom_b = [0.0, 0.3, 0.0, 0.0]),
            JuTrack.SBEND(len = 0.9, angle = 0.15, e1 = 0.03, e2 = 0.02,
                          NumIntSteps = 10, PolynomB = [0.0, 0.3, 0.0, 0.0]),
        )
        _assert_parity(
            "RBend gradient auto-order",
            RBend(0.9, 0.15; num_int_steps = 10, polynom_b = [0.0, 0.25, 0.0, 0.0]),
            JuTrack.RBEND(len = 0.9, angle = 0.15, NumIntSteps = 10,
                          PolynomB = [0.0, 0.25, 0.0, 0.0]),
        )

        # Forest (13.29) multipole entrance/exit fringes.
        _assert_parity(
            "Quadrupole fringes",
            Quadrupole(0.4, 1.3; num_int_steps = 10, fringe_entrance = 1, fringe_exit = 1),
            JuTrack.KQUAD(len = 0.4, k1 = 1.3, NumIntSteps = 10,
                          FringeQuadEntrance = 1, FringeQuadExit = 1),
        )
        _assert_parity(
            "Sextupole fringes",
            Sextupole(0.4, 2.0; num_int_steps = 10, fringe_entrance = 1),
            JuTrack.KSEXT(len = 0.4, k2 = 2.0, NumIntSteps = 10, FringeQuadEntrance = 1),
        )
        _assert_parity(
            "Octupole fringes",
            Octupole(0.4, -3.0; num_int_steps = 10, fringe_exit = 1),
            JuTrack.KOCT(len = 0.4, k3 = -3.0, NumIntSteps = 10, FringeQuadExit = 1),
        )
        _assert_parity(
            "ThinMultipole fringes",
            ThinMultipole(0.0, [0.0, 0.0, 0.0, 0.0], [0.0, 0.2, -0.1, 0.05];
                          max_order = 3, fringe_entrance = 1),
            JuTrack.thinMULTIPOLE(len = 0.0, PolynomA = [0.0, 0.0, 0.0, 0.0],
                                  PolynomB = [0.0, 0.2, -0.1, 0.05], MaxOrder = 3,
                                  FringeQuadEntrance = 1),
        )
        _assert_parity(
            "SBend quad-fringe gates",
            SBend(0.9, 0.15, 0.03, 0.02; num_int_steps = 10,
                  fringe_quad_entrance = 1, fringe_quad_exit = 1),
            JuTrack.SBEND(len = 0.9, angle = 0.15, e1 = 0.03, e2 = 0.02,
                          NumIntSteps = 10, FringeQuadEntrance = 1, FringeQuadExit = 1),
        )
        _assert_parity(
            "ExactSBend quad-fringe ordering",
            ExactSBend(0.9, 0.15, 0.03, 0.02; num_int_steps = 10,
                       fringe_quad_entrance = 1, fringe_quad_exit = 1),
            JuTrack.ESBEND(len = 0.9, angle = 0.15, e1 = 0.03, e2 = 0.02,
                           NumIntSteps = 10, FringeQuadEntrance = 1,
                           FringeQuadExit = 1),
        )
        _assert_parity(
            "Sextupole kick angle with fringes",
            Sextupole(0.4, 2.0; num_int_steps = 10,
                      kick_angle = [1.2e-4, -0.8e-4],
                      fringe_entrance = 1, fringe_exit = 1),
            JuTrack.KSEXT(len = 0.4, k2 = 2.0, NumIntSteps = 10,
                          KickAngle = [1.2e-4, -0.8e-4],
                          FringeQuadEntrance = 1, FringeQuadExit = 1),
        )
        _assert_parity(
            "Octupole kick angle with fringes",
            Octupole(0.4, -3.0; num_int_steps = 10,
                     kick_angle = [-0.7e-4, 1.1e-4],
                     fringe_entrance = 1, fringe_exit = 1),
            JuTrack.KOCT(len = 0.4, k3 = -3.0, NumIntSteps = 10,
                         KickAngle = [-0.7e-4, 1.1e-4],
                         FringeQuadEntrance = 1, FringeQuadExit = 1),
        )

        # Space-charge canonical family.
        _assert_parity("DriftSC", DriftSC(0.5; a = 0.01, b = 0.02, Nl = 12, Nm = 14, Nsteps = 2), JuTrack.DRIFT_SC(len = 0.5, a = 0.01, b = 0.02, Nl = 12, Nm = 14, Nsteps = 2))
        _assert_parity("QuadrupoleSC", QuadrupoleSC(0.4; k1 = 0.0, a = 0.01, b = 0.02), JuTrack.KQUAD_SC(len = 0.4, k1 = 0.0, a = 0.01, b = 0.02))
        _assert_parity("SextupoleSC", SextupoleSC(0.3; k2 = 2.4, a = 0.01, b = 0.02), JuTrack.KSEXT_SC(len = 0.3, k2 = 2.4, a = 0.01, b = 0.02))
        _assert_parity("OctupoleSC", OctupoleSC(0.2; k3 = 3.6, a = 0.01, b = 0.02), JuTrack.KOCT_SC(len = 0.2, k3 = 3.6, a = 0.01, b = 0.02))
        _assert_parity("SBendSC", SBendSC(1.0, 0.0, 0.0, 0.0; a = 0.01, b = 0.02), JuTrack.SBEND_SC(len = 1.0, angle = 0.0, e1 = 0.0, e2 = 0.0, a = 0.01, b = 0.02))
        _assert_parity("RBendSC", RBendSC(1.2, 0.0), JuTrack.RBEND_SC(len = 1.2, angle = 0.0))
        _assert_parity("LBend", LBend(0.7, 0.0; K = 0.0), JuTrack.LBEND(len = 0.7, angle = 0.0, K = 0.0))
        _assert_parity("SpaceCharge", SpaceCharge(0.9; effective_len = 0.4, Nl = 11, Nm = 9, a = 0.015, b = 0.017), JuTrack.SPACECHARGE(len = 0.9, effective_len = 0.4, Nl = 11, Nm = 9, a = 0.015, b = 0.017))

        # JuTrack has a Float64 collective RLC pass, but its Beam-owned grid,
        # normalization, and longitudinal coordinate differ from TrackPad's.
        # A zero wake therefore checks tracking plumbing, while the Green
        # functions themselves are compared directly at nonzero strength.
        tp_rlc_zero = LongitudinalRLCWake(freq = 1.0e9, Rshunt = 0.0,
                                          Q0 = 1.2, scale = 1.0)
        jt_rlc_zero = JuTrack.LongitudinalRLCWake(freq = 1.0e9,
                                                  Rshunt = 0.0, Q0 = 1.2)
        _assert_parity("LongitudinalRLCWake", tp_rlc_zero, jt_rlc_zero)
        tp_rlc = LongitudinalRLCWake(freq = 1.0e9, Rshunt = 1.0e6, Q0 = 1.2)
        jt_rlc = JuTrack.LongitudinalRLCWake(freq = 1.0e9,
                                             Rshunt = 1.0e6, Q0 = 1.2)
        for t in (-2.0e-9, -1.0e-10, 0.0, 1.0e-10)
            @test wakefieldfunc_RLCWake(tp_rlc, t) ≈
                  JuTrack.wakefieldfunc_RLCWake(jt_rlc, t) rtol = 1.0e-15
        end
        # StrongGaussianBeam is not compared with JuTrack: JuTrack's beam-beam
        # code is a placeholder. See test/verify_beambeam.jl.
    finally
        JuTrack.use_exact_beti = old_exact_beti
    end
end

@testset "Wiggler constructor validation" begin
    @test_throws ArgumentError Wiggler(1.2)
    @test_throws ArgumentError Wiggler(1.2; lw = 0.2, Nsteps = 0)
    @test_throws ArgumentError Wiggler(1.2; lw = 0.2, energy = M_ELECTRON)
    @test_throws ArgumentError Wiggler(1.2; lw = 0.2, By = [1, 1, 0])
    @test_throws ArgumentError Wiggler(1.2; lw = 0.2, Bx = [1, 1, 1])
    @test_throws ArgumentError Wiggler(
        1.2; lw = 0.2, Bx = [1, 1, 0, 0, 1, 0])
    @test_throws ArgumentError Wiggler(
        1.2; lw = 0.2, Bx = [1, 1, 1, 0, 0, 0])
    @test_throws ArgumentError Wiggler(
        1.2; lw = 0.2, By = [1, 1, 0, 0, 1, 0])
    @test_throws ArgumentError Wiggler(
        1.2; lw = 0.2, By = [1, 1, 0, 1, 0, 0])
end

@testset "Canonical longitudinal slip map" begin
    rf = AccelCavity(0.0; freq=500.0e6, h=1200.0, energy=ENERGY_VAL)
    elem = LongitudinalRFMap(2.0e-4, rf)
    beam = Beam(ENERGY_VAL)
    delta_e = 2.0e-3
    input = SVector(0.0, 0.0, 0.0, 0.0, 1.0e-3, delta_e)
    output = pass!(elem, input, inv(beam.beta))
    eta = elem.alphac - (1 - beam.beta^2)
    expected_z = input[5] - (2π * rf.h * eta / (rf.k * beam.beta)) * delta_e
    @test output[5] ≈ expected_z atol=1.0e-15
    @test output[5] < input[5]
end

@testset "Finite-beta canonical maps" begin
    beam = Beam(50.0e6; mass=M_PROTON, charge=1.0)
    beti = inv(beam.beta)
    reference = SVector(1.0e-3, 2.0e-3, -7.0e-4, 8.0e-4, 3.0e-3, 2.0e-2)
    symplectic_form = zeros(6, 6)
    for i in (1, 3, 5)
        symplectic_form[i, i + 1] = 1
        symplectic_form[i + 1, i] = -1
    end

    function numerical_map(elem; h=1.0e-5)
        map = zeros(6, 6)
        for j in 1:6
            offset = zeros(6)
            offset[j] = h
            plus = pass!(elem, reference + SVector{6}(offset), beti)
            minus = pass!(elem, reference - SVector{6}(offset), beti)
            map[:, j] = (plus - minus) / (2h)
        end
        return map
    end

    elements = AbstractElement[
        Drift(0.7),
        SBend(0.9, 0.15; num_int_steps=10),
        ExactSBend(0.9, 0.15; num_int_steps=10),
        Solenoid(0.5, 0.8),
        Corrector(0.4, 1.5e-4, -2.2e-4),
        RFCavity(0.0, 2.0e5, 80.0e6, 0.01;
                  energy=beam.energy, charge=beam.charge),
        CrabCavity(0.0; volt=2.0e5, freq=80.0e6,
                   energy=beam.energy, charge=beam.charge),
        AccelCavity(0.0; volt=2.0e5, freq=80.0e6,
                    energy=beam.energy, charge=beam.charge),
        Translation(0.0; dx=1.0e-3, dy=-2.0e-3, ds=3.0e-3),
        Patch(z_offset=3.0e-3, t_offset=2.0e-12),
        YRotation(0.0; angle=0.03),
        LorentzBoost(0.03),
        InvLorentzBoost(0.03),
    ]
    for elem in elements
        map = numerical_map(elem)
        @test norm(map' * symplectic_form * map - symplectic_form, Inf) < 1.0e-7
    end

    p0c = beam.beta * (beam.energy + beam.mass)
    lag = TrackPad.C_LIGHT / (4 * 80.0e6)
    cavity = RFCavity(0.0, 2.0e5, 80.0e6, lag;
                      energy=beam.energy, charge=beam.charge)
    output = pass!(cavity, zero(reference), beti)
    @test output[6] ≈ cavity.charge * cavity.volt / p0c rtol=1.0e-14

    boosted = pass!(LorentzBoost(0.03), reference, beti)
    restored = pass!(InvLorentzBoost(0.03), boosted, beti)
    @test restored ≈ reference atol=1.0e-15
end

# Reference implementation of the collective wake kick: cloud-in-cell
# deposition about the nearest bin center on a one-bin-padded grid, discrete
# convolution against the Green function, bin-edge reconstruction of the
# potential, and per-particle linear interpolation. Mirrors src/tracking.jl
# so the tests pin the documented numerics.
function _wake_reference_kicks(wake::Union{LongitudinalRLCWake, LongitudinalWake},
                               zs::AbstractVector{Float64})
    nb = wake.nbins
    green(t) = wake isa LongitudinalRLCWake ?
        wakefieldfunc_RLCWake(wake, t) : wakefieldfunc(wake, t)
    zmin, zmax = extrema(zs)
    if zmax > zmin
        span = zmax - zmin
        dz_pad = span / nb
        z0 = zmin - dz_pad
        dz = (span + 2 * dz_pad) / nb
        hist = zeros(nb)
        bins = zeros(Int, length(zs))
        if nb == 1
            hist[1] = length(zs)
            bins .= 1
        else
            for (i, z) in enumerate(zs)
                s = (z - z0) / dz
                m = clamp(round(Int, s - 0.5) + 1, 1, nb)
                d = s - (m - 0.5)
                bins[i] = floor(Int, s) + 1
                hist[m] += 1 - abs(d)
                if d > 0
                    hist[m+1] += d
                elseif d < 0
                    hist[m-1] -= d
                end
            end
        end
        zc(k) = z0 + (k - 0.5) * dz
        pot = [sum(hist[j] * green((zc(k) - zc(j)) / TrackPad.C_LIGHT)
                   for j in 1:nb) for k in 1:nb]
        edges = zeros(nb + 1)
        if nb == 1
            edges .= pot[1]
        elseif nb == 2
            edges[1] = (3 * pot[1] - pot[2]) / 2
            edges[2] = (pot[1] + pot[2]) / 2
            edges[3] = (3 * pot[2] - pot[1]) / 2
        else
            for k in 2:nb
                edges[k] = (pot[k - 1] + pot[k]) / 2
            end
            edges[1] = 2 * edges[2] - edges[3]
            edges[end] = 2 * edges[end - 1] - edges[end - 2]
        end
        return [-wake.scale *
                (edges[b] + (edges[b + 1] - edges[b]) * (z - (z0 + (b - 1) * dz)) / dz)
                for (z, b) in zip(zs, bins)]
    else
        return fill(-wake.scale * length(zs) * green(0.0), length(zs))
    end
end

@testset "Longitudinal wake convolution" begin
    # The wake Green function must be convolved with the histogram of r[5],
    # not evaluated at a particle's own coordinate.
    beam = Beam(ENERGY_VAL)

    @testset "constructor validation" begin
        @test_throws ArgumentError LongitudinalRLCWake(nbins = 0)
        @test_throws ArgumentError LongitudinalRLCWake(freq = 0.0)
        @test_throws ArgumentError LongitudinalRLCWake(Rshunt = -1.0)
        @test_throws ArgumentError LongitudinalRLCWake(Q0 = 0.5)
        @test_throws ArgumentError LongitudinalRLCWake(Q0 = 0.4)
        @test_throws ArgumentError LongitudinalRLCWake(scale = Inf)
        @test_throws ArgumentError LongitudinalWake([0.0, 1.0], [1.0, 2.0]; nbins = 0)
        @test_throws ArgumentError LongitudinalWake([1.0, 2.0], [1.0, 2.0])
        @test_throws ArgumentError LongitudinalWake([0.0, 0.0], [1.0, 2.0])
        @test_throws ArgumentError LongitudinalWake([0.0, 2.0, 1.0], [1.0, 2.0, 3.0])
        @test_throws ArgumentError LongitudinalWake([0.0, Inf], [1.0, 2.0])
        @test_throws ArgumentError LongitudinalWake([0.0, 1.0], [1.0, NaN])
        @test_throws ArgumentError LongitudinalWake([0.0, 1.0], [1.0, 2.0]; fliphalf = 1.0)
        @test_throws ArgumentError LongitudinalWake([0.0, 1.0], [1.0, 2.0]; scale = NaN)
    end

    @testset "RLC convolution with cloud-in-cell deposition" begin
        wake = LongitudinalRLCWake(freq = 1.0e9, Rshunt = 1.0e6, Q0 = 10.0,
                                   scale = 2.5, nbins = 4)
        zs = [0.0, 1.5e-3, 2.5e-3, 4.0e-3]
        coords = [zeros(4) zeros(4) zeros(4) zeros(4) zs fill(0.1, 4)]
        lost = zeros(Int, 4)
        linepass!(coords, Lattice([wake]), beam, lost)

        expected = 0.1 .+ _wake_reference_kicks(wake, zs)
        @test coords[:, 6] ≈ expected atol = 1.0e-12 rtol = 1.0e-12
        # Trailing particles lose more energy than the leading one.
        @test coords[4, 6] > coords[1, 6]
    end

    @testset "causality and self term" begin
        # NOTE: with only a handful of macroparticles the grid-based kick of
        # an isolated edge particle is smoothed at the bin scale (as in
        # JuTrack's histogram scheme), so sparse-bunch kicks are only pinned
        # against the reference implementation, not against W(0).
        wake = LongitudinalRLCWake(freq = 1.0e9, Rshunt = 1.0e6, Q0 = 10.0,
                                   scale = 1.0, nbins = 4)
        w0 = wakefieldfunc_RLCWake(wake, 0.0)
        # Leader alone: degenerate single-position bunch -> every particle
        # sees N * W(0) exactly, independent of any grid choice.
        leader_only = reshape([0.0, 0.0, 0.0, 0.0, 4.0e-3, 0.05], 1, 6)
        linepass!(leader_only, Lattice([wake]), beam, zeros(Int, 1))
        @test leader_only[1, 6] ≈ 0.05 - w0 atol = 1.0e-12

        # Adding a trailer behind must give the trailer extra energy loss
        # while the leading particle keeps the smaller shared-grid kick.
        pair = [0.0 0.0 0.0 0.0 4.0e-3  0.05
                0.0 0.0 0.0 0.0 3.9e-3  0.05]
        linepass!(pair, Lattice([wake]), beam, zeros(Int, 2))
        expected = 0.05 .+ _wake_reference_kicks(wake, [4.0e-3, 3.9e-3])
        @test pair[:, 6] ≈ expected atol = 1.0e-12 rtol = 1.0e-12
        @test pair[2, 6] < pair[1, 6] < 0.05
    end

    @testset "tabulated wake convolution" begin
        tw = LongitudinalWake([0.0, 5.0e-9], [2.0, 1.0]; scale = 1.5, nbins = 2)
        @test wakefieldfunc(tw, -10.0e-9) ≈ 0.0 atol = 1.0e-15
        # Separation 1 m -> delay 1/c ≈ 3.33 ns, inside the table.
        # z is positive for early particles: the z = 1 particle leads.
        coords = [0.0 0.0 0.0 0.0 0.0 0.2
                  0.0 0.0 0.0 0.0 1.0 0.2]
        linepass!(coords, Lattice(AbstractElement[tw]), beam, zeros(Int, 2))
        expected = 0.2 .+ _wake_reference_kicks(tw, [0.0, 1.0])
        @test coords[:, 6] ≈ expected atol = 1.0e-12 rtol = 1.0e-12
        # The trailer feels its own bin plus the leading bin; the leader has
        # no sources ahead of it and loses less energy.
        @test coords[1, 6] < coords[2, 6] < 0.2 + 1.0e-12
    end

    @testset "sparse and small-bin grids" begin
        # A constant causal Green function makes every target grid point at or
        # behind a populated source nonzero, including currently empty bins.
        constant_wake(nb) = LongitudinalWake(
            [0.0, 10.0e-9], [1.0, 1.0]; scale = 1.0, nbins = nb)
        zs = [0.0, 1.0]

        sparse = [zeros(2) zeros(2) zeros(2) zeros(2) zs zeros(2)]
        linepass!(sparse, Lattice([constant_wake(8)]), beam, zeros(Int, 2))
        @test sparse[:, 6] ≈ _wake_reference_kicks(constant_wake(8), zs)

        one_bin = [zeros(2) zeros(2) zeros(2) zeros(2) zs zeros(2)]
        linepass!(one_bin, Lattice([constant_wake(1)]), beam, zeros(Int, 2))
        @test one_bin[:, 6] == [-2.0, -2.0]

        two_bins = [zeros(2) zeros(2) zeros(2) zeros(2) zs zeros(2)]
        linepass!(two_bins, Lattice([constant_wake(2)]), beam, zeros(Int, 2))
        @test all(isfinite, two_bins[:, 6])
        @test two_bins[:, 6] ≈ _wake_reference_kicks(constant_wake(2), zs)
    end

    @testset "bin resolution convergence" begin
        # A smooth RLC Green function on a uniform bunch: refining the
        # histogram must converge the interpolated kicks.
        zs = collect(range(0.0, 4.0e-3; length = 32))
        kicks(nb) = _wake_reference_kicks(
            LongitudinalRLCWake(freq = 1.0e9, Rshunt = 1.0e6, Q0 = 10.0,
                                scale = 1.0, nbins = nb), zs)
        fine = kicks(128)
        d_coarse = maximum(abs.(kicks(4) .- fine))
        d_medium = maximum(abs.(kicks(16) .- fine))
        @test d_medium <= d_coarse
        @test d_medium <= 5.0e-2 * maximum(abs.(fine))
    end

    @testset "physical_wake_scale" begin
        beam_e = Beam(ENERGY_VAL)
        bunch_charge = -1.0e-9
        nmacro = 1000
        p0c = beam_e.beta * (beam_e.energy + beam_e.mass)
        sc = physical_wake_scale(beam_e, bunch_charge, nmacro)
        # W [V/C] * Qmacro [C] * q/e gives an energy change in eV.
        expected = beam_e.charge * (bunch_charge / nmacro) / p0c
        @test sc ≈ expected rtol = 1.0e-15
        @test sc > 0
        @test_throws ArgumentError physical_wake_scale(beam_e, 1.0e-9, 0)
        @test_throws ArgumentError physical_wake_scale(beam_e, Inf, 1000)
        @test_throws ArgumentError physical_wake_scale(Beam(0.0), -1.0e-9, 1000)
        @test_throws ArgumentError physical_wake_scale(
            Beam(ENERGY_VAL; charge = Inf), -1.0e-9, 1000)
        @test physical_wake_scale(beam_e, 2 * bunch_charge, nmacro) ≈ 2 * sc

        # Like-sign electron and positive-charge bunches both decelerate for a
        # positive wake; inconsistent source/test signs reverse the kick.
        beam_p = Beam(ENERGY_VAL; charge = 1.0)
        @test physical_wake_scale(beam_p, -bunch_charge, nmacro) ≈ sc
        @test physical_wake_scale(beam_p, bunch_charge, nmacro) ≈ -sc

        wake = LongitudinalRLCWake(freq = 1.0e9, Rshunt = 1.0e6, Q0 = 10.0,
                                   scale = sc, nbins = 64)
        coords = [zeros(2) zeros(2) zeros(2) zeros(2) [3.9e-3, 4.0e-3] zeros(2)]
        linepass!(coords, Lattice([wake]), beam_e, zeros(Int, 2))
        @test coords[1, 6] < 0.0  # trailer loses energy
        @test coords[2, 6] > coords[1, 6]
    end

    @testset "zero scale is a no-op" begin
        wake = LongitudinalRLCWake(freq = 1.0e9, Rshunt = 1.0e6, Q0 = 10.0,
                                   scale = 0.0)
        coords = [0.0 0.0 0.0 0.0 1.0e-3 0.3
                  0.0 0.0 0.0 0.0 2.0e-3 0.4]
        ref = copy(coords)
        linepass!(coords, Lattice([wake]), beam, zeros(Int, 2))
        @test coords == ref
    end

    @testset "single-particle tracking rejects collective element" begin
        wake = LongitudinalRLCWake(freq = 1.0e9, Rshunt = 1.0e6, Q0 = 10.0, scale = 1.0)
        lat = Lattice(AbstractElement[Drift(0.1), wake])
        @test_throws ArgumentError linepass(lat, SVector(0.0, 0.0, 0.0, 0.0, 1.0e-3, 0.0), beam)
    end
end

@testset "Aperture loss parity" begin
    # Two macroparticles sit outside the apertures; both codes must flag
    # exactly those and keep identical evolved coordinates.
    wide = vcat(particles_initial,
                [2.0e-3 0.0 0.0 0.0 1.0e-3 0.0
                 0.0 0.0 2.0e-3 0.0 1.0e-3 0.0])
    rap = [-5.0e-4, 5.0e-4, -5.0e-4, 5.0e-4, 0.0, 0.0]
    eap = [1.0e-3, 8.0e-4, 0.0, 0.0, 0.0, 0.0]

    function run_aperture_pair(elem_tp, elem_jt)
        tpb = Beam(ENERGY_VAL)
        jtb = JuTrack.Beam(flip_longitudinal_coordinate(wide),
                           energy = ENERGY_VAL, mass = JuTrack.m_e)
        c = copy(wide)
        lf = zeros(Int, size(c, 1))
        linepass!(c, Lattice([elem_tp]), tpb, lf)
        JuTrack.linepass!([elem_jt], jtb)
        return c, lf, flip_longitudinal_coordinate(jtb.r), jtb.lost_flag
    end

    cases = [
        ("Drift rectangular",
         Drift(0.7; r_apertures = rap),
         JuTrack.DRIFT(len = 0.7, RApertures = collect(rap))),
        ("Drift elliptical",
         Drift(0.7; e_apertures = eap),
         JuTrack.DRIFT(len = 0.7, EApertures = collect(eap))),
        ("SBend rectangular",
         SBend(0.9, 0.15; num_int_steps = 10, r_apertures = rap),
         JuTrack.SBEND(len = 0.9, angle = 0.15, NumIntSteps = 10,
                       RApertures = collect(rap))),
    ]
    @testset "$name" for (name, etp, ejt) in cases
        c, lf, jr, jlf = run_aperture_pair(etp, ejt)
        @test lf == jlf
        @test sum(lf) == 2  # exactly the two out-of-aperture macroparticles
        @test c ≈ jr atol = PARITY_ATOL
    end

    # Without apertures nobody is lost.
    c, lf, jr, jlf = run_aperture_pair(Drift(0.7), JuTrack.DRIFT(len = 0.7))
    @test lf == zeros(Int, length(lf))
    @test jlf == zeros(Int, length(jlf))
end

@testset "Element Gaps" begin
    @test_skip false # JuTrack has no Float64 pass! for LongitudinalWake.
    # StrongThinGaussianBeam / StrongGaussianBeam are verified from first
    # principles in test/verify_beambeam.jl; JuTrack's beam-beam is a placeholder.
end
