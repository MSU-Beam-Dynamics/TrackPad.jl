using Test
using TrackPad
using LinearAlgebra
import JuTrack

const PARITY_ATOL = 1e-15
const ENERGY_VAL = 3.5e9

# Deterministic initial coordinates shared across element parity checks.
const particles_initial = [
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
    beam = JuTrack.Beam(copy(particles), energy = energy, mass = JuTrack.m_e, current = current)
    JuTrack.linepass!([elem], beam)
    return beam.r
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
        _assert_parity(
            "LongitudinalRFMap",
            LongitudinalRFMap(2.0e-4, AccelCavity(0.0; volt = 2.5e6, freq = 500.0e6, h = 1200.0, phis = 0.2, energy = ENERGY_VAL)),
            JuTrack.LongitudinalRFMap(2.0e-4, JuTrack.AccelCavity(len = 0.0, volt = 2.5e6, freq = 500.0e6, h = 1200.0, phis = 0.2)),
        )

        # Auxiliary helpers.
        _assert_parity("Solenoid", Solenoid(0.5, 0.8), JuTrack.SOLENOID(len = 0.5, ks = 0.8))
        _assert_parity("Corrector", Corrector(0.4, 1.5e-4, -2.2e-4), JuTrack.CORRECTOR(len = 0.4, xkick = 1.5e-4, ykick = -2.2e-4))
        _assert_parity("HKicker", HKicker(L = 0.0, xkick = 2.5e-4), JuTrack.HKICKER(len = 0.0, xkick = 2.5e-4))
        _assert_parity("VKicker", VKicker(L = 0.0, ykick = -1.5e-4), JuTrack.VKICKER(len = 0.0, ykick = -1.5e-4))
        _assert_parity("Translation", Translation(0.0; dx = 1e-3, dy = -2e-3, ds = 3e-3), JuTrack.TRANSLATION(len = 0.0, dx = 1e-3, dy = -2e-3, ds = 3e-3))
        _assert_parity("YRotation", YRotation(0.0; angle = 0.02), JuTrack.YROTATION(len = 0.0, angle = 0.02))
        _assert_parity("Wiggler", Wiggler(1.2; lw = 0.2, Bmax = 0.8, Nsteps = 8), JuTrack.WIGGLER(len = 1.2, lw = 0.2, Bmax = 0.8, Nsteps = 8))
        _assert_parity("LorentzBoost", LorentzBoost(0.03), JuTrack.LorentzBoost(0.03))
        _assert_parity("InvLorentzBoost", InvLorentzBoost(0.03), JuTrack.InvLorentzBoost(0.03))

        # Space-charge canonical family.
        _assert_parity("DriftSC", DriftSC(0.5; a = 0.01, b = 0.02, Nl = 12, Nm = 14, Nsteps = 2), JuTrack.DRIFT_SC(len = 0.5, a = 0.01, b = 0.02, Nl = 12, Nm = 14, Nsteps = 2))
        _assert_parity("QuadrupoleSC", QuadrupoleSC(0.4; k1 = 0.0, a = 0.01, b = 0.02), JuTrack.KQUAD_SC(len = 0.4, k1 = 0.0, a = 0.01, b = 0.02))
        _assert_parity("SextupoleSC", SextupoleSC(0.3; k2 = 2.4, a = 0.01, b = 0.02), JuTrack.KSEXT_SC(len = 0.3, k2 = 2.4, a = 0.01, b = 0.02))
        _assert_parity("OctupoleSC", OctupoleSC(0.2; k3 = 3.6, a = 0.01, b = 0.02), JuTrack.KOCT_SC(len = 0.2, k3 = 3.6, a = 0.01, b = 0.02))
        _assert_parity("SBendSC", SBendSC(1.0, 0.0, 0.0, 0.0; a = 0.01, b = 0.02), JuTrack.SBEND_SC(len = 1.0, angle = 0.0, e1 = 0.0, e2 = 0.0, a = 0.01, b = 0.02))
        _assert_parity("RBendSC", RBendSC(1.2, 0.0), JuTrack.RBEND_SC(len = 1.2, angle = 0.0))
        _assert_parity("LBend", LBend(0.7, 0.0; K = 0.0), JuTrack.LBEND(len = 0.7, angle = 0.0, K = 0.0))
        _assert_parity("SpaceCharge", SpaceCharge(0.9; effective_len = 0.4, Nl = 11, Nm = 9, a = 0.015, b = 0.017), JuTrack.SPACECHARGE(len = 0.9, effective_len = 0.4, Nl = 11, Nm = 9, a = 0.015, b = 0.017))

        # Collective/wake models with strict zero-kick settings.
        _assert_parity("LongitudinalRLCWake", LongitudinalRLCWake(freq = 1.0e9, Rshunt = 0.0, Q0 = 1.2, scale = 1.0), JuTrack.LongitudinalRLCWake(freq = 1.0e9, Rshunt = 0.0, Q0 = 1.2))
        tp_sgb = StrongGaussianBeam(0.0, TrackPad.M_ELECTRON, 1.0, 1_000_000, ENERGY_VAL, [1e-3, 1.2e-3]; nzslice = 3)
        jt_op = JuTrack.optics4DUC(1.0, 0.0, 1.0, 0.0)
        jt_sgb = JuTrack.StrongGaussianBeam(0.0, JuTrack.m_e, 1.0, 1_000_000, ENERGY_VAL, jt_op, [1e-3, 1.2e-3], 3)
        _assert_parity("StrongGaussianBeam", tp_sgb, jt_sgb)
    finally
        JuTrack.use_exact_beti = old_exact_beti
    end
end

@testset "Element Gaps" begin
    @test_skip false # JuTrack has no Float64 pass! for LongitudinalWake.
    @test_skip false # JuTrack has no Float64 pass! for StrongThinGaussianBeam.
end
