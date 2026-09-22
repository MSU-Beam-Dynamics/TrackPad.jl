# Regenerate `test/jutrack_reference.jl`, the frozen JuTrack outputs that the
# parity tests compare TrackPad against.
#
#     julia --project=test/jutrack_reference test/jutrack_reference/generate.jl
#
# TrackPad itself does not depend on JuTrack; only this script does. Run it when
# a parity case is added, or to deliberately re-baseline against a newer
# JuTrack, and read the resulting diff: every number that moves is a change in
# what the two codes agree on.
#
# The JuTrack side of every case lives here. The TrackPad side stays in the test
# files, which look the reference up by the same key; `test/verify_elements.jl`
# asserts that the two sets of keys match exactly, so a case added on one side
# and not the other fails loudly.

using TrackPad
import JuTrack
using LinearAlgebra
using StaticArrays

Base.include(@__MODULE__, joinpath(@__DIR__, "..", "convention_helpers.jl"))

const REF = Pair{String,Any}[]
record!(key, value) = (push!(REF, String(key) => value); value)

# Every parity case is taken in the shared ultrarelativistic limit with
# JuTrack's exact-beti switch on: its finite-beta longitudinal normalisation
# differs between element families, which the conventions page documents.
JuTrack.use_exact_beti = 1
JuTrack.use_exact_Hamiltonian = 1

const P0 = PARITY_PARTICLES
const ENERGY_VAL = 1.0e15          # ultrarelativistic parity energy (kinetic, JuTrack)
const OPTICS_ENERGY = 3.0e9        # kinetic energy of the optics/TPSA comparisons

function track_jt(elems, particles; energy = ENERGY_VAL, current = 0.0)
    b = JuTrack.Beam(flip_longitudinal_coordinate(particles),
                     energy = energy, mass = JuTrack.m_e, current = current)
    JuTrack.linepass!(elems isa AbstractVector ? elems : [elems], b)
    return flip_longitudinal_coordinate(b.r)
end

# ── element-by-element parity (test/verify_elements.jl) ─────────────────────
element_cases = Pair{String,Any}[
    "Marker"                                 => JuTrack.MARKER(),
    "Drift"                                  => JuTrack.DRIFT(len = 0.7),
    "Quadrupole"                             => JuTrack.KQUAD(len = 0.4, k1 = 1.3, NumIntSteps = 10),
    "Sextupole"                              => JuTrack.KSEXT(len = 0.4, k2 = 2.0, NumIntSteps = 10),
    "Octupole"                               => JuTrack.KOCT(len = 0.4, k3 = -3.0, NumIntSteps = 10),
    "ThinMultipole"                          => JuTrack.thinMULTIPOLE(len = 0.0, PolynomA = [0.0, 0.0, 0.0, 0.0], PolynomB = [0.0, 0.2, -0.1, 0.05], MaxOrder = 3),
    "SBend"                                  => JuTrack.SBEND(len = 0.9, angle = 0.15, e1 = 0.03, e2 = 0.02, NumIntSteps = 10),
    "RBend"                                  => JuTrack.RBEND(len = 0.9, angle = 0.15, NumIntSteps = 10),
    "ExactSBend"                             => JuTrack.ESBEND(len = 0.9, angle = 0.15, e1 = 0.03, e2 = 0.02, NumIntSteps = 10),
    "ERBend"                                 => JuTrack.ERBEND(len = 0.9, angle = 0.15, NumIntSteps = 10),
    "RFCavity"                               => JuTrack.RFCA(len = 0.3, volt = 2.0e6, freq = 500.0e6, lag = 0.002, h = 1200.0, philag = 0.3, energy = ENERGY_VAL),
    "CrabCavity"                             => JuTrack.CRABCAVITY(len = 0.3, volt = 2.0e6, freq = 500.0e6, phi = 0.1, energy = ENERGY_VAL),
    "AccelCavity"                            => JuTrack.AccelCavity(len = 0.0, volt = 2.5e6, freq = 500.0e6, h = 1200.0, phis = 0.2),
    "Solenoid"                               => JuTrack.SOLENOID(len = 0.5, ks = 0.8),
    "Corrector"                              => JuTrack.CORRECTOR(len = 0.4, xkick = 1.5e-4, ykick = -2.2e-4),
    "HKicker"                                => JuTrack.HKICKER(len = 0.0, xkick = 2.5e-4),
    "VKicker"                                => JuTrack.VKICKER(len = 0.0, ykick = -1.5e-4),
    "YRotation"                              => JuTrack.YROTATION(len = 0.0, angle = 0.02),
    "Wiggler"                                => JuTrack.WIGGLER(len = 1.2, lw = 0.2, Bmax = 0.8, Nsteps = 8),
    "Wiggler vertical harmonics"             => JuTrack.WIGGLER(len = 1.2, lw = 0.2, Bmax = 0.8, Nsteps = 8, By = Int[], Bx = [1, 1, 1, 0, 1, 0]),
    "Wiggler mixed harmonics"                => JuTrack.WIGGLER(len = 1.2, lw = 0.2, Bmax = 0.8, Nsteps = 8, By = [1, 1, 0, 1, 1, 0], Bx = [1, 2, 1, 0, 1, 0]),
    "SBend gradient auto-order"              => JuTrack.SBEND(len = 0.9, angle = 0.15, e1 = 0.03, e2 = 0.02, NumIntSteps = 10, PolynomB = [0.0, 0.3, 0.0, 0.0]),
    "RBend gradient auto-order"              => JuTrack.RBEND(len = 0.9, angle = 0.15, NumIntSteps = 10, PolynomB = [0.0, 0.25, 0.0, 0.0]),
    "Quadrupole fringes"                     => JuTrack.KQUAD(len = 0.4, k1 = 1.3, NumIntSteps = 10, FringeQuadEntrance = 1, FringeQuadExit = 1),
    "Sextupole fringes"                      => JuTrack.KSEXT(len = 0.4, k2 = 2.0, NumIntSteps = 10, FringeQuadEntrance = 1),
    "Octupole fringes"                       => JuTrack.KOCT(len = 0.4, k3 = -3.0, NumIntSteps = 10, FringeQuadExit = 1),
    "ThinMultipole fringes"                  => JuTrack.thinMULTIPOLE(len = 0.0, PolynomA = [0.0, 0.0, 0.0, 0.0], PolynomB = [0.0, 0.2, -0.1, 0.05], MaxOrder = 3, FringeQuadEntrance = 1),
    "SBend quad-fringe gates"                => JuTrack.SBEND(len = 0.9, angle = 0.15, e1 = 0.03, e2 = 0.02, NumIntSteps = 10, FringeQuadEntrance = 1, FringeQuadExit = 1),
    "ExactSBend quad-fringe ordering"        => JuTrack.ESBEND(len = 0.9, angle = 0.15, e1 = 0.03, e2 = 0.02, NumIntSteps = 10, FringeQuadEntrance = 1, FringeQuadExit = 1),
    "Sextupole kick angle with fringes"      => JuTrack.KSEXT(len = 0.4, k2 = 2.0, NumIntSteps = 10, KickAngle = [1.2e-4, -0.8e-4], FringeQuadEntrance = 1, FringeQuadExit = 1),
    "Octupole kick angle with fringes"       => JuTrack.KOCT(len = 0.4, k3 = -3.0, NumIntSteps = 10, KickAngle = [-0.7e-4, 1.1e-4], FringeQuadEntrance = 1, FringeQuadExit = 1),
    "DriftSC"                                => JuTrack.DRIFT_SC(len = 0.5, a = 0.01, b = 0.02, Nl = 12, Nm = 14, Nsteps = 2),
    "QuadrupoleSC"                           => JuTrack.KQUAD_SC(len = 0.4, k1 = 0.0, a = 0.01, b = 0.02),
    "SextupoleSC"                            => JuTrack.KSEXT_SC(len = 0.3, k2 = 2.4, a = 0.01, b = 0.02),
    "OctupoleSC"                             => JuTrack.KOCT_SC(len = 0.2, k3 = 3.6, a = 0.01, b = 0.02),
    "SBendSC"                                => JuTrack.SBEND_SC(len = 1.0, angle = 0.0, e1 = 0.0, e2 = 0.0, a = 0.01, b = 0.02),
    "RBendSC"                                => JuTrack.RBEND_SC(len = 1.2, angle = 0.0),
    "LBend"                                  => JuTrack.LBEND(len = 0.7, angle = 0.0, K = 0.0),
    "SpaceCharge"                            => JuTrack.SPACECHARGE(len = 0.9, effective_len = 0.4, Nl = 11, Nm = 9, a = 0.015, b = 0.017),
    "LongitudinalRLCWake"                    => JuTrack.LongitudinalRLCWake(freq = 1.0e9, Rshunt = 0.0, Q0 = 1.2),
]
for (name, jt) in element_cases
    record!("element/" * name, track_jt(jt, P0))
end

# RLC wake Green function, at the delays the test compares
let jt_rlc = JuTrack.LongitudinalRLCWake(freq = 1.0e9, Rshunt = 1.0e6, Q0 = 1.2)
    record!("wake/rlc_delays", [-2.0e-9, -1.0e-10, 0.0, 1.0e-10])
    record!("wake/rlc", [JuTrack.wakefieldfunc_RLCWake(jt_rlc, t)
                         for t in (-2.0e-9, -1.0e-10, 0.0, 1.0e-10)])
end

# ── aperture losses (test/verify_elements.jl) ───────────────────────────────
let wide = vcat(P0, [2.0e-3 0.0 0.0 0.0 1.0e-3 0.0
                     0.0 0.0 2.0e-3 0.0 1.0e-3 0.0]),
    rap = [-5.0e-4, 5.0e-4, -5.0e-4, 5.0e-4, 0.0, 0.0],
    eap = [1.0e-3, 8.0e-4, 0.0, 0.0, 0.0, 0.0]

    record!("aperture/particles", wide)
    for (name, ejt) in [
        "Drift rectangular" => JuTrack.DRIFT(len = 0.7, RApertures = collect(rap)),
        "Drift elliptical"  => JuTrack.DRIFT(len = 0.7, EApertures = collect(eap)),
        "SBend rectangular" => JuTrack.SBEND(len = 0.9, angle = 0.15, NumIntSteps = 10,
                                             RApertures = collect(rap)),
        "Drift open"        => JuTrack.DRIFT(len = 0.7),
    ]
        jtb = JuTrack.Beam(flip_longitudinal_coordinate(wide),
                           energy = ENERGY_VAL, mass = JuTrack.m_e)
        JuTrack.linepass!([ejt], jtb)
        record!("aperture/" * name * "/r", flip_longitudinal_coordinate(jtb.r))
        record!("aperture/" * name * "/lost", collect(jtb.lost_flag))
    end
end

# ── whole-line tracking (verify_bend, verify_exact_bend, verify_fodo, verify_dba)
record!("line/single_bend", track_jt(
    [JuTrack.SBEND(len = 1.0, angle = pi/4, e1 = pi/8, e2 = pi/8)], P0))
record!("line/exact_bend", track_jt(
    [JuTrack.ESBEND(len = 1.0, angle = pi/9, e1 = pi/18, e2 = pi/18)], P0))
record!("line/fodo", track_jt([
    JuTrack.DRIFT(name = "D1", len = 1.0),
    JuTrack.SBEND(name = "Q1", len = 1.0, angle = 0.0, PolynomB = [0.0, -0.9, 0.0, 0.0]),
    JuTrack.DRIFT(name = "D2", len = 1.0),
    JuTrack.SBEND(name = "B1", len = 0.6, angle = pi / 15.0),
    JuTrack.DRIFT(name = "D3", len = 1.0),
    JuTrack.SBEND(name = "Q2", len = 1.0, angle = 0.0, PolynomB = [0.0, 0.3, 0.0, 0.0]),
    JuTrack.DRIFT(name = "D4", len = 1.0),
], P0))
record!("line/dba", track_jt([
    JuTrack.DRIFT(len = 2.0),
    JuTrack.SBEND(len = 1.0, angle = pi/9, e1 = pi/18, e2 = pi/18),
    JuTrack.DRIFT(len = 0.5),
    JuTrack.KQUAD(len = 0.4, k1 = 1.5, NumIntSteps = 10),
    JuTrack.DRIFT(len = 0.5),
    JuTrack.KQUAD(len = 0.4, k1 = -1.2, NumIntSteps = 10),
    JuTrack.DRIFT(len = 0.5),
    JuTrack.KQUAD(len = 0.4, k1 = 1.5, NumIntSteps = 10),
    JuTrack.DRIFT(len = 0.5),
    JuTrack.SBEND(len = 1.0, angle = pi/9, e1 = pi/18, e2 = pi/18),
    JuTrack.DRIFT(len = 2.0),
], P0))

# ── closed orbit (test/verify_closed_orbit.jl) ──────────────────────────────
let ring_jt = [
        JuTrack.DRIFT(len = 1.0),
        JuTrack.SBEND(len = 0.5, angle = 0.0, PolynomB = [0.0, 0.6, 0.0, 0.0]),
        JuTrack.DRIFT(len = 1.0),
        JuTrack.SBEND(len = 0.5, angle = 0.0, PolynomB = [0.0, -0.6, 0.0, 0.0]),
    ]
    record!("closed_orbit/6d", collect(JuTrack.find_closed_orbit_6d(
        ring_jt; energy = OPTICS_ENERGY, mass = JuTrack.m_e)))
    record!("closed_orbit/4d", collect(JuTrack.find_closed_orbit_4d(
        ring_jt; dp = 0.0, energy = OPTICS_ENERGY, mass = JuTrack.m_e)))
end

# ── optics interfaces (test/verify_optics.jl) ───────────────────────────────
let ring_jt = [
        JuTrack.DRIFT(len = 1.0),
        JuTrack.KQUAD(len = 0.5, k1 = 0.6, NumIntSteps = 8),
        JuTrack.DRIFT(len = 1.0),
        JuTrack.KQUAD(len = 0.5, k1 = -0.6, NumIntSteps = 8),
    ]
    record!("optics/tune", collect(JuTrack.gettune(ring_jt; energy = OPTICS_ENERGY, mass = JuTrack.m_e)))
    record!("optics/chrom", collect(JuTrack.getchrom(ring_jt; energy = OPTICS_ENERGY, mass = JuTrack.m_e)))
    record!("optics/m66", canonicalize_jutrack_map(
        JuTrack.fastfindm66(ring_jt, 0.0; E0 = OPTICS_ENERGY, m0 = JuTrack.m_e)))
    op = JuTrack.periodicEdwardsTengTwiss(ring_jt, 0.0, 0; E0 = OPTICS_ENERGY, m0 = JuTrack.m_e)
    record!("optics/edwards_teng", [op.betax, op.alphax, op.betay, op.alphay])
    tw = JuTrack.twissring(ring_jt, 0.0, 0, [2, 4]; E0 = OPTICS_ENERGY, m0 = JuTrack.m_e)
    record!("optics/twissring_refpts",
            [[t.betax, t.alphax, t.betay, t.alphay] for t in tw])
end

# ── TPSA maps (test/verify_tpsa.jl) ─────────────────────────────────────────
# The test's cell, and its canonicalisation: JuTrack's fifth coordinate has the
# opposite sign, so the identity seed and the result are both flipped in z.
let line_jt = [
        JuTrack.KQUAD(len = 0.5, k1 = 1.5, NumIntSteps = 10),
        JuTrack.DRIFT(len = 2.0),
        JuTrack.KQUAD(len = 0.5, k1 = -1.5, NumIntSteps = 10),
        JuTrack.DRIFT(len = 2.0),
    ]
    for order in (1, 2)
        rin = [JuTrack.CTPS(0.0, i, 6, order) for i in 1:6]
        rin[5] = -rin[5]
        JuTrack.linepass_TPSA!(line_jt, rin; E0 = OPTICS_ENERGY, m0 = JuTrack.m_e)
        rin[5] = -rin[5]
        nterms = length(rin[1].map)      # 7 at order 1, 28 at order 2
        record!("tpsa/order$order", [rin[i].map[k] for i in 1:6, k in 1:nterms])
    end
end

# ── Enzyme finite-difference references (test/verify_enzyme_compat.jl) ──────
let r0 = [1e-3, 2e-4, 5e-4, -1e-4, 0.0, 0.0], h = 1e-6
    loss_jt(k) = begin
        line = [JuTrack.DRIFT(len = 1.0),
                JuTrack.SBEND(len = 0.4, angle = 0.0, PolynomB = [0.0, k, 0.0, 0.0]),
                JuTrack.DRIFT(len = 1.0)]
        b = JuTrack.Beam(reshape(copy(r0), 1, 6), energy = OPTICS_ENERGY, mass = JuTrack.m_e)
        JuTrack.linepass!(line, b)
        b.r[1, 1]
    end
    record!("enzyme/fd_static", (loss_jt(0.6 + h) - loss_jt(0.6 - h)) / (2h))
    record!("enzyme/fd_time",
            (loss_jt(0.6 + 0.15 * sin(0.3 + h)) - loss_jt(0.6 + 0.15 * sin(0.3 - h))) / (2h))
end

# ── time-dependent parameters (test/verify_time_dependence.jl) ──────────────
let r0 = [1e-3, 2e-4, 5e-4, -1e-4, 0.0, 0.0]
    jt_line(r, k1; num_steps) = begin
        line = [JuTrack.DRIFT(len = 1.0),
                JuTrack.SBEND(len = 0.4, angle = 0.0, PolynomB = [0.0, k1, 0.0, 0.0],
                              NumIntSteps = num_steps),
                JuTrack.DRIFT(len = 1.0)]
        b = JuTrack.Beam(reshape(flip_longitudinal_coordinate(collect(r)), 1, 6),
                         energy = OPTICS_ENERGY, mass = JuTrack.m_e)
        JuTrack.linepass!(line, b)
        flip_longitudinal_coordinate(vec(b.r))
    end
    record!("time/static", jt_line(r0, 0.6; num_steps = 10))
    record!("time/t1", jt_line(r0, 0.6 + 0.2 * sin(1.0) + 0.01 * 5; num_steps = 12))
    r = collect(r0)
    for k in 0:2
        r = jt_line(r, 0.6 + 0.2 * sin(0.1 * k) + 0.01 * (4 + k); num_steps = 8)
    end
    record!("time/three_turns", r)
end

# ── emit ────────────────────────────────────────────────────────────────────
emit(io, x::Float64) = print(io, repr(x))
emit(io, x::Int) = print(io, x)
function emit(io, v::AbstractVector)
    print(io, "[")
    for (i, x) in enumerate(v)
        i > 1 && print(io, ", ")
        emit(io, x)
    end
    print(io, "]")
end
function emit(io, m::AbstractMatrix)
    println(io, "[")
    for i in axes(m, 1)
        print(io, "        ")
        for j in axes(m, 2)
            emit(io, m[i, j])
            j < size(m, 2) && print(io, "  ")
        end
        println(io, i < size(m, 1) ? ";" : "")
    end
    print(io, "    ]")
end

open(joinpath(@__DIR__, "..", "jutrack_reference.jl"), "w") do io
    println(io, """
# Frozen JuTrack outputs for the parity tests. GENERATED -- do not edit.
#
#     julia --project=test/jutrack_reference test/jutrack_reference/generate.jl
#
# JuTrack $(pkgversion(JuTrack)), Julia $(VERSION), $(JuTrack.use_exact_beti == 1 ? "use_exact_beti = 1" : "use_exact_beti = 0").
# Each entry is what JuTrack produced for the case of the same name in the test
# suite; see jutrack_reference/README.md.

const JUTRACK_REFERENCE = Dict{String,Any}(""")
    for (k, v) in REF
        print(io, "    ", repr(k), " => ")
        emit(io, v)
        println(io, ",")
    end
    println(io, ")")
end
println("recorded ", length(REF), " reference entries")
