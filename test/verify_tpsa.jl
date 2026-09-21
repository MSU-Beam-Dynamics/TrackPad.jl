using Test
using TrackPad
Base.include(@__MODULE__, joinpath(@__DIR__, "convention_helpers.jl"))
using PolySeries
using StaticArrays
using LinearAlgebra
import JuTrack

# Keep JuTrack TPSA comparisons on the same Hamiltonian convention as TrackPad.
# TrackPad default is exact Hamiltonian + exact beti (deltaE/p0 style sixth coord).
function with_matched_jutrack_hamiltonian(f::Function)
    old_h = JuTrack.use_exact_Hamiltonian
    old_b = JuTrack.use_exact_beti
    try
        if TrackPad.USE_EXACT_HAMILTONIAN
            JuTrack.use_exact_Hamiltonian = 1
            JuTrack.use_exact_beti = 1
        else
            JuTrack.use_exact_Hamiltonian = 0
            JuTrack.use_exact_beti = 0
        end
        return f()
    finally
        JuTrack.use_exact_Hamiltonian = old_h
        JuTrack.use_exact_beti = old_b
    end
end

# ── Common lattice (same parameters used in both TrackPad and JuTrack) ──────
const L_q = 0.5
const L_d = 2.0
const k1  = 1.5

# TrackPad lattice
QF = Quadrupole(L_q,  k1; num_int_steps=10)
QD = Quadrupole(L_q, -k1; num_int_steps=10)
D  = Drift(L_d)
ring = Lattice([QF, D, QD, D]; periodic=true)
beam = jutrack_beam(3.0e9)   # JuTrack TPSA calls use E0=3.0e9 (kinetic)

# JuTrack lattice (KQUAD matches TrackPad Quadrupole; same NumIntSteps)
const line_jt = [
    JuTrack.KQUAD(len=L_q, k1= k1, NumIntSteps=10),
    JuTrack.DRIFT(len=L_d),
    JuTrack.KQUAD(len=L_q, k1=-k1, NumIntSteps=10),
    JuTrack.DRIFT(len=L_d),
]

# ── Helpers ──────────────────────────────────────────────────────────────────

# PolySeries leaves inactive degree blocks uninitialized. Read coefficients
# through the degree mask so inactive terms have their mathematical value zero.
@inline function tp_coefficient(series, index::Int)
    degree = Int(series.desc.polymap.map[index, 1])
    active = (series.degree_mask[] & (UInt64(1) << degree)) != 0
    return active ? series.c[index] : zero(eltype(series.c))
end

# Extract 6×6 linear map from TrackPad PolySeries output
function tp_to_matrix(r_tpsa::SVector{6})
    M = zeros(Float64, 6, 6)
    for i in 1:6, j in 1:6
        M[i, j] = tp_coefficient(r_tpsa[i], j + 1)
    end
    return M
end

# Extract 6×6 linear map from JuTrack CTPS vector
function jt_to_matrix(rin)
    M = zeros(Float64, 6, 6)
    for i in 1:6, j in 1:6
        M[i, j] = rin[i].map[j+1]    # map[1]=const, map[j+1]=coeff of var j
    end
    return M
end

function canonical_jutrack_tpsa(order::Int)
    rin = [JuTrack.CTPS(0.0, i, 6, order) for i in 1:6]
    rin[5] = -rin[5]
    JuTrack.linepass_TPSA!(line_jt, rin; E0=3.0e9, m0=JuTrack.m_e)
    rin[5] = -rin[5]
    return rin
end

@testset "TPSA linear map matches finite-difference one_turn_map" begin
    # First-order TPSA map about closed orbit (zero for this cell)
    r_tpsa = tpsa_map(ring, beam; order=1)

    # Finite-difference reference
    M_ref = one_turn_map(ring, beam)

    M_tpsa = tp_to_matrix(r_tpsa)

    # Tolerance set to ~1e-7: FD roundoff with default h=1e-8 is O(eps/h) ≈ 2.2e-8
    for i in 1:6
        for j in 1:6
            @test isapprox(M_tpsa[i,j], M_ref[i,j]; atol=1e-7)
        end
    end
end

@testset "TPSA constant term matches linepass at expansion point" begin
    r_tpsa = tpsa_map(ring, beam; order=1)
    r_ref  = linepass(ring, zero(SVector{6,Float64}), beam)
    for i in 1:6
        @test isapprox(tp_coefficient(r_tpsa[i], 1), r_ref[i]; atol=1e-14)
    end
end

@testset "TPSA second-order: quadratic coefficients are nonzero" begin
    r2 = tpsa_map(ring, beam; order=2)
    # For a 6-variable order-2 CTPS, index layout:
    #   c[1]       = const
    #   c[2..7]    = linear terms (vars 1..6)
    #   c[8..28]   = quadratic terms (21 monomials)
    # At least some quadratic coefficients should be non-zero for a thick quad ring.
    any_nonzero = any(
        i -> any(k -> abs(tp_coefficient(r2[i], k)) > 1e-20, 8:length(r2[i].c)),
        1:6,
    )
    @test any_nonzero
end

@testset "TPSA with non-zero closed orbit" begin
    co = [1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]
    r_tpsa_co = tpsa_map(ring, beam; order=1, closed_orbit=co)
    r_ref_co  = linepass(ring, SVector{6,Float64}(co...), beam)
    for i in 1:6
        @test isapprox(tp_coefficient(r_tpsa_co[i], 1), r_ref_co[i]; atol=1e-14)
    end
end

@testset "Float64 tracking unaffected (regression)" begin
    # Confirm ordinary Float64 tracking still works correctly after TPSA type changes
    r0 = SVector(1e-4, 2e-4, -1e-4, 3e-4, 0.0, 1e-3)
    r1 = linepass(ring, r0, beam)
    @test all(isfinite, r1)
    @test !any(isnan, r1)
end

# ── JuTrack CTPS comparison ──────────────────────────────────────────────────
# JuTrack uses its own CTPS type (HighOrderTPS) with the same PolyMap index
# ordering as PolySeries. JuTrack stores dense coefficients in `.map`; PolySeries
# uses a degree mask and may leave inactive blocks of `.c` uninitialized.
# Both packages were built from the same original C++ code base.

@testset "JuTrack CTPS: first-order map matches TrackPad PolySeries" begin
    with_matched_jutrack_hamiltonian() do
        # JuTrack setup: 6 identity CTPS variables, order 1
        rin_jt = canonical_jutrack_tpsa(1)

        r_tp = tpsa_map(ring, beam; order=1)

        M_jt = jt_to_matrix(rin_jt)
        M_tp = tp_to_matrix(r_tp)

        # With matched Hamiltonian settings, maps agree to floating-noise level.
        for i in 1:6, j in 1:6
            @test isapprox(M_tp[i,j], M_jt[i,j]; atol=1e-12)
        end
    end
end

@testset "JuTrack CTPS: constant term (closed-orbit value) matches" begin
    with_matched_jutrack_hamiltonian() do
        rin_jt = canonical_jutrack_tpsa(1)

        r_tp = tpsa_map(ring, beam; order=1)

        for i in 1:6
            @test isapprox(tp_coefficient(r_tp[i], 1), rin_jt[i].map[1]; atol=1e-14)
        end
    end
end

@testset "JuTrack CTPS: second-order coefficients match TrackPad PolySeries" begin
    with_matched_jutrack_hamiltonian() do
        # order-2: 28 terms for 6 variables  (1 + 6 + 21)
        rin_jt2 = canonical_jutrack_tpsa(2)

        r2_tp = tpsa_map(ring, beam; order=2)

        nterms = length(rin_jt2[1].map)   # = 28
        @test nterms == length(r2_tp[1].c)
        for i in 1:6, k in 1:nterms
            @test isapprox(tp_coefficient(r2_tp[i], k), rin_jt2[i].map[k]; atol=1e-12)
        end
    end
end

# ── Per-element TPSA vs finite-difference maps ───────────────────────────────
# The linear part of the Taylor map must agree with the finite-difference
# transfer map for every element type that has a TPSA path, including the
# chromatic terms (dispersion, ∂py/∂δ) that only show up with a nonzero
# fringe-field integral.

const tpsa_elem_beam = jutrack_beam(3.0e9)

function tpsa_vs_fd(elem; atol=1e-6)
    lat = Lattice(AbstractElement[elem])
    M_tpsa = tp_to_matrix(tpsa_map(lat, tpsa_elem_beam; order=1))
    M_fd = transfer_map(lat, tpsa_elem_beam)
    return maximum(abs, M_tpsa - M_fd) <= atol
end

@testset "TPSA linear maps match finite differences per element" begin
    angle_val = 0.15
    @test tpsa_vs_fd(SBend(0.9, angle_val; num_int_steps=10))
    # Nonzero fringe integral and gap: the edge focusing depends on δ (Brown)
    # and on δ twice (SOLEIL); the map must carry those derivatives.
    @test tpsa_vs_fd(SBend(0.9, angle_val, angle_val/2, angle_val/2;
                           fint1=0.5, fint2=0.5, gap=0.05, num_int_steps=10))
    @test tpsa_vs_fd(SBend(0.9, angle_val, angle_val/2, angle_val/2;
                           fint1=0.5, fint2=0.5, gap=0.05,
                           fringe_bend_entrance=2, fringe_bend_exit=2, num_int_steps=10))
    @test tpsa_vs_fd(SBend(0.9, angle_val, angle_val/2, angle_val/2;
                           fint1=0.5, fint2=0.5, gap=0.05,
                           fringe_bend_entrance=3, fringe_bend_exit=3, num_int_steps=10))
    # Elements served by the generic core kernels.
    @test tpsa_vs_fd(Marker())
    # Two-argument `pass!` must not default `beti` from the coordinate type:
    # CTPS has no type-level `one`/`zero` (a polynomial needs a descriptor,
    # which the type does not carry), only the instance methods.
    let rc = SVector{6,CTPS{Float64}}(ntuple(i -> CTPS(0.0, i, 6, 1), 6))
        @test pass!(Marker(), rc) === rc
        @test_throws MethodError one(CTPS{Float64})
        @test iszero(zero(rc[1])) && !iszero(one(rc[1]))
    end
    @test tpsa_vs_fd(Patch(x_pitch=0.01, y_pitch=-0.02, tilt=0.1, z_offset=2e-3, t_offset=1e-12))
    @test tpsa_vs_fd(Translation(0.0; dx=1.0e-3, dy=-2.0e-3, ds=3.0e-3))
    @test tpsa_vs_fd(YRotation(0.0; angle=0.03))
    @test tpsa_vs_fd(ExactSBend(0.9, angle_val, angle_val/2, angle_val/2;
                                fringe_bend_entrance=0, fringe_bend_exit=0, num_int_steps=10))
    # default hard-edge dipole fringe (needs atan on CTPS)
    @test tpsa_vs_fd(ExactSBend(0.9, angle_val, angle_val/2, angle_val/2; num_int_steps=10))
    @test tpsa_vs_fd(ExactSBend(0.9, angle_val; num_int_steps=10))
    @test tpsa_vs_fd(Solenoid(0.5, 0.8))
    @test tpsa_vs_fd(Corrector(0.4, 1.5e-4, -2.2e-4))
end

@testset "TPSA rejects maps it cannot represent with a clear error" begin
    @test_throws ArgumentError tpsa_map(Lattice(AbstractElement[LBend(0.9, 0.15)]), tpsa_elem_beam)
end

@testset "Ring optics through Taylor maps (method=:tpsa)" begin
    pbeam = Beam(kinetic=1.0e9, mass=M_PROTON, charge=1.0)
    cell = AbstractElement[
        Quadrupole(0.5, 0.9), Drift(0.6), SBend(1.2, 2π/40; num_int_steps=10), Drift(0.6),
        Sextupole(0.2, 3.0), Drift(0.4), Corrector(0.0, 2e-5, -1e-5),
        Quadrupole(0.5, -0.9), Drift(0.6), SBend(1.2, 2π/40; num_int_steps=10), Drift(0.6),
        Sextupole(0.2, -3.0), Drift(0.4),
    ]
    arc = Lattice(vcat([copy(cell) for _ in 1:20]...); periodic=true)
    fd = periodic_twiss(arc, pbeam; max_step=0.05, second_order=true, radiation_integrals=true, method=:fd)
    tp = periodic_twiss(arc, pbeam; max_step=0.05, second_order=true, radiation_integrals=true)   # default method
    @test tp.method === :tpsa
    # exact Jacobians vs finite differences: agreement at the FD noise level
    @test tp.tunex ≈ fd.tunex atol=1e-10
    @test tp.tuney ≈ fd.tuney atol=1e-10
    @test tp.betax ≈ fd.betax rtol=1e-8
    @test tp.dx ≈ fd.dx atol=1e-8       # FD Jacobian roundoff ε/h per piece, ~1000 pieces
    @test tp.alphac ≈ fd.alphac rtol=1e-7
    @test tp.radiation.I5 ≈ fd.radiation.I5 rtol=1e-8
    @test tp.chromx ≈ fd.chromx rtol=1e-5
    @test tp.chromy ≈ fd.chromy rtol=1e-5
    @test tp.chrom2x ≈ fd.chrom2x rtol=1e-3
    # the analytic (no finite-difference step) chromaticity is the limit of the
    # finite-difference one as its step shrinks
    # (truncation ∝ dpp² down to the ~3e-6 finite-difference floor)
    ξt = getchrom(arc, pbeam; method=:tpsa)
    errs = [abs(getchrom(arc, pbeam; centered=true, dpp=d, method=:fd)[1] - ξt[1]) for d in (1e-3, 1e-4, 1e-6)]
    @test errs[1] > 10 * errs[2]
    @test errs[end] < 5e-6
    # amplitude detuning from the order-3 map matches tracking
    beam = Beam(3.0e9)
    fodo = AbstractElement[Quadrupole(0.5, 0.95), Drift(1.0), Quadrupole(0.5, -0.80), Drift(1.0)]
    oring = Lattice(vcat([Octupole(0.01, 2000.0; num_int_steps=1)], [copy(fodo) for _ in 1:10]...); periodic=true)
    dfd = periodic_twiss(oring, beam; detuning=true, method=:fd).detuning
    dtp = periodic_twiss(oring, beam; detuning=true, method=:tpsa).detuning
    @test dtp ≈ dfd rtol=1e-3
end
