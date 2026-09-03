using Test
using TrackPad
import JuTrack
using LinearAlgebra
using StaticArrays
Base.include(@__MODULE__, joinpath(@__DIR__, "convention_helpers.jl"))

function with_jutrack_exact_beti(f::Function)
    old_exact_beti = JuTrack.use_exact_beti
    JuTrack.use_exact_beti = 1
    try
        return f()
    finally
        JuTrack.use_exact_beti = old_exact_beti
    end
end

with_jutrack_exact_beti() do

# Simple periodic cell to validate optics APIs
D = Drift(1.0; name="D")
QF = Quadrupole(0.5, 0.6; name="QF", num_int_steps=8)
QD = Quadrupole(0.5, -0.6; name="QD", num_int_steps=8)

ring = Lattice([D, QF, D, QD]; periodic=true)
beam = Beam(3.0e9)

@testset "Open line and periodic ring boundaries" begin
    line = Lattice([D, QF])
    @test !isperiodic(line)
    @test isperiodic(ring)
    @test size(transfer_map(line, beam)) == (6, 6)
    entrance = optics4DUC(2.0, 0.1, 3.0, -0.2)
    transported = transport_twiss(line, beam, entrance)
    @test transported isa TransportTwissResult
    @test length(transported.s) == length(line) + 1
    @test transported.betax[1] == entrance.optics_x.beta
    @test transported.betay[1] == entrance.optics_y.beta
    @test periodic_twiss(ring, beam).tunex == twissline(ring, beam).tunex
    dispersion = periodic_dispersion(ring, beam)
    @test dispersion isa DispersionLineResult
    @test length(dispersion.s) == length(ring) + 1
    @test all(iszero, dispersion.dx)
    @test all(iszero, dispersion.dpx)
    @test all(iszero, dispersion.dy)
    @test all(iszero, dispersion.dpy)
    @test_throws ArgumentError one_turn_map(line, beam)
    @test_throws ArgumentError gettune(line, beam)
    @test_throws ArgumentError getchrom(line, beam)
    @test_throws ArgumentError twissline(line, beam)
    @test_throws ArgumentError periodic_twiss(line, beam)
    @test_throws ArgumentError periodic_dispersion(line, beam)
    @test_throws ArgumentError find_closed_orbit_4d(line, beam)
    @test_throws ArgumentError ringpass(
        line, zero(SVector{6,Float64}), beam, 1,
    )
    @test isperiodic(materialize_lattice(ring))
end

@testset "Twiss sampling inside thick elements" begin
    sampled = refine_lattice(
        ring; sample_integrator_steps=true, max_step=0.25,
    )
    # Two four-piece drifts and two eight-step quadrupoles.
    @test length(sampled) == 24
    @test isperiodic(sampled)
    @test sampled.name == ring.name
    @test all(isapprox.(get_length.(sampled.elements[1:4]), 0.25))
    @test all(e -> e isa Quadrupole && e.num_int_steps == 1,
              sampled.elements[5:12])

    tw_coarse = periodic_twiss(ring, beam)
    tw_sampled = periodic_twiss(
        ring, beam; sample_integrator_steps=true, max_step=0.25,
    )
    @test length(tw_coarse.s) == length(ring) + 1
    @test length(tw_sampled.s) == length(sampled) + 1
    @test tw_sampled.s[end] == tw_coarse.s[end]
    @test isapprox(tw_sampled.tunex, tw_coarse.tunex; atol=2e-15)
    @test isapprox(tw_sampled.tuney, tw_coarse.tuney; atol=2e-15)
    @test isapprox(tw_sampled.betax[end], tw_coarse.betax[end]; atol=2e-13)
    @test isapprox(tw_sampled.betay[end], tw_coarse.betay[end]; atol=2e-13)

    dispersion_sampled = periodic_dispersion(
        ring, beam; sample_integrator_steps=true, max_step=0.25,
    )
    @test dispersion_sampled.s == tw_sampled.s

    line = Lattice(AbstractElement[Drift(1.0), QF])
    entrance = optics4DUC(2.0, 0.1, 3.0, -0.2)
    transported = transport_twiss(
        line, beam, entrance;
        sample_integrator_steps=true, max_step=0.2,
    )
    @test length(transported.s) == 5 + QF.num_int_steps + 1
    @test transported.s[end] == 1.0 + QF.L
    @test_throws ArgumentError refine_lattice(line; max_step=0.0)
    @test_throws ArgumentError refine_lattice(line; max_step=Inf)

    sc_line = Lattice(AbstractElement[
        DriftSC(0.6), QuadrupoleSC(0.3; k1=0.5, num_int_steps=3),
    ])
    sc_sampled = refine_lattice(
        sc_line; sample_integrator_steps=true, max_step=0.2,
    )
    @test length(sc_sampled) == 6
    @test all(e -> e isa QuadrupoleSC && e.num_int_steps == 1,
              sc_sampled.elements[4:6])
end

@testset "Optics slicing preserves bend endpoints" begin
    t1 = @SVector [1e-5, 0.0, -2e-5, 0.0, 0.0, 0.0]
    t2 = -t1
    bend = SBend(
        1.0, 0.1, 0.02, 0.03;
        num_int_steps=5, fint1=0.4, fint2=0.5, gap=0.02,
        fringe_bend_entrance=1, fringe_bend_exit=1,
        fringe_quad_entrance=1, fringe_quad_exit=1,
        polynom_b=[0.0, 0.2, 0.0, 0.0],
        kick_angle=[2e-4, -3e-4], t1=t1, t2=t2,
    )
    bend_line = Lattice(AbstractElement[bend])
    pieces = refine_lattice(bend_line; sample_integrator_steps=true)
    @test length(pieces) == bend.num_int_steps
    @test pieces[1].e1 == bend.e1
    @test pieces[1].fringe_bend_entrance == 1
    @test pieces[1].fringe_quad_entrance == 1
    @test pieces[1].t1 == bend.t1
    @test all(p -> iszero(p.e1) && p.fringe_bend_entrance == 0 &&
                    p.fringe_quad_entrance == 0 && iszero(p.t1),
              pieces.elements[2:end])
    @test pieces[end].e2 == bend.e2
    @test pieces[end].fringe_bend_exit == 1
    @test pieces[end].fringe_quad_exit == 1
    @test pieces[end].t2 == bend.t2
    @test all(p -> iszero(p.e2) && p.fringe_bend_exit == 0 &&
                    p.fringe_quad_exit == 0 && iszero(p.t2),
              pieces.elements[1:end-1])

    r0 = @SVector [1e-3, 2e-4, -3e-4, 1e-4, 0.0, 1e-3]
    @test isapprox(
        linepass(bend_line, r0, beam), linepass(pieces, r0, beam);
        atol=5e-15, rtol=0,
    )

    linear_bend = LBend(
        1.2, 0.12, 0.03, 0.04;
        K=0.2, by_error=1e-4, fint1=0.3, fint2=0.4, full_gap=0.02,
    )
    linear_line = Lattice(AbstractElement[linear_bend])
    linear_pieces = refine_lattice(linear_line; max_step=0.25)
    @test length(linear_pieces) == 5
    @test linear_pieces[1].e1 == linear_bend.e1
    @test linear_pieces[end].e2 == linear_bend.e2
    @test isapprox(
        linepass(linear_line, r0, beam), linepass(linear_pieces, r0, beam);
        atol=5e-15, rtol=0,
    )
end

@testset "Longitudinal coordinate convention" begin
    low_energy_beam = Beam(50.0e6; mass=M_PROTON, charge=1.0)
    beta0 = low_energy_beam.beta
    delta_e = 0.02
    delta_p = sqrt(1 + 2delta_e / beta0 + delta_e^2) - 1

    # The sixth coordinate is ΔE/(P0*c), which differs visibly from ΔP/P0
    # away from the ultrarelativistic limit.
    @test delta_p != delta_e
    @test 1 + delta_p == sqrt(1 + 2delta_e / beta0 + delta_e^2)

    L = 1.7
    r_energy = SVector(0.0, 0.0, 0.0, 0.0, 0.0, delta_e)
    tracked_energy = drift6(r_energy, L, inv(beta0))
    pi_s = 1 + delta_p
    expected_z = -L * ((inv(beta0) + delta_e) / pi_s - inv(beta0))
    @test tracked_energy[5] ≈ expected_z rtol=1.0e-14

    # A longer geometric path arrives late and decreases z=s/β0-c*t.
    r_angle = SVector(0.0, 0.1, 0.0, 0.0, 0.0, 0.0)
    @test drift6(r_angle, L, inv(beta0))[5] < 0

    canonical_line = Lattice(AbstractElement[
        Drift(0.7),
        RFCavity(0.0, 2.0e5, 80.0e6, 0.01;
                  energy=low_energy_beam.energy, charge=low_energy_beam.charge),
        Drift(0.4),
    ])
    map6 = transfer_map(canonical_line, low_energy_beam; h=1.0e-7)
    symplectic_form = zeros(6, 6)
    for i in (1, 3, 5)
        symplectic_form[i, i + 1] = 1
        symplectic_form[i + 1, i] = -1
    end
    @test norm(map6' * symplectic_form * map6 - symplectic_form, Inf) < 1.0e-8
end

ring_jt = [
    JuTrack.DRIFT(len=1.0),
    JuTrack.KQUAD(len=0.5, k1=0.6, NumIntSteps=8),
    JuTrack.DRIFT(len=1.0),
    JuTrack.KQUAD(len=0.5, k1=-0.6, NumIntSteps=8),
]
tune_jt = JuTrack.gettune(ring_jt; energy=3.0e9, mass=JuTrack.m_e)
chrom_jt = JuTrack.getchrom(ring_jt; energy=3.0e9, mass=JuTrack.m_e)

@testset "TrackPad Optics API" begin
    qx, qy = gettune(ring, beam)
    @test isfinite(qx)
    @test isfinite(qy)
    @test 0.0 <= qx <= 1.0
    @test 0.0 <= qy <= 1.0
    @test qx == tune_jt[1]
    @test qy == tune_jt[2]

    ξx, ξy = getchrom(ring, beam; dp=0.0)
    @test isfinite(ξx)
    @test isfinite(ξy)
    @test ξx == chrom_jt[1]
    @test ξy == chrom_jt[2]

    ξx_centered, ξy_centered = getchrom(
        ring, beam;
        dpp=1.0e-5, centered=true, closed_orbit=true,
    )
    @test isfinite(ξx_centered)
    @test isfinite(ξy_centered)
    @test_throws ArgumentError getchrom(ring, beam; dpp=0.0)

    tw = twissline(ring, beam)
    @test length(tw.s) == length(ring) + 1
    @test length(tw.betax) == length(ring) + 1
    @test length(tw.betay) == length(ring) + 1
    @test all(tw.betax .> 0)
    @test all(tw.betay .> 0)
    @test isapprox(tw.tunex, qx; atol=1e-12)
    @test isapprox(tw.tuney, qy; atol=1e-12)
end

@testset "TrackPad JuTrack-Compatible Optics/Map Interface" begin
    m66 = fastfindm66(ring, 0.0; E0=3.0e9, m0=M_ELECTRON)
    m66_ref = one_turn_map(ring, beam; h=1.5e-8)
    @test m66 == m66_ref

    m66_ord0 = findm66(ring, 0.0, 0; E0=3.0e9, m0=M_ELECTRON)
    @test m66_ord0 == m66

    m66_jt = canonicalize_jutrack_map(
        JuTrack.fastfindm66(ring_jt, 0.0; E0=3.0e9, m0=JuTrack.m_e),
    )
    @test m66 == m66_jt

    refpts = [2, 4]
    mref = fastfindm66_refpts(ring, 0.0, refpts; E0=3.0e9, m0=M_ELECTRON)
    @test size(mref) == (6, 6, 2)
    @test isapprox(mref[:, :, 2] * mref[:, :, 1], m66; atol=5e-7)

    op = periodicEdwardsTengTwiss(ring, 0.0, 0; E0=3.0e9, m0=M_ELECTRON)
    @test op isa optics4DUC
    @test op.optics_x.beta > 0
    @test op.optics_y.beta > 0

    op_jt = JuTrack.periodicEdwardsTengTwiss(ring_jt, 0.0, 0; E0=3.0e9, m0=JuTrack.m_e)
    @test isapprox(op.optics_x.beta, op_jt.betax; atol=1e-13)
    @test isapprox(op.optics_x.alpha, op_jt.alphax; atol=1e-13)
    @test isapprox(op.optics_y.beta, op_jt.betay; atol=1e-13)
    @test isapprox(op.optics_y.alpha, op_jt.alphay; atol=1e-13)

    twr = twissring(ring, 0.0, 0; E0=3.0e9, m0=M_ELECTRON)
    @test twr isa TwissLineResult
    @test length(twr.s) == length(ring) + 1
    @test isapprox(twr.tunex, tune_jt[1]; atol=5e-7)
    @test isapprox(twr.tuney, tune_jt[2]; atol=5e-7)

    twr_ref = twissring(ring, 0.0, 0, refpts; E0=3.0e9, m0=M_ELECTRON)
    @test length(twr_ref) == length(refpts)
    @test twr_ref[1] isa optics4DUC

    tin = optics4DUC(op.optics_x, op.optics_y)
    tout = twissline(tin, ring, 0.0, 0, 4; E0=3.0e9, m0=M_ELECTRON)
    @test tout isa optics4DUC
    @test isapprox(tout.optics_x.beta, op.optics_x.beta; atol=1e-13)
    @test isapprox(tout.optics_x.alpha, op.optics_x.alpha; atol=1e-13)
    @test isapprox(tout.optics_y.beta, op.optics_y.beta; atol=1e-13)
    @test isapprox(tout.optics_y.alpha, op.optics_y.alpha; atol=1e-13)

    twline_ref = twissline(tin, ring, 0.0, 0, refpts; E0=3.0e9, m0=M_ELECTRON)
    @test length(twline_ref) == length(refpts)
    @test twline_ref[end] isa optics4DUC
    twj_ref = JuTrack.twissring(ring_jt, 0.0, 0, refpts; E0=3.0e9, m0=JuTrack.m_e)
    for i in eachindex(refpts)
        @test isapprox(twline_ref[i].optics_x.beta, twj_ref[i].betax; atol=1e-13)
        @test isapprox(twline_ref[i].optics_x.alpha, twj_ref[i].alphax; atol=1e-13)
        @test isapprox(twline_ref[i].optics_y.beta, twj_ref[i].betay; atol=1e-13)
        @test isapprox(twline_ref[i].optics_y.alpha, twj_ref[i].alphay; atol=1e-13)
    end
end
end
