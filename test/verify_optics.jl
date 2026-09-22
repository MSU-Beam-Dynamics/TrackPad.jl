using Test
using TrackPad
using LinearAlgebra
using StaticArrays
Base.include(@__MODULE__, joinpath(@__DIR__, "convention_helpers.jl"))

# Simple periodic cell to validate optics APIs
D = Drift(1.0; name="D")
QF = Quadrupole(0.5, 0.6; name="QF", num_int_steps=8)
QD = Quadrupole(0.5, -0.6; name="QD", num_int_steps=8)

ring = Lattice([D, QF, D, QD]; periodic=true)
beam = jutrack_beam(3.0e9)   # JuTrack calls below use energy=3.0e9 (kinetic)

@testset "Open line and periodic ring boundaries" begin
    line = Lattice([D, QF])
    @test !isperiodic(line)
    @test isperiodic(ring)
    @test size(transfer_map(line, beam)) == (6, 6)
    entrance = optics4DUC(2.0, 0.1, 3.0, -0.2)
    transported = transport_twiss(line, beam, entrance)
    @test transported isa TwissResult
    @test !transported.periodic && transported.alphac === nothing
    @test length(transported.s) == length(line) + 1
    @test transported.betax[1] == entrance.optics_x.beta
    @test transported.betay[1] == entrance.optics_y.beta
    @test periodic_twiss(ring, beam).tunex == twissline(ring, beam).tunex
    dispersion = periodic_twiss(ring, beam)        # a bend-free ring has no dispersion
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
    @test_throws ArgumentError find_closed_orbit_4d(line, beam)
    @test_throws ArgumentError track(
        line, zero(SVector{6,Float64}), beam; nturns=2,
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
    low_energy_beam = Beam(kinetic=50.0e6, mass=M_PROTON, charge=1.0)
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

# Frozen JuTrack references for the same ring, built as
# DRIFT(1.0), KQUAD(len=0.5, k1=0.6, NumIntSteps=8), DRIFT(1.0),
# KQUAD(len=0.5, k1=-0.6, NumIntSteps=8) at energy=3.0e9 (kinetic).
tune_jt = jutrack_reference("optics/tune")
chrom_jt = jutrack_reference("optics/chrom")

@testset "TrackPad Optics API" begin
    qx, qy = gettune(ring, beam)
    @test isfinite(qx)
    @test isfinite(qy)
    @test 0.0 <= qx <= 1.0
    @test 0.0 <= qy <= 1.0
    @test qx == tune_jt[1]
    @test qy == tune_jt[2]

    # JuTrack differentiates with respect to its own stored coordinate. It
    # launches the off-momentum particle on-axis, but this ring has no
    # dispersion, so its off-momentum closed orbit *is* the axis and the two
    # conventions coincide exactly.
    ξx, ξy = getchrom(ring, beam; dp=0.0, wrt=:deltae, method=:fd)
    @test isfinite(ξx)
    @test isfinite(ξy)
    @test ξx == chrom_jt[1]
    @test ξy == chrom_jt[2]
    # the default Taylor-map chromaticity is the exact derivative; JuTrack's
    # forward difference (dpp = 1e-8) is off by its truncation, ~1e-8
    ξxt, ξyt = getchrom(ring, beam; dp=0.0, wrt=:deltae)
    @test ξxt ≈ chrom_jt[1] atol=1e-7
    @test ξyt ≈ chrom_jt[2] atol=1e-7

    @test_throws ArgumentError getchrom(ring, beam; wrt=:bogus)

    ξx_centered, ξy_centered = getchrom(
        ring, beam;
        dpp=1.0e-5, centered=true, method=:fd,
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
    # `E0` is the total energy; `beam` was built from JuTrack's kinetic 3 GeV.
    m66 = fastfindm66(ring, 0.0; E0=beam.energy, m0=M_ELECTRON)
    m66_ref = one_turn_map(ring, beam; h=1.5e-8)
    @test m66 == m66_ref

    m66_ord0 = findm66(ring, 0.0, 0; E0=beam.energy, m0=M_ELECTRON)
    @test m66_ord0 == m66

    m66_jt = jutrack_reference("optics/m66")
    # The transverse block agrees to the bit. M56 of this bend-free ring is the
    # kinematic L/γ² ≈ 9e-8, which JuTrack's drift computes as the difference of
    # two O(L) numbers; its finite-difference value carries ~5e-9 of roundoff
    # that TrackPad's cancellation-free drift no longer reproduces.
    @test m66[1:4, 1:4] == m66_jt[1:4, 1:4]
    @test isapprox(m66, m66_jt; atol=1e-8)
    @test isapprox(m66[5, 6], sum(get_length(e) for e in ring.elements) / beam.gamma^2; rtol=1e-3)

    refpts = [2, 4]
    mref = fastfindm66_refpts(ring, 0.0, refpts; E0=3.0e9, m0=M_ELECTRON)
    @test size(mref) == (6, 6, 2)
    @test isapprox(mref[:, :, 2] * mref[:, :, 1], m66; atol=5e-7)

    op = periodicEdwardsTengTwiss(ring, 0.0, 0; E0=3.0e9, m0=M_ELECTRON)
    @test op isa optics4DUC
    @test op.optics_x.beta > 0
    @test op.optics_y.beta > 0

    # [betax, alphax, betay, alphay] from JuTrack.periodicEdwardsTengTwiss
    op_jt = jutrack_reference("optics/edwards_teng")
    @test isapprox(op.optics_x.beta, op_jt[1]; atol=1e-13)
    @test isapprox(op.optics_x.alpha, op_jt[2]; atol=1e-13)
    @test isapprox(op.optics_y.beta, op_jt[3]; atol=1e-13)
    @test isapprox(op.optics_y.alpha, op_jt[4]; atol=1e-13)

    twr = twissring(ring, 0.0, 0; E0=3.0e9, m0=M_ELECTRON)
    @test twr isa TwissResult
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
    # one [betax, alphax, betay, alphay] per refpt, from JuTrack.twissring
    twj_ref = jutrack_reference("optics/twissring_refpts")
    for i in eachindex(refpts)
        @test isapprox(twline_ref[i].optics_x.beta, twj_ref[i][1]; atol=1e-13)
        @test isapprox(twline_ref[i].optics_x.alpha, twj_ref[i][2]; atol=1e-13)
        @test isapprox(twline_ref[i].optics_y.beta, twj_ref[i][3]; atol=1e-13)
        @test isapprox(twline_ref[i].optics_y.alpha, twj_ref[i][4]; atol=1e-13)
    end
end

@testset "Energy-derivative convention (δP vs δE)" begin
    # Exact inverses, and stable at finite-difference-sized arguments: a naive
    # `sqrt(1 + small) - 1` loses ~8 digits at δ ~ 1e-8 and fails this.
    for β0 in (0.5, 0.875, 0.99, 1.0 - 1.0e-9), δP in (1.0e-12, 1.0e-8, 1.0e-3, 0.1)
        δE = deltae_from_deltap(δP, β0)
        @test deltap_from_deltae(δE, β0) ≈ δP rtol=1e-14
        @test δE ≈ β0 * δP rtol=(10δP + 1e-14)          # first-order limit
    end
    # The exact relation, not the linear one.
    let β0 = 0.875, δP = 0.2
        @test deltae_from_deltap(δP, β0) ≈ sqrt(1/β0^2 + δP*(2 + δP)) - 1/β0 rtol=1e-13
    end

    # A 1 GeV-kinetic proton makes β0 a 12.5% effect, far above the FD noise
    # floor, so this test fails if the conversion is dropped. (At 3 GeV
    # electrons β0 = 1 - 1.4e-8 and no tolerance could tell them apart.)
    pbeam = Beam(kinetic=1.0e9, charge=1, mass=M_PROTON)
    @test 0.87 < pbeam.beta < 0.88
    ξe = getchrom(ring, pbeam; centered=true, dpp=1.0e-6, wrt=:deltae, method=:fd)
    ξp = getchrom(ring, pbeam; centered=true, dpp=1.0e-6, method=:fd)
    @test ξp[1] ≈ pbeam.beta * ξe[1] rtol=1e-6
    @test ξp[2] ≈ pbeam.beta * ξe[2] rtol=1e-6
    @test !isapprox(ξp[1], ξe[1]; rtol=1e-3)

    # Dispersion follows the same rule.
    dring = Lattice(AbstractElement[
        Quadrupole(0.5, 0.9), Drift(0.4), SBend(1.0, 2π/80; num_int_steps=4),
        Drift(0.4), Quadrupole(0.5, -0.9), Drift(0.4),
        SBend(1.0, 2π/80; num_int_steps=4), Drift(0.4),
    ]; periodic=true)
    De = periodic_twiss(dring, pbeam; wrt=:deltae)
    Dp = periodic_twiss(dring, pbeam)
    @test maximum(abs, De.dx) > 0.1
    @test Dp.dx ≈ pbeam.beta .* De.dx rtol=1e-10
    @test Dp.dpx ≈ pbeam.beta .* De.dpx rtol=1e-10
    @test_throws ArgumentError periodic_twiss(dring, pbeam; wrt=:bogus)

    # Scalar `dp` inputs are δP at every entry point (the convention outer
    # codes speak); coordinate vectors (`orb`, `reference`) stay in δE.
    δP = 5.0e-3
    δE = deltae_from_deltap(δP, pbeam.beta)
    @test !isapprox(δE, δP; rtol=1e-3)
    @test findm66(dring, δP, 0; E0=pbeam.energy, m0=pbeam.mass) ==
          findm66(dring, δE, 0; E0=pbeam.energy, m0=pbeam.mass, wrt=:deltae)
    @test fastfindm66(dring, δP; E0=pbeam.energy, m0=pbeam.mass) ==
          fastfindm66(dring, δE; E0=pbeam.energy, m0=pbeam.mass, wrt=:deltae)

    co_p = find_closed_orbit_4d(dring, pbeam; dp=δP)
    @test co_p ≈ find_closed_orbit_4d(dring, pbeam; dp=δE, wrt=:deltae) atol=1e-14
    # ... and the conversion is not a no-op for a 1 GeV proton
    @test !isapprox(co_p, find_closed_orbit_4d(dring, pbeam; dp=δP, wrt=:deltae);
                    atol=1e-6)

    # `orb` is a coordinate vector: `wrt` must not touch it.
    orb = [0.0, 0.0, 0.0, 0.0, 0.0, δE]
    @test fastfindm66(dring, 0.0; E0=pbeam.energy, m0=pbeam.mass, orb=orb) ==
          fastfindm66(dring, 0.0; E0=pbeam.energy, m0=pbeam.mass, orb=orb,
                      wrt=:deltae)

    # twissring's dp is likewise δP
    @test twissring(dring, δP, 0; E0=pbeam.energy, m0=pbeam.mass).betax ==
          twissring(dring, δE, 0; E0=pbeam.energy, m0=pbeam.mass,
                    wrt=:deltae).betax
end

@testset "Chromaticity is measured about the closed orbit" begin
    mkring(k2) = Lattice(vcat([AbstractElement[
        Quadrupole(0.5, 0.9), Drift(0.6), SBend(1.2, 2π/80; num_int_steps=10),
        Drift(0.6), Sextupole(0.2, k2), Drift(0.6),
        Quadrupole(0.5, -0.9), Drift(0.6), SBend(1.2, 2π/80; num_int_steps=10),
        Drift(0.6), Sextupole(0.2, -k2), Drift(0.6),
    ] for _ in 1:20]...); periodic=true)
    sring = mkring(3.0)
    bare = mkring(0.0)

    ξ_co = getchrom(sring, beam; centered=true, dpp=1.0e-6, method=:fd)
    # Reference for the on-axis launch (JuTrack's convention), computed by
    # hand from the tune of the Jacobian at (0,0,0,0,0,±δ): genuinely a
    # different number, which is why TrackPad does not offer it.
    tune_ax(δP) = TrackPad._tune_from_map(findm66(sring, δP, 0; E0=beam.energy, m0=beam.mass))
    ξ_ax = ((tune_ax(1e-6)[1] - tune_ax(-1e-6)[1]) / 2e-6,
            (tune_ax(1e-6)[2] - tune_ax(-1e-6)[2]) / 2e-6)
    @test !isapprox(ξ_co[1], ξ_ax[1]; rtol=1e-4)
    @test !isapprox(ξ_co[2], ξ_ax[2]; rtol=1e-4)
    @test_throws ArgumentError getchrom(sring, beam; method=:bogus)
    @test_throws ArgumentError getchrom(Lattice(sring.elements), beam)   # open line: no closed orbit

    # Sextupole feed-down identity, Δξx = +(1/4π)Σ βx (k2 L) D and
    # Δξy = -(1/4π)Σ βy (k2 L) D. This holds only if getchrom,
    # periodic_twiss and getchrom share one convention, so it pins the
    # closed-orbit default and the δP dispersion together.
    ξ0 = getchrom(bare, beam; centered=true, dpp=1.0e-6, method=:fd)
    tw = periodic_twiss(sring, beam)
    dsp = tw
    ax = 0.0
    ay = 0.0
    for (i, el) in enumerate(sring.elements)
        el isa Sextupole || continue
        bx = (tw.betax[i] + tw.betax[i + 1]) / 2
        by = (tw.betay[i] + tw.betay[i + 1]) / 2
        D = (dsp.dx[i] + dsp.dx[i + 1]) / 2
        ax += bx * (el.k2 * el.L) * D / (4π)
        ay -= by * (el.k2 * el.L) * D / (4π)
    end
    @test ax ≈ ξ_co[1] - ξ0[1] rtol=2e-3
    @test ay ≈ ξ_co[2] - ξ0[2] rtol=2e-3
    # The on-axis launch satisfies it far less well.
    @test abs(ax / (ξ_ax[1] - ξ0[1]) - 1) > 5 * abs(ax / (ξ_co[1] - ξ0[1]) - 1)
end
