using Test
using TrackPad
using StaticArrays
using LinearAlgebra

# Ring optics beyond the linear Twiss functions: dispersion, momentum
# compaction, chromaticity to second order, radiation integrals, amplitude
# detuning. Every quantity is checked against an independent reference.

# A dispersive proton ring at β0 = 0.875 so that every β0 factor is visible.
const pbeam = Beam(kinetic=1.0e9, mass=M_PROTON, charge=1.0)
const arc_cell = AbstractElement[
    Quadrupole(0.5, 0.9), Drift(0.6), SBend(1.2, 2π/40; num_int_steps=10), Drift(0.6),
    Sextupole(0.2, 3.0), Drift(0.4), Corrector(0.0, 2e-5, -1e-5),
    Quadrupole(0.5, -0.9), Drift(0.6), SBend(1.2, 2π/40; num_int_steps=10), Drift(0.6),
    Sextupole(0.2, -3.0), Drift(0.4),
]
const arc = Lattice(vcat([copy(arc_cell) for _ in 1:20]...); periodic=true)

# The finite-difference path, checked against independent references; the
# Taylor-map path (the default) is checked the same way in verify_tpsa.jl.
@testset "periodic_twiss returns dispersion, compaction and chromaticity (method=:fd)" begin
    tw = periodic_twiss(arc, pbeam; max_step=0.05, second_order=true, radiation_integrals=true, method=:fd)
    @test tw.method === :fd
    @test tw.length ≈ total_length(arc)

    @test maximum(abs, tw.dx) > 0.5

    # chromaticity is getchrom's (same closed-orbit machinery; the orbit search
    # is re-seeded from the converged orbit, so agreement is to the FD floor)
    ξ = getchrom(arc, pbeam; centered=true, dpp=1e-6, h=1e-8, method=:fd)
    @test tw.chromx ≈ ξ[1] rtol=1e-6
    @test tw.chromy ≈ ξ[2] rtol=1e-6

    # momentum compaction against the dispersion integral (1/C)∮ D/ρ ds: a
    # trapezoid over the 5 cm sampling (accurate to (Δs/ρ)²/12 ≈ 3e-5), and the
    # Simpson-integrated I1 of the radiation integrals, which is exact to 1e-7.
    fine = refine_lattice(arc; sample_integrator_steps=false, max_step=0.05)
    I1 = sum(0.5 * (tw.dx[i] + tw.dx[i+1]) * el.angle for (i, el) in enumerate(fine.elements) if el isa SBend)
    @test tw.alphac ≈ I1 / tw.length rtol=1e-4
    @test tw.radiation.I1 ≈ I1 rtol=1e-4
    # With the correctors' closed orbit the path length also picks up
    # ∮ x₀′ D′ ds, so I1 = αc·C only holds on the axis: check it there.
    arc0 = Lattice([e isa Corrector ? Drift(0.0) : e for e in arc.elements]; periodic=true)
    tw0c = periodic_twiss(arc0, pbeam; radiation_integrals=true, method=:fd)
    @test tw0c.radiation.I1 ≈ tw0c.alphac * tw0c.length rtol=1e-7
    @test tw.slip ≈ tw.alphac - 1 / pbeam.gamma^2
    @test transition_gamma(tw) ≈ 1 / sqrt(tw.alphac)
    @test tw.slip < 0                           # below transition, as γ0 = 2.07 < γ_tr = 5.3

    # ... and against the path length of the off-momentum closed orbit
    function pathlength(δP)
        co = find_closed_orbit_4d(arc, pbeam; dp=δP)
        δE = deltae_from_deltap(δP, pbeam.beta)
        Δz = linepass(arc, SVector(co[1], co[2], co[3], co[4], 0.0, δE), pbeam)[5]
        pc = p0c(pbeam) * (1 + δP); β = pc / hypot(pc, pbeam.mass)
        return β * (tw.length / pbeam.beta - Δz)
    end
    @test tw.alphac ≈ (pathlength(1e-5) - pathlength(-1e-5)) / (2e-5 * tw.length) rtol=1e-4

    # second-order chromaticity: δP² coefficient of a quadratic fit to Q(δP)
    δs = [-2e-4, -1e-4, 0.0, 1e-4, 2e-4]
    q = [TrackPad._closed_orbit_tunes(arc, pbeam, deltae_from_deltap(d, pbeam.beta),
                                      zero(SVector{6,Float64}), 1e-8, Val(:fd))[1] for d in δs]
    A = hcat(ones(5), δs, δs .^ 2)
    cx = A \ first.(q); cy = A \ last.(q)
    @test tw.chromx ≈ cx[2] rtol=1e-5
    @test tw.chrom2x ≈ cx[3] rtol=1e-3
    @test tw.chrom2y ≈ cy[3] rtol=1e-3

    # radiation integrals of an isomagnetic ring
    ρ = 1.2 / (2π/40)
    @test tw.radiation.I2 ≈ 2π / ρ rtol=1e-12
    @test tw.radiation.I3 ≈ 2π / ρ^2 rtol=1e-12
    @test tw.radiation.I5 > 0
    @test tw.radiation.I4 ≈ tw.radiation.I1 / ρ^2 rtol=1e-10   # sector bends, k1 = 0: I4 = ∮D/ρ³

    # the integrals sample the bend interiors themselves, so they do not
    # depend on the output sampling. Compared at identical integration
    # (boundaries only vs one piece per integration step) they agree to
    # the Simpson resolution; the 5 cm pieces above also refine the 4-step quadrupoles'
    # integration, which moves every quantity by the integrator error of the
    # coarse lattice (~1e-4), not by a sampling artefact.
    twb = periodic_twiss(arc, pbeam; radiation_integrals=true, method=:fd)
    twi = periodic_twiss(arc, pbeam; radiation_integrals=true, sample_integrator_steps=true, method=:fd)
    for k in (:I1, :I2, :I3, :I4, :I5)
        @test getfield(twi.radiation, k) ≈ getfield(twb.radiation, k) rtol=1e-7
        @test getfield(tw.radiation, k) ≈ getfield(twb.radiation, k) rtol=2e-4
    end

    # optional fields are nothing unless requested
    tw0 = periodic_twiss(arc, pbeam; method=:fd)
    @test tw0.chrom2x === nothing && tw0.radiation === nothing && tw0.detuning === nothing
    @test tw0.alphac ≈ twi.alphac rtol=1e-9     # step-boundary sampling does not move the compaction
    @test tw0.alphac ≈ tw.alphac rtol=2e-4      # finer integration moves it by the integrator error
end

@testset "periodic_twiss is evaluated about the closed orbit" begin
    # The correctors give this ring a non-zero closed orbit; the optics must be
    # taken there, not about the axis.
    co = find_closed_orbit_4d(arc, pbeam)
    @test maximum(abs, co) > 1e-5
    tw = periodic_twiss(arc, pbeam)
    ref = SVector(co[1], co[2], co[3], co[4], 0.0, 0.0)
    M = transfer_map(arc, pbeam; reference=ref, h=5e-9)
    @test tw.tunex ≈ TrackPad._tune_from_map(M)[1] atol=1e-10   # Taylor map vs finite-difference map
    @test_throws ArgumentError periodic_twiss(Lattice(arc.elements), pbeam)   # open line
    @test_throws ArgumentError periodic_twiss(arc, pbeam; method=:nope)
end

@testset "Amplitude-dependent tune shift" begin
    beam = Beam(3.0e9)
    cell = AbstractElement[Quadrupole(0.5, 0.95), Drift(1.0), Quadrupole(0.5, -0.80), Drift(1.0)]
    mk(k3) = Lattice(vcat([Octupole(0.01, k3; num_int_steps=1)], [copy(cell) for _ in 1:10]...); periodic=true)
    # NAFF on a pure rotation
    q = 0.3141592653
    @test TrackPad._naff_tune([cis(2π * q * n) for n in 0:1023]) ≈ q atol=1e-9
    # The ring alone detunes kinematically (px⁴/8 of the exact drift). The
    # octupole adds ∂Qx/∂Jx = k3 L βx²/(16π), ∂Qx/∂Jy = −k3 L βxβy/(8π) on top.
    base = periodic_twiss(mk(0.0), beam; detuning=true, detuning_turns=2048)
    tw = periodic_twiss(mk(2000.0), beam; detuning=true, detuning_turns=2048)
    βx, βy = tw.betax[1], tw.betay[1]
    k3L = 2000.0 * 0.01
    ana = [k3L*βx^2/(16π)  -k3L*βx*βy/(8π); -k3L*βx*βy/(8π)  k3L*βy^2/(16π)]
    D = tw.detuning .- base.detuning
    @test D[1, 1] ≈ ana[1, 1] rtol=1e-2
    @test D[2, 2] ≈ ana[2, 2] rtol=1e-2
    @test D[1, 2] ≈ ana[1, 2] rtol=1e-3
    @test D[2, 1] ≈ ana[2, 1] rtol=1e-3
    @test tw.detuning[1, 2] ≈ tw.detuning[2, 1] rtol=1e-3     # symmetric
    @test all(base.detuning .> 0)                             # kinematic detuning is positive
    @test_throws ArgumentError periodic_twiss(mk(0.0), beam; detuning=true, detuning_actions=(1e-8,))
end

# ── Cross-code reference ────────────────────────────────────────────────────
# A 24-cell FODO ring (2π of 7.5° sector bends, ρ = 27.5 m, chromatic
# sextupoles, one octupole per cell) evaluated with MAD-X 5.09.03 (twiss,
# chrom), PTC (exact=true, model=1, method=6, nst=20; ptc_normal no=4),
# Xsuite 0.114 (Bend rot-kick-rot, 20 kicks per sextupole/octupole, edge
# 'linear' and 'full') and pyAT (BndMPoleSymplectic4Pass /
# ExactSectorBendPass). Every reference below is the value those codes print;
# the TrackPad numbers are the same physics to the quoted digits, which pins
# the closed-orbit chromaticity, the δP convention, αc, ξ₂, I1–I5 and ∂Q/∂J
# against independent implementations. See docs/src/conventions.md for the
# two model choices that matter here: `SBend` is the AT expanded bend, and
# `ExactSBend` carries the hard-edge dipole fringe that PTC, MAD-X and AT's
# exact bend also apply (Xsuite only with `edge_*_model='full'`).
@testset "Cross-code reference ring (MAD-X, PTC, Xsuite, AT)" begin
    ncell = 24; angle = 2π / (2ncell)
    function xring(bend; steps=20)
        QFH = Quadrupole(0.10, 1.60; num_int_steps=steps); QD = Quadrupole(0.20, -1.45; num_int_steps=steps)
        SF = Sextupole(0.05, 6.0; num_int_steps=steps); SD = Sextupole(0.05, -9.0; num_int_steps=steps)
        OC = Octupole(0.10, 400.0; num_int_steps=steps)
        DQS = Drift(0.05); DQB = Drift(0.50); DQO = Drift(0.05)
        cell = AbstractElement[QFH, DQO, OC, DQO, SF, DQB, bend, DQB, SD, DQS, QD, DQS, SD, DQB, bend, DQB, SF, DQS, QFH]
        Lattice(vcat([copy(cell) for _ in 1:ncell]...); periodic=true)
    end
    ebeam = Beam(3.0e9)

    # AT-model bends: pyAT BndMPoleSymplectic4Pass gives ξ = (12.94235892, 5.19309737);
    # MAD-X synch_1..5 = 5.45046519, 0.22846306, 0.00830715146, 0.00720619203, 0.00104532590
    tw = periodic_twiss(xring(SBend(3.60, angle; num_int_steps=20)), ebeam; radiation_integrals=true)
    @test tw.tunex ≈ 0.35638354 atol=1e-7
    @test tw.tuney ≈ 0.04626899 atol=1e-7
    @test tw.chromx ≈ 12.94235892 rtol=1e-6
    @test tw.chromy ≈ 5.19309737 rtol=1e-6
    @test tw.alphac ≈ 0.0223746519 rtol=1e-7                 # MAD-X alfa, PTC alpha_c
    @test tw.betax[1] ≈ 18.3047072 rtol=1e-7                 # MAD-X / PTC betx at the cell start
    @test tw.dx[1] ≈ 1.32575914 rtol=1e-7
    @test tw.radiation.I1 ≈ 5.45046519 rtol=1e-7
    @test tw.radiation.I2 ≈ 0.22846306484 rtol=1e-9
    @test tw.radiation.I3 ≈ 0.0083071514597 rtol=1e-9
    @test tw.radiation.I4 ≈ 0.00720619203 rtol=1e-7
    @test tw.radiation.I5 ≈ 0.00104532590 rtol=1e-8

    # Exact bends with the hard-edge fringe: PTC dq1, dq2 = 13.4312666, 5.4494198;
    # PTC d²Q/dδ² = 2178.0595, −414.0865; PTC anharmonicities (dQ/dε, ε = 2J)
    # 3219.32, −1065.63, 160.41 → ∂Q/∂J = 6438.6, −2131.3, 320.8 (Xsuite 'full'
    # edges, finite amplitudes: 6428.6, −2131.8, 320.1).
    twe = periodic_twiss(xring(ExactSBend(3.60, angle; num_int_steps=20)), ebeam;
                         second_order=true, detuning=true,
                         detuning_actions=(0.5e-9, 1.0e-9, 1.5e-9, 2.0e-9))
    @test twe.chromx ≈ 13.4312666 rtol=1e-6
    @test twe.chromy ≈ 5.4494198 rtol=1e-6
    @test 2twe.chrom2x ≈ 2178.0595 rtol=1e-4
    @test 2twe.chrom2y ≈ -414.0865 rtol=1e-4
    @test twe.detuning[1, 1] ≈ 6438.6 rtol=3e-3
    @test twe.detuning[1, 2] ≈ -2131.3 rtol=2e-3
    @test twe.detuning[2, 1] ≈ -2131.3 rtol=2e-3
    @test twe.detuning[2, 2] ≈ 320.8 rtol=3e-3

    # Without the dipole fringe: Xsuite (edge 'linear') dqx, dqy = 13.4312567, 5.3818942
    twn = periodic_twiss(xring(ExactSBend(3.60, angle; num_int_steps=20,
                                          fringe_bend_entrance=0, fringe_bend_exit=0)), ebeam)
    @test twn.chromx ≈ 13.4312567 rtol=1e-6
    @test twn.chromy ≈ 5.3818942 rtol=1e-6

    # The same ring for a 1 GeV-kinetic proton (β0 = 0.875): every δP quantity
    # is species independent; the δE derivatives are the δP ones over β0, which
    # is what MAD-X prints as DX and DQ1 for a proton (1.51510887, 15.3508).
    pbm = Beam(kinetic=1.0e9, mass=M_PROTON, charge=1.0)
    lat = xring(ExactSBend(3.60, angle; num_int_steps=20))
    twp = periodic_twiss(lat, pbm)
    twpE = periodic_twiss(lat, pbm; wrt=:deltae)
    @test twp.chromx ≈ twe.chromx rtol=1e-6
    @test twp.chromy ≈ twe.chromy rtol=1e-6
    @test twp.alphac ≈ tw.alphac rtol=1e-6
    @test twpE.dx[1] ≈ 1.51510887 rtol=1e-6
    @test twpE.chromx ≈ twp.chromx / pbm.beta rtol=1e-6      # (finite-difference floor)
    @test twp.slip ≈ twp.alphac - 1 / pbm.gamma^2
end

@testset "twiss: one interface for rings and lines" begin
    ebeam = Beam(3.0e9)
    cell = AbstractElement[Quadrupole(0.5, 0.9; num_int_steps=4), Drift(0.6), SBend(1.2, 2π/40; num_int_steps=4),
                           Drift(0.6), Quadrupole(0.5, -0.9; num_int_steps=4), Drift(0.6),
                           SBend(1.2, 2π/40; num_int_steps=4), Drift(0.6)]
    ring = Lattice(vcat([copy(cell) for _ in 1:20]...); periodic=true)
    line = Lattice(copy(cell))

    tw = twiss(ring, ebeam)
    @test tw isa TwissResult && tw.periodic && tw.method === :tpsa
    twp = periodic_twiss(ring, ebeam)
    @test tw.betax == twp.betax && tw.dx == twp.dx && tw.chromx == twp.chromx && tw.alphac == twp.alphac
    @test tw.length ≈ total_length(ring)
    @test_throws ArgumentError twiss(line, ebeam)                # open line needs an entrance
    @test_throws ArgumentError periodic_twiss(line, ebeam)

    # the periodic solution transported once around the ring reproduces itself,
    # dispersion included (entrance given as optics4DUC with eta/etap ...)
    ent = optics4DUC(optics2D(tw.betax[1], tw.alphax[1], 0.0, tw.dx[1], tw.dpx[1]),
                     optics2D(tw.betay[1], tw.alphay[1], 0.0, tw.dy[1], tw.dpy[1]))
    tr = transport_twiss(ring, ebeam, ent)
    @test !tr.periodic && tr.alphac === nothing && tr.chromx === nothing
    @test tr.betax ≈ tw.betax rtol=1e-10
    @test tr.dx ≈ tw.dx atol=1e-10
    @test tr.dpx ≈ tw.dpx atol=1e-10
    @test tr.tunex ≈ mod(tw.mux[end] / 2π, 1) + floor(tw.mux[end] / 2π)   # total phase advance / 2π
    @test tr.tunex ≈ tw.mux[end] / 2π
    @test mod(tr.tunex, 1) ≈ tw.tunex atol=1e-10
    # ... or as another TwissResult, whose exit becomes the entrance (chaining)
    half = Lattice(vcat([copy(cell) for _ in 1:10]...))
    t1 = twiss(half, ebeam; entrance=tw)
    t2 = twiss(half, ebeam; entrance=t1)
    @test t1.betax[end] ≈ tw.betax[81] rtol=1e-10
    @test t2.betax[end] ≈ tw.betax[1] rtol=1e-10
    @test t2.dx[end] ≈ tw.dx[1] atol=1e-10
    @test t2.mux[end] - t1.mux[1] ≈ tw.mux[end] rtol=1e-10   # phases continue across the chain
    @test_throws ArgumentError twiss(half, ebeam; entrance=tw, second_order=true)
    @test_throws ArgumentError twiss(half, ebeam; entrance=tw, detuning=true)
    @test_throws ArgumentError transition_gamma(t1)
    # radiation integrals of the line equal the ring's (same optics, one turn)
    twr = twiss(ring, ebeam; radiation_integrals=true)
    trr = transport_twiss(ring, ebeam, ent; radiation_integrals=true)
    @test trr.radiation.I1 ≈ twr.radiation.I1 rtol=1e-8
    @test trr.radiation.I5 ≈ twr.radiation.I5 rtol=1e-8

    # `slices` oversamples every element of finite length without touching the
    # lattice. Each piece integrates in one step, so with 10-step magnets
    # `slices=10` reproduces the boundary values to roundoff ...
    cell10 = AbstractElement[e isa Drift ? e : TrackPad._replace_element_fields(e, (num_int_steps=10,)) for e in cell]
    ring10 = Lattice(vcat([copy(cell10) for _ in 1:20]...); periodic=true)
    tw10 = twiss(ring10, ebeam)
    tws = twiss(ring10, ebeam; slices=10)
    @test length(tws.s) == 10 * length(ring10) + 1
    @test issorted(tws.s)
    @test tws.tunex ≈ tw10.tunex atol=1e-12
    @test tws.betax[1:10:end] ≈ tw10.betax rtol=1e-11
    @test tws.dx[1:10:end] ≈ tw10.dx atol=1e-12
    @test tws.alphac ≈ tw10.alphac rtol=1e-10
    @test tws.chromx ≈ tw10.chromx rtol=1e-9
    # ... while more slices than steps refine the integration: the 4-step ring
    # moves by its own integrator error (1e-4 here), towards the 10-step one
    tws4 = twiss(ring, ebeam; slices=10)
    @test abs(tws4.tunex - tw.tunex) < 3e-4
    @test abs(tws4.tunex - tw10.tunex) < abs(tw.tunex - tw10.tunex)
    # slices never reduce a configured integrator: 2 slices of a 4-step magnet stay 4 pieces
    tw2 = twiss(ring, ebeam; slices=2)
    @test length(tw2.s) == (4 * 4 + 4 * 2) * 20 + 1
    # max_step now also splits thick multipoles
    twm = twiss(ring, ebeam; max_step=0.1)
    @test length(twm.s) == (2 * 5 + 2 * 12 + 4 * 6) * 20 + 1
    @test_throws ArgumentError twiss(ring, ebeam; slices=0)
end
