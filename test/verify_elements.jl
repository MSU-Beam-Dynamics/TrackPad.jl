using Test
using TrackPad
using LinearAlgebra
using StaticArrays
Base.include(@__MODULE__, joinpath(@__DIR__, "convention_helpers.jl"))

const PARITY_ATOL = 1e-15
# JuTrack uses mutually inconsistent finite-beta longitudinal normalizations
# across element families. Exact parity is meaningful only in the common
# ultrarelativistic limit; canonical finite-beta behavior is tested below.
const ENERGY_VAL = 1.0e15

const particles_initial = PARITY_PARTICLES

function _track_tp(elem; energy::Float64 = ENERGY_VAL, particles = particles_initial)
    coords = copy(particles)
    lost_flags = zeros(Int, size(coords, 1))
    linepass!(coords, Lattice([elem]), jutrack_beam(energy), lost_flags)
    return coords
end

# JuTrack's exact-bend body evaluates x_new = (pz_new − pzmx·cos + px·sin − 1)/h
# directly, which cancels O(1) terms and leaves ~ε/h ≈ 1e-15 m of roundoff per
# step; TrackPad's `exact_bend_body` is algebraically identical but
# cancellation-free, so the two agree only to JuTrack's roundoff.
const EXACT_BEND_ATOL = 1e-13

# Every case is checked against the JuTrack output frozen in
# test/jutrack_reference.jl under the key "element/<name>".
const PARITY_CASES = String[]

function _assert_parity(name::AbstractString, tp_elem; energy::Float64 = ENERGY_VAL,
                        particles = particles_initial, atol::Float64 = PARITY_ATOL)
    push!(PARITY_CASES, name)
    tp = _track_tp(tp_elem; energy = energy, particles = particles)
    jt = jutrack_reference("element/" * name)
    diff_norm = norm(tp - jt)
    diff_max = maximum(abs.(tp - jt))
    @testset "$name" begin
        @test diff_norm < atol
        @test diff_max < atol
        @test isapprox(tp, jt, atol = atol)
    end
end

@testset "Element JuTrack Parity" begin
    begin
        # Core linear/multipole/bend elements.
        _assert_parity("Marker", Marker())
        _assert_parity("Drift", Drift(0.7))
        _assert_parity("Quadrupole", Quadrupole(0.4, 1.3; num_int_steps = 10))
        _assert_parity("Sextupole", Sextupole(0.4, 2.0; num_int_steps = 10))
        _assert_parity("Octupole", Octupole(0.4, -3.0; num_int_steps = 10))
        _assert_parity("ThinMultipole", ThinMultipole(0.0, [0.0, 0.0, 0.0, 0.0], [0.0, 0.2, -0.1, 0.05]; max_order = 3))
        _assert_parity("SBend", SBend(0.9, 0.15, 0.03, 0.02; num_int_steps = 10))
        _assert_parity("RBend", RBend(0.9, 0.15; num_int_steps = 10))
        _assert_parity("ExactSBend", ExactSBend(0.9, 0.15, 0.03, 0.02; num_int_steps = 10); atol = EXACT_BEND_ATOL)
        _assert_parity("ERBend", ERBend(0.9, 0.15; num_int_steps = 10); atol = EXACT_BEND_ATOL)

        # RF/cavity and related maps.
        _assert_parity("RFCavity", RFCavity(0.3, 2.0e6, 500.0e6, 0.002; h = 1200.0, philag = 0.3, energy = ENERGY_VAL))
        _assert_parity("CrabCavity", CrabCavity(0.3; volt = 2.0e6, freq = 500.0e6, phi = 0.1, energy = ENERGY_VAL))
        _assert_parity("AccelCavity", AccelCavity(0.0; volt = 2.5e6, freq = 500.0e6, h = 1200.0, phis = 0.2, energy = ENERGY_VAL))
        # Auxiliary helpers.
        _assert_parity("Solenoid", Solenoid(0.5, 0.8))
        _assert_parity("Corrector", Corrector(0.4, 1.5e-4, -2.2e-4))
        _assert_parity("HKicker", HKicker(L = 0.0, xkick = 2.5e-4))
        _assert_parity("VKicker", VKicker(L = 0.0, ykick = -1.5e-4))
        _assert_parity("YRotation", YRotation(0.0; angle = 0.02))
        _assert_parity("Wiggler", Wiggler(1.2; lw = 0.2, Bmax = 0.8, Nsteps = 8))
        # Vertical wiggler harmonics need kx != 0 (the field decays along x);
        # degenerate all-zero wave-vector blocks produce NaN in both codes.
        _assert_parity("Wiggler vertical harmonics", Wiggler(1.2; lw = 0.2, Bmax = 0.8, Nsteps = 8, By = Int[], Bx = [1, 1, 1, 0, 1, 0]))
        _assert_parity("Wiggler mixed harmonics", Wiggler(1.2; lw = 0.2, Bmax = 0.8, Nsteps = 8, By = [1, 1, 0, 1, 1, 0], Bx = [1, 2, 1, 0, 1, 0]))

        # Translation longitudinal shift follows TrackPad's negated z axis and
        # is drift-consistent; with dx = dy = 0 it must equal a Drift of ds
        # exactly. JuTrack's TRANSLATION mixes an opposing x-sign with its
        # c(t-t0) update and is not drift-consistent, so only the in-package
        # invariant is asserted here (see conventions.md).
        @testset "Translation drift consistency" begin
            ds = 3.0e-3
            beam_tb = jutrack_beam(ENERGY_VAL)
            coords_t = copy(particles_initial)
            lost_t = zeros(Int, size(coords_t, 1))
            linepass!(coords_t, Lattice([Translation(0.0; dx = 1.0e-3, dy = -2.0e-3, ds = ds)]),
                      beam_tb, lost_t)
            coords_d = copy(particles_initial)
            linepass!(coords_d, Lattice([Drift(ds)]), jutrack_beam(ENERGY_VAL), zeros(Int, size(coords_d, 1)))
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
        _assert_parity("SBend gradient auto-order", SBend(0.9, 0.15, 0.03, 0.02; num_int_steps = 10,
                  polynom_b = [0.0, 0.3, 0.0, 0.0]))
        _assert_parity("RBend gradient auto-order", RBend(0.9, 0.15; num_int_steps = 10, polynom_b = [0.0, 0.25, 0.0, 0.0]))

        # Forest (13.29) multipole entrance/exit fringes.
        _assert_parity("Quadrupole fringes", Quadrupole(0.4, 1.3; num_int_steps = 10, fringe_entrance = 1, fringe_exit = 1))
        _assert_parity("Sextupole fringes", Sextupole(0.4, 2.0; num_int_steps = 10, fringe_entrance = 1))
        _assert_parity("Octupole fringes", Octupole(0.4, -3.0; num_int_steps = 10, fringe_exit = 1))
        _assert_parity("ThinMultipole fringes", ThinMultipole(0.0, [0.0, 0.0, 0.0, 0.0], [0.0, 0.2, -0.1, 0.05];
                          max_order = 3, fringe_entrance = 1))
        _assert_parity("SBend quad-fringe gates", SBend(0.9, 0.15, 0.03, 0.02; num_int_steps = 10,
                  fringe_quad_entrance = 1, fringe_quad_exit = 1))
        _assert_parity("ExactSBend quad-fringe ordering", ExactSBend(0.9, 0.15, 0.03, 0.02; num_int_steps = 10,
                       fringe_quad_entrance = 1, fringe_quad_exit = 1); atol = EXACT_BEND_ATOL)
        _assert_parity("Sextupole kick angle with fringes", Sextupole(0.4, 2.0; num_int_steps = 10,
                      kick_angle = [1.2e-4, -0.8e-4],
                      fringe_entrance = 1, fringe_exit = 1))
        _assert_parity("Octupole kick angle with fringes", Octupole(0.4, -3.0; num_int_steps = 10,
                     kick_angle = [-0.7e-4, 1.1e-4],
                     fringe_entrance = 1, fringe_exit = 1))

        # Space-charge canonical family.
        _assert_parity("DriftSC", DriftSC(0.5; a = 0.01, b = 0.02, Nl = 12, Nm = 14, Nsteps = 2))
        _assert_parity("QuadrupoleSC", QuadrupoleSC(0.4; k1 = 0.0, a = 0.01, b = 0.02, num_int_steps = 10))
        _assert_parity("SextupoleSC", SextupoleSC(0.3; k2 = 2.4, a = 0.01, b = 0.02, num_int_steps = 10))
        _assert_parity("OctupoleSC", OctupoleSC(0.2; k3 = 3.6, a = 0.01, b = 0.02, num_int_steps = 10))
        _assert_parity("SBendSC", SBendSC(1.0, 0.0, 0.0, 0.0; a = 0.01, b = 0.02, num_int_steps = 10))
        _assert_parity("RBendSC", RBendSC(1.2, 0.0))
        _assert_parity("LBend", LBend(0.7, 0.0; K = 0.0))
        _assert_parity("SpaceCharge", SpaceCharge(0.9; effective_len = 0.4, Nl = 11, Nm = 9, a = 0.015, b = 0.017))

        # JuTrack has a Float64 collective RLC pass, but its Beam-owned grid,
        # normalization, and longitudinal coordinate differ from TrackPad's.
        # A zero wake therefore checks tracking plumbing, while the Green
        # functions themselves are compared directly at nonzero strength.
        tp_rlc_zero = LongitudinalRLCWake(freq = 1.0e9, Rshunt = 0.0,
                                          Q0 = 1.2, scale = 1.0)
        _assert_parity("LongitudinalRLCWake", tp_rlc_zero)
        tp_rlc = LongitudinalRLCWake(freq = 1.0e9, Rshunt = 1.0e6, Q0 = 1.2)
        for (t, w) in zip(jutrack_reference("wake/rlc_delays"), jutrack_reference("wake/rlc"))
            @test wakefieldfunc_RLCWake(tp_rlc, t) ≈ w rtol = 1.0e-15
        end
        # StrongGaussianBeam is not compared with JuTrack: JuTrack's beam-beam
        # code is a placeholder. See test/verify_beambeam.jl.
    end
end

@testset "Every parity case has frozen reference data" begin
    # A case added here without regenerating test/jutrack_reference.jl, or a
    # reference left behind by a deleted case, is a silent hole in the parity
    # net; both fail here instead.
    recorded = Set(k[9:end] for k in keys(JUTRACK_REFERENCE) if startswith(k, "element/"))
    @test Set(PARITY_CASES) == recorded
    @test length(PARITY_CASES) == length(unique(PARITY_CASES))
end

@testset "Wiggler constructor validation" begin
    @test_throws ArgumentError Wiggler(1.2)
    @test_throws ArgumentError Wiggler(1.2; lw = 0.2, Nsteps = 0)
    @test_throws ArgumentError Wiggler(1.2; lw = 0.2, energy = M_ELECTRON)
    @test_throws ArgumentError Wiggler(1.2; lw = 0.2, energy = 0.5e9, mass = M_PROTON)  # below rest mass
    @test_throws ArgumentError Wiggler(1.2; lw = 0.2, energy = 1e9, charge = 0.0)
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
    beam = jutrack_beam(ENERGY_VAL)
    delta_e = 2.0e-3
    input = SVector(0.0, 0.0, 0.0, 0.0, 1.0e-3, delta_e)
    output = pass!(elem, input, inv(beam.beta))
    eta = elem.alphac - (1 - beam.beta^2)
    # Δz = −C η δP / β0 with C = 2πh/k and δP = δE/β0  ⇒  −C η δE / β0²
    expected_z = input[5] - (2π * rf.h * eta / (rf.k * beam.beta^2)) * delta_e
    @test output[5] ≈ expected_z atol=1.0e-15
    @test output[5] < input[5]

    # The slip map must reproduce a real ring's one-turn slip at finite β0:
    # compare against the off-momentum closed orbit of a dispersive proton ring
    # whose αc is taken from the dispersion integral ∮D/ρ ds.
    pbeam = Beam(kinetic=1.0e9, mass=M_PROTON, charge=1.0)
    ring = Lattice(vcat([AbstractElement[
        Quadrupole(0.5, 0.9), Drift(0.6), SBend(1.2, 2π/40; num_int_steps=10), Drift(0.6),
        Quadrupole(0.5, -0.9), Drift(0.6), SBend(1.2, 2π/40; num_int_steps=10), Drift(0.6),
    ] for _ in 1:20]...); periodic=true)
    C0 = total_length(ring)
    disp = periodic_twiss(ring, pbeam; max_step=0.02)
    fine = refine_lattice(ring; sample_integrator_steps=false, max_step=0.02)
    I1 = sum(0.5 * (disp.dx[i] + disp.dx[i+1]) * el.angle for (i, el) in enumerate(fine.elements) if el isa SBend)
    αc = I1 / C0
    δP = 1.0e-5
    δE = deltae_from_deltap(δP, pbeam.beta)
    co = find_closed_orbit_4d(ring, pbeam; dp=δP)
    Δz_ring = linepass(ring, SVector(co[1], co[2], co[3], co[4], 0.0, δE), pbeam)[5]
    frf = 100 * TrackPad.C_LIGHT / C0                        # h = 100 fits the ring
    slip = LongitudinalRFMap(αc, RFCavity(0.0, 1.0e6, frf; h=100.0, energy=pbeam.energy, charge=pbeam.charge))
    Δz_map = pass!(slip, SVector(0.0, 0.0, 0.0, 0.0, 0.0, δE), inv(pbeam.beta))[5]
    @test Δz_map ≈ Δz_ring rtol=2e-4     # second order in δ and the ∮ quadrature
    @test !isapprox(Δz_map, Δz_ring * pbeam.beta; rtol=1e-2)   # the old β0⁻¹ form is 12.5% off
end

@testset "Finite-beta canonical maps" begin
    beam = Beam(kinetic=50.0e6, mass=M_PROTON, charge=1.0)
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

    p0c = TrackPad.p0c(beam)
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
    beam = jutrack_beam(ENERGY_VAL)

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
        beam_e = jutrack_beam(ENERGY_VAL)
        bunch_charge = -1.0e-9
        nmacro = 1000
        p0c = TrackPad.p0c(beam_e)
        sc = physical_wake_scale(beam_e, bunch_charge, nmacro)
        # W [V/C] * Qmacro [C] * q/e gives an energy change in eV.
        expected = beam_e.charge * (bunch_charge / nmacro) / p0c
        @test sc ≈ expected rtol = 1.0e-15
        @test sc > 0
        @test_throws ArgumentError physical_wake_scale(beam_e, 1.0e-9, 0)
        @test_throws ArgumentError physical_wake_scale(beam_e, Inf, 1000)
        @test_throws ArgumentError physical_wake_scale(Beam(kinetic=0.0), -1.0e-9, 1000)
        @test_throws ArgumentError physical_wake_scale(
            jutrack_beam(ENERGY_VAL; charge = Inf), -1.0e-9, 1000)
        @test physical_wake_scale(beam_e, 2 * bunch_charge, nmacro) ≈ 2 * sc

        # Like-sign electron and positive-charge bunches both decelerate for a
        # positive wake; inconsistent source/test signs reverse the kick.
        beam_p = jutrack_beam(ENERGY_VAL; charge = 1.0)
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
    wide = jutrack_reference("aperture/particles")
    rap = [-5.0e-4, 5.0e-4, -5.0e-4, 5.0e-4, 0.0, 0.0]
    eap = [1.0e-3, 8.0e-4, 0.0, 0.0, 0.0, 0.0]

    function run_aperture(elem_tp)
        c = copy(wide)
        lf = zeros(Int, size(c, 1))
        linepass!(c, Lattice([elem_tp]), jutrack_beam(ENERGY_VAL), lf)
        return c, lf
    end

    cases = [
        ("Drift rectangular", Drift(0.7; r_apertures = rap)),
        ("Drift elliptical",  Drift(0.7; e_apertures = eap)),
        ("SBend rectangular", SBend(0.9, 0.15; num_int_steps = 10, r_apertures = rap)),
    ]
    @testset "$name" for (name, etp) in cases
        c, lf = run_aperture(etp)
        @test lf == jutrack_reference("aperture/$name/lost")
        @test sum(lf) == 2  # exactly the two out-of-aperture macroparticles
        @test c ≈ jutrack_reference("aperture/$name/r") atol = PARITY_ATOL
    end

    # Without apertures nobody is lost.
    c, lf = run_aperture(Drift(0.7))
    @test lf == zeros(Int, length(lf))
    @test jutrack_reference("aperture/Drift open/lost") == zeros(Int, length(lf))
end

@testset "Element Gaps" begin
    @test_skip false # JuTrack has no Float64 pass! for LongitudinalWake.
    # StrongThinGaussianBeam / StrongGaussianBeam are verified from first
    # principles in test/verify_beambeam.jl; JuTrack's beam-beam is a placeholder.
end

@testset "Wiggler is normalised to the reference particle" begin
    # The map depends on the species only through γ = E/m and the wiggler
    # parameter |q|·Bmax·lw/m, so a proton wiggler equals an electron wiggler
    # at the same γ with Bmax scaled by m_e/m_p (non-radiating).
    γ = 20.0
    r0 = SVector(1.0e-3, 2.0e-4, -8.0e-4, 1.5e-4, 1.0e-3, 5.0e-4)
    By = [1.0, 0.5, 0.0, 1.0, 1.0, 0.3]
    wp = Wiggler(1.2; lw = 0.2, Bmax = 0.8, Nsteps = 8, By = By,
                 energy = γ * M_PROTON, mass = M_PROTON, charge = 1.0)
    we = Wiggler(1.2; lw = 0.2, Bmax = 0.8 * M_ELECTRON / M_PROTON, Nsteps = 8, By = By,
                 energy = γ * M_ELECTRON, mass = M_ELECTRON, charge = -1.0)
    beti = γ / sqrt(γ^2 - 1)
    @test pass!(wp, r0, beti) ≈ pass!(we, r0, beti) rtol = 1e-13
    # a doubly charged ion at the same γ and field sees twice the kick strength
    w2 = Wiggler(1.2; lw = 0.2, Bmax = 0.8, Nsteps = 8, By = By,
                 energy = γ * M_PROTON, mass = M_PROTON, charge = 2.0)
    w1b = Wiggler(1.2; lw = 0.2, Bmax = 1.6, Nsteps = 8, By = By,
                  energy = γ * M_PROTON, mass = M_PROTON, charge = 1.0)
    @test pass!(w2, r0, beti) == pass!(w1b, r0, beti)
    # electron defaults are unchanged (JuTrack parity above relies on this)
    wdef = Wiggler(1.2; lw = 0.2, Bmax = 0.8, Nsteps = 8)
    @test wdef.mass === M_ELECTRON && wdef.charge === -1.0
end
