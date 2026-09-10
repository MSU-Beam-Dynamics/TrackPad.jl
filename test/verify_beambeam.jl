using Test
using TrackPad
using StaticArrays
using LinearAlgebra
using PolySeries

# Strong beam-beam kick: Faddeeva function, Bassetti-Erskine field, the two
# elements built on them, their packed-GPU and TPSA paths, and the PALS mapping.
# Everything is checked from first principles (series reference for w(z),
# Maxwell's equations and Gauss's law for the field, direct integration of the
# Coulomb field of the charge distribution); JuTrack's beam-beam code is a
# placeholder and is deliberately not used as a reference.

# ── References ───────────────────────────────────────────────────────────────

# w(z) = exp(-z^2) (1 - erf(-iz)) from the everywhere-convergent power series
# of erf, summed in 512-bit arithmetic. Independent of every approximation in
# src/ and good to ~1e-30 for |z| ≲ 6.
function faddeeva_reference(z::Complex{Float64})
    return setprecision(BigFloat, 512) do
        zb = Complex{BigFloat}(z)
        u = -im * zb
        term = u
        s = u
        n = 0
        while n < 3000
            n += 1
            term *= -u * u / n
            add = term / (2n + 1)
            s += add
            n > 20 && abs(add) < big(1e-60) * abs(s) && break
        end
        erf_u = 2 / sqrt(big(pi)) * s
        Complex{Float64}(exp(-zb * zb) * (1 - erf_u))
    end
end

# Round-beam closed form, written out independently of the implementation.
round_field_reference(x, y, σ) = begin
    r2 = x^2 + y^2
    r2 == 0 && return (0.0, 0.0)
    g = 2 * (1 - exp(-r2 / (2σ^2))) / r2
    (g * x, g * y)
end

const bb_beam = Beam(3.0e9)               # weak beam: 3 GeV electrons
const bb_beti = 1 / bb_beam.beta

@testset "Faddeeva function" begin
    @test faddeeva_w(0.0 + 0.0im) ≈ 1.0 atol=1e-14
    # Abramowitz & Stegun, Table 7.9
    @test faddeeva_w(1.0 + 1.0im) ≈ 0.3047442052 + 0.2082189382im atol=1e-10
    # Real axis: Re w(x) = exp(-x^2). The approximation is accurate in absolute
    # terms (~1e-15 across the plane); at x = 4 the real part is e^{-16} ≈ 1e-7,
    # far below |w| ≈ 0.14, so absolute accuracy is the meaningful criterion —
    # Bassetti-Erskine combines both parts of w at the scale of |w|.
    for x in (0.3, 1.0, 2.5, 4.0)
        @test real(faddeeva_w(complex(x, 0.0))) ≈ exp(-x^2) atol=1e-14
    end
    # Imaginary axis: w(iy) = erfcx(y) is real, and equals the series reference.
    for y in (0.1, 0.5, 1.0, 2.0, 4.0, 8.0)
        @test faddeeva_w(complex(0.0, y)) ≈ faddeeva_reference(complex(0.0, y)) rtol=1e-13
        @test abs(imag(faddeeva_w(complex(0.0, y)))) < 1e-15
    end
    # Whole plane against the series reference, both half-planes.
    worst = 0.0
    for x in -5.0:0.5:5.0, y in -3.0:0.5:5.0
        z = complex(x, y)
        ref = faddeeva_reference(z)
        worst = max(worst, abs(faddeeva_w(z) - ref) / abs(ref))
    end
    @test worst < 1e-12
    # Symmetry w(-conj(z)) = conj(w(z)).
    z = 1.7 + 0.4im
    @test faddeeva_w(-conj(z)) ≈ conj(faddeeva_w(z)) rtol=1e-13
    # Float32 is self-consistent to single precision.
    @test faddeeva_w(ComplexF32(1.0f0, 1.0f0)) ≈ ComplexF32(faddeeva_w(1.0 + 1.0im)) rtol=1e-5
end

@testset "Gaussian beam field" begin
    σx, σy = 2.0e-4, 1.0e-4

    @testset "round beam closed form" begin
        σ = 1.5e-4
        for (x, y) in ((1e-4, 0.0), (0.0, 3e-4), (2e-4, -1e-4), (6e-4, 5e-4))
            @test collect(gaussian_beam_field(x, y, σ, σ)) ≈ collect(round_field_reference(x, y, σ)) rtol=1e-13
        end
        # Linear regime: E ≈ r / σ^2, i.e. focusing strength 1/σ^2.
        @test gaussian_beam_field(1e-9, 0.0, σ, σ)[1] * σ^2 / 1e-9 ≈ 1.0 rtol=1e-8
        @test collect(gaussian_beam_field(0.0, 0.0, σ, σ)) == [0.0, 0.0]
        # Far tail: 1/r Coulomb field of the whole charge, E_r → 2/r.
        @test gaussian_beam_field(40σ, 0.0, σ, σ)[1] ≈ 2 / (40σ) rtol=1e-12
        # Odd in the offset.
        @test gaussian_beam_field(-1e-4, 2e-4, σ, σ)[1] ≈ -gaussian_beam_field(1e-4, 2e-4, σ, σ)[1]
    end

    @testset "elliptical beam is continuous with the round limit" begin
        σ = 1.5e-4
        for ε in (1e-7, 1e-5, 1e-3)
            Ex_r, Ey_r = gaussian_beam_field(1.2e-4, 0.6e-4, σ, σ)
            Ex_e, Ey_e = gaussian_beam_field(1.2e-4, 0.6e-4, σ, σ * (1 - ε))
            @test abs(Ex_e - Ex_r) / abs(Ex_r) < 5ε
            @test abs(Ey_e - Ey_r) / abs(Ey_r) < 5ε
        end
    end

    @testset "symmetries" begin
        Ex, Ey = gaussian_beam_field(1.3e-4, 0.7e-4, σx, σy)
        Ex2, Ey2 = gaussian_beam_field(1.3e-4, -0.7e-4, σx, σy)
        @test Ex2 ≈ Ex && Ey2 ≈ -Ey                      # y → -y
        Ex3, Ey3 = gaussian_beam_field(-1.3e-4, 0.7e-4, σx, σy)
        @test Ex3 ≈ -Ex && Ey3 ≈ Ey                      # x → -x
        Ey4, Ex4 = gaussian_beam_field(0.7e-4, 1.3e-4, σy, σx)
        @test Ex4 ≈ Ex && Ey4 ≈ Ey                       # x ↔ y with σx ↔ σy
    end

    @testset "Maxwell: div E = 4π ρ, curl E = 0 (elliptical, both orderings)" begin
        h = 1e-9
        for (sx, sy) in ((σx, σy), (σy, σx))
            for (x, y) in ((0.0, 0.0), (0.8e-4, 0.3e-4), (-2.1e-4, 1.4e-4), (3.0e-4, -3.5e-4))
                dExdx = (gaussian_beam_field(x + h, y, sx, sy)[1] - gaussian_beam_field(x - h, y, sx, sy)[1]) / 2h
                dEydy = (gaussian_beam_field(x, y + h, sx, sy)[2] - gaussian_beam_field(x, y - h, sx, sy)[2]) / 2h
                dExdy = (gaussian_beam_field(x, y + h, sx, sy)[1] - gaussian_beam_field(x, y - h, sx, sy)[1]) / 2h
                dEydx = (gaussian_beam_field(x + h, y, sx, sy)[2] - gaussian_beam_field(x - h, y, sx, sy)[2]) / 2h
                # In this normalization a round beam has E_r = 2(1 - e^{-u})/r, so
                # div E = 4π ρ with ρ = exp(-x²/2σx² - y²/2σy²) / (2π σx σy).
                ρ = exp(-x^2 / (2sx^2) - y^2 / (2sy^2)) / (2π * sx * sy)
                @test dExdx + dEydy ≈ 4π * ρ rtol=1e-5
                @test abs(dExdy - dEydx) < 1e-5 * (abs(dExdx) + abs(dEydy))
            end
        end
    end

    @testset "Gauss's law in integral form" begin
        # Flux of E through a circle of radius R equals 4π × the charge inside,
        # for an elliptical beam whose field has no closed form.
        R = 2.5e-4
        nθ = 4000
        flux = 0.0
        for k in 0:nθ-1
            θ = 2π * k / nθ
            Ex, Ey = gaussian_beam_field(R * cos(θ), R * sin(θ), σx, σy)
            flux += (Ex * cos(θ) + Ey * sin(θ)) * R * 2π / nθ
        end
        n = 600
        q = 0.0
        d = 2R / n
        for i in 1:n, j in 1:n
            x = -R + (i - 0.5) * d
            y = -R + (j - 0.5) * d
            x^2 + y^2 <= R^2 || continue
            q += exp(-x^2 / (2σx^2) - y^2 / (2σy^2)) / (2π * σx * σy) * d^2
        end
        @test flux ≈ 4π * q rtol=2e-3
    end

    @testset "direct integration of the Coulomb field of the charge distribution" begin
        # E(r) = 2 ∫ ρ(r') (r - r') / |r - r'|² d²r'  (a unit point charge gives
        # E_r = 2/r in this normalization). Midpoint rule on a fine grid with the
        # field point on a cell centre, so the 1/|d| singularity cancels by
        # symmetry between neighbouring cells.
        function integrated_field(x, y, sx, sy; n=1001, span=6.0)
            hx = 2span * sx / n
            hy = 2span * sy / n
            # shift the grid so (x, y) is exactly a cell centre
            i0 = round((x + span * sx) / hx - 0.5)
            j0 = round((y + span * sy) / hy - 0.5)
            x0 = x - (i0 + 0.5) * hx
            y0 = y - (j0 + 0.5) * hy
            Ex = 0.0; Ey = 0.0
            norm_ρ = 1 / (2π * sx * sy)
            for i in 0:n-1
                xp = x0 + (i + 0.5) * hx
                gx = exp(-xp^2 / (2sx^2))
                for j in 0:n-1
                    yp = y0 + (j + 0.5) * hy
                    dx = x - xp; dy = y - yp
                    d2 = dx^2 + dy^2
                    d2 <= 1e-12 * hx * hy && continue   # the cell holding the field point
                    w = 2 * norm_ρ * gx * exp(-yp^2 / (2sy^2)) * hx * hy / d2
                    Ex += w * dx; Ey += w * dy
                end
            end
            return Ex, Ey
        end
        for (sx, sy) in ((σx, σy), (σy, σx), (1.5e-4, 1.5e-4))
            for (x, y) in ((0.7e-4, 0.4e-4), (-1.6e-4, 2.2e-4), (3.0e-4, -0.5e-4))
                Ex_int, Ey_int = integrated_field(x, y, sx, sy)
                Ex, Ey = gaussian_beam_field(x, y, sx, sy)
                @test Ex ≈ Ex_int rtol=2e-3
                @test Ey ≈ Ey_int rtol=2e-3
            end
        end
    end

    @testset "mixed argument types promote" begin
        @test collect(gaussian_beam_field(1e-4, 0.0, 2e-4, 1e-4)) ≈
              collect(gaussian_beam_field(1f-4, 0.0, 2e-4, 1e-4)) rtol=1e-6
    end
end

@testset "beambeam_amplitude and classical_radius" begin
    @test classical_radius(M_ELECTRON, -1.0) ≈ 2.8179403e-15 rtol=1e-6
    @test classical_radius(M_PROTON, 1.0) ≈ 1.5346982e-18 rtol=1e-6
    @test classical_radius(bb_beam) == classical_radius(M_ELECTRON, -1.0)
    N = 1.0e11
    # Electron weak beam colliding with N positrons: opposite charges, focusing.
    a = beambeam_amplitude(bb_beam, N, +1.0)
    @test a ≈ N * classical_radius(bb_beam) * (-1.0) * (+1.0) / bb_beam.gamma
    @test a < 0
    # Magnitude check against the textbook numbers: r_e = 2.818e-15 m, γ = 5871.
    @test abs(a) ≈ N * 2.8179403e-15 / 5871.8 rtol=1e-4
end

@testset "StrongThinGaussianBeam kick" begin
    amp = beambeam_amplitude(bb_beam, 1.0e11, 1.0)
    elem = StrongThinGaussianBeam(amp, 2.0e-4, 1.0e-4; xoffset=1e-5, yoffset=-2e-5)
    r = SVector(1.3e-4, 2.0e-4, -0.7e-4, 1.0e-4, 3.0e-3, 2.0e-3)
    out = pass!(elem, r, bb_beti)
    Ex, Ey = gaussian_beam_field(r[1] - 1e-5, r[3] + 2e-5, 2.0e-4, 1.0e-4)
    @test out[2] - r[2] ≈ amp * Ex
    @test out[4] - r[4] ≈ amp * Ey
    @test out[[1, 3, 5, 6]] == r[[1, 3, 5, 6]]

    # A thin kick derived from a potential is symplectic: J' S J = S.
    S = [0 1 0 0 0 0; -1 0 0 0 0 0; 0 0 0 1 0 0; 0 0 -1 0 0 0; 0 0 0 0 0 1; 0 0 0 0 -1 0]
    J = zeros(6, 6)
    h = 1e-9
    for j in 1:6
        e = SVector{6}(ntuple(k -> k == j ? 1.0 : 0.0, 6))
        J[:, j] = (pass!(elem, r + h * e, bb_beti) - pass!(elem, r - h * e, bb_beti)) / 2h
    end
    @test norm(J' * S * J - S, Inf) < 1e-6

    # Linear beam-beam parameter: for a round beam at the origin the kick is
    # amp·x/σ², so a lattice of (drift, kick) has ξ = β* amp/(4π σ²) in the
    # weak-focusing sense: check the focal length directly.
    σ = 1e-4
    round_elem = StrongThinGaussianBeam(amp, σ, σ)
    x0 = 1e-9
    @test (pass!(round_elem, SVector(x0, 0, 0, 0, 0, 0), bb_beti)[2]) / x0 ≈ amp / σ^2 rtol=1e-8
end

@testset "StrongGaussianBeam synchro-beam mapping" begin
    N = 1.0e11
    strong = (charge=1.0, mass=M_ELECTRON, energy=3.0e9)
    # Coupling from the weak beam.
    sgb1 = StrongGaussianBeam(strong.charge, strong.mass, 1.0, Int(N), strong.energy, (2.0e-4, 1.0e-4);
                              weak_beam=bb_beam, nzslice=1)
    @test sgb1.kick_scale ≈ classical_radius(bb_beam) * bb_beam.charge * strong.charge / bb_beam.gamma
    @test sgb1.zslice_npar == [N]
    @test_throws ArgumentError StrongGaussianBeam(1.0, M_ELECTRON, 1.0, 10, 3.0e9, (1e-4, 1e-4))
    @test_throws ArgumentError StrongGaussianBeam(1.0, M_ELECTRON, 1.0, 10, 3.0e9, (1e-4, 1e-4);
                                                  weak_beam=bb_beam, nzslice=2, zslice_center=[0.0])

    # One slice at the IP is exactly the thin element with amplitude N·kick_scale
    # for a particle at z = 0.
    thin = StrongThinGaussianBeam(N * sgb1.kick_scale, 2.0e-4, 1.0e-4)
    r0 = SVector(1.3e-4, 2.0e-4, -0.7e-4, 1.0e-4, 0.0, 2.0e-3)
    @test pass!(sgb1, r0, bb_beti) ≈ pass!(thin, r0, bb_beti) rtol=1e-14

    # Off the IP the weak particle meets the slice at s* = (z - zc)/2: the kick
    # is evaluated at the drifted position and the momenta are drifted back.
    zc = 4.0e-3
    sgb_z = StrongGaussianBeam(strong.charge, strong.mass, 1.0, Int(N), strong.energy, (2.0e-4, 1.0e-4);
                               weak_beam=bb_beam, nzslice=1, zslice_center=[zc])
    r = SVector(1.3e-4, 2.0e-4, -0.7e-4, 1.0e-4, 1.0e-3, 0.0)
    sstar = (r[5] - zc) / 2
    xc, yc = r[1] + r[2] * sstar, r[3] + r[4] * sstar
    Ex, Ey = gaussian_beam_field(xc, yc, 2.0e-4, 1.0e-4)
    px = r[2] + sgb_z.kick_scale * N * Ex
    py = r[4] + sgb_z.kick_scale * N * Ey
    expected = SVector(xc - px * sstar, px, yc - py * sstar, py, r[5], r[6])
    @test pass!(sgb_z, r, bb_beti) ≈ expected rtol=1e-14
    # ... and z, δ are untouched.
    @test pass!(sgb_z, r, bb_beti)[5:6] == r[5:6]

    # Several slices with offsets: the transverse map stays symplectic.
    sgb3 = StrongGaussianBeam(strong.charge, strong.mass, 1.0, Int(N), strong.energy, (2.0e-4, 1.0e-4);
                              weak_beam=bb_beam, nzslice=3, zslice_center=[-3e-3, 0.0, 3e-3],
                              xoffsets=[1e-5, 0.0, -1e-5], yoffsets=[0.0, 2e-5, 0.0])
    S4 = [0 1 0 0; -1 0 0 0; 0 0 0 1; 0 0 -1 0]
    J = zeros(4, 4)
    h = 1e-9
    for j in 1:4
        e = SVector{6}(ntuple(k -> k == j ? 1.0 : 0.0, 6))
        J[:, j] = (pass!(sgb3, r + h * e, bb_beti) - pass!(sgb3, r - h * e, bb_beti))[1:4] / 2h
    end
    @test norm(J' * S4 * J - S4, Inf) < 1e-6
end

@testset "Packed GPU kernel matches the scalar kick" begin
    amp = beambeam_amplitude(bb_beam, 1.0e11, 1.0)
    for (sx, sy) in ((2.0e-4, 1.0e-4), (1.0e-4, 2.0e-4), (1.5e-4, 1.5e-4))
        elem = StrongThinGaussianBeam(amp, sx, sy; xoffset=1e-5, yoffset=-2e-5)
        lat = Lattice(AbstractElement[Drift(0.3), elem, Quadrupole(0.2, 0.8; num_int_steps=2)])
        coords = [1.3e-4 2.0e-4 -0.7e-4 1.0e-4 3.0e-3 2.0e-3;
                  -2.0e-4 1.0e-4 3.0e-4 -2.0e-4 -1.0e-3 -1.0e-3;
                  0.0 0.0 0.0 0.0 0.0 0.0]
        reference = copy(coords)
        flags = zeros(Int, size(coords, 1))
        linepass!(reference, lat, bb_beam, flags)

        gpu64 = copy(coords)
        batch_linepass!(gpu64, GPULattice(lat, bb_beam; dtype=Float64))
        @test gpu64 ≈ reference rtol=1e-13 atol=1e-18

        gpu32 = Float32.(coords)
        batch_linepass!(gpu32, GPULattice(lat, bb_beam; dtype=Float32))
        @test Float64.(gpu32) ≈ reference rtol=2e-5 atol=1e-9
    end
    # The element is a legitimate sweep target (amplitude, sizes, offsets).
    elem = StrongThinGaussianBeam(amp, 2e-4, 1e-4)
    gl = GPULattice(Lattice(AbstractElement[elem]), bb_beam; dtype=Float64)
    amps = [0.5amp, amp, 2amp]
    sweep = ParamSweepLattice(gl, [(1, 2, amps)])
    r0 = [1.3e-4 0.0 -0.7e-4 0.0 0.0 0.0]
    out = zeros(3, 6)
    param_sweep_linepass!(out, repeat(r0, 3, 1), sweep)
    Ex, Ey = gaussian_beam_field(1.3e-4, -0.7e-4, 2e-4, 1e-4)
    @test out[:, 2] ≈ amps .* Ex
    @test out[:, 4] ≈ amps .* Ey
end

# PolySeries stores inactive degree blocks lazily; read through the mask.
function tp_coefficient(series, index::Int)
    degree = Int(series.desc.polymap.map[index, 1])
    active = (series.degree_mask[] & (UInt64(1) << degree)) != 0
    return active ? series.c[index] : zero(eltype(series.c))
end

@testset "TPSA: round beam has a series form, elliptical is rejected" begin
    amp = 1.0e-8
    σ = 1.0e-4
    lat = Lattice(AbstractElement[Drift(0.2), StrongThinGaussianBeam(amp, σ, σ), Drift(0.2)])
    for co in (zeros(6), [0.6e-4, 0.0, -0.3e-4, 0.0, 0.0, 0.0], [3e-4, 1e-4, 2e-4, 0.0, 0.0, 0.0])
        m = tpsa_map(lat, bb_beam; order=1, closed_orbit=co)
        M_tpsa = [tp_coefficient(m[i], j + 1) for i in 1:6, j in 1:6]
        M_fd = transfer_map(lat, bb_beam; reference=SVector{6}(co))
        @test M_tpsa ≈ M_fd atol=1e-6
    end
    ell = Lattice(AbstractElement[StrongThinGaussianBeam(amp, 2σ, σ)])
    @test_throws ArgumentError tpsa_map(ell, bb_beam; order=1)
end

@testset "PALS BeamBeam builds a physical amplitude" begin
    path, io = mktemp()
    write(io, """
    PALS:
      facility:
        - beginning:
            kind: BeginningEle
            ReferenceP:
              species_ref: electron
              pc_ref: 3.0e9
        - bb:
            kind: BeamBeam
            BeamBeamP:
              sigma_x: 2.0e-4
              sigma_y: 1.0e-4
              N_particle: 1.0e11
              charge: 1.0
        - line:
            kind: BeamLine
            line:
              - beginning
              - bb
    """)
    close(io)
    lat, beam = read_pals(path)
    rm(path; force=true)
    bb = lat.elements[findfirst(e -> e isa StrongThinGaussianBeam, lat.elements)]
    @test bb.rmssizex == 2.0e-4 && bb.rmssizey == 1.0e-4
    @test bb.amplitude ≈ beambeam_amplitude(beam, 1.0e11, 1.0) rtol=1e-12
    @test bb.amplitude < 0                      # e⁻ on e⁺: focusing
    @test abs(bb.amplitude) < 1e-6              # not the raw particle count
end
