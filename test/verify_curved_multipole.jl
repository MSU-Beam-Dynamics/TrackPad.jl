using Test
using TrackPad
using StaticArrays
using LinearAlgebra
import TrackPad: _curved_multipole_field, bndmultipolekick, strthinkick, CURVED_MULTIPOLES

# The multipoles of a bend live in a frame bent with curvature h, where the
# field term of the Hamiltonian is -ψ with ψ = (1+hx)·a_s and Maxwell requires
#   ψ_xx + ψ_yy − (h/(1+hx))·ψ_x = 0.
# `_curved_multipole_field` solves that; `Val(false)` is the straight kick AT,
# JuTrack and PTC (exact=false) apply instead. These tests pin the difference.

const PB = SVector(3.0e-3, 0.42, 1.7, -2.3)     # dipole error, gradient, sext, oct
const PA = SVector(-2.0e-3, 0.11, -0.8, 1.4)    # the same, skew
const PTS = [(1.3e-3, -0.9e-3), (-2.1e-3, 1.6e-3), (4.0e-3, 3.0e-3), (-5e-4, -7e-4)]

curved(x, y, h; pa = PA, pb = PB, mo = 3) =
    _curved_multipole_field(x, y, pa, pb, h, mo, Val(true))
straight(x, y, h; pa = PA, pb = PB, mo = 3) =
    _curved_multipole_field(x, y, pa, pb, h, mo, Val(false))

@testset "Straight limit reproduces the plain multipole kick" begin
    @test CURVED_MULTIPOLES                       # the fixtures below assume the default
    for (x, y) in PTS, mo in 0:3
        c = curved(x, y, 0.0; mo = mo)
        s = straight(x, y, 0.0; mo = mo)
        @test c[1] ≈ s[1] atol=1e-18
        @test c[2] ≈ s[2] atol=1e-18
    end
    # ... and the element-level kick is then exactly `strthinkick`
    r = SVector(1.3e-3, 2.0e-4, -0.9e-3, 1.0e-4, 5e-4, 1e-3)
    @test bndmultipolekick(r, PA, PB, 0.37, 0.0, 3) == strthinkick(r, PA, PB, 0.37, 3)
end

@testset "A pure dipole is untouched by the curvature terms" begin
    z4 = SVector(0.0, 0.0, 0.0, 0.0)
    for (x, y) in PTS, h in (0.0, 0.09, -0.21)
        @test all(iszero, curved(x, y, h; pa = z4, pb = z4))
        @test all(iszero, straight(x, y, h; pa = z4, pb = z4))
    end
end

@testset "The curved-frame field satisfies Maxwell; the straight one does not" begin
    # b_y = −dpx/(1+hx), b_x = +dpy/(1+hx); current-free in the bent frame means
    # (∇×B)_s = ∂b_y/∂x − ∂b_x/∂y = 0.  (∇·B = 0 holds for both models, since
    # both kicks are gradients of a scalar, so it does not discriminate.)
    by(f, x, y, h) = -f(x, y, h)[1] / (1 + h * x)
    bx(f, x, y, h) = +f(x, y, h)[2] / (1 + h * x)
    function curl(f, x, y, h; d = 1e-6)
        (by(f, x + d, y, h) - by(f, x - d, y, h)) / 2d -
        (bx(f, x, y + d, h) - bx(f, x, y - d, h)) / 2d
    end
    for (x, y) in PTS, h in (0.09, -0.21)
        # With dpx = ψ_x and dpy = ψ_y the residual is
        #   −(ψ_xx + ψ_yy)/(1+hx) + h ψ_x/(1+hx)².
        # The curved field kills it by construction. The straight field is
        # harmonic instead (Re/Im of an analytic function), so its residual is
        # exactly h·ψ_x/(1+hx)² — a sharper statement than "it is not zero".
        @test abs(curl(curved, x, y, h)) < 1e-7
        @test curl(straight, x, y, h) ≈
              h * straight(x, y, h)[1] / (1 + h * x)^2 rtol=1e-5
        @test abs(curl(straight, x, y, h)) > 1e-4      # and is numerically real
    end
    # both are Maxwellian when the frame is straight
    for (x, y) in PTS
        @test abs(curl(straight, x, y, 0.0)) < 1e-7
        @test abs(curl(curved, x, y, 0.0)) < 1e-7
    end
end

@testset "Closed form of the curved gradient kick" begin
    # For a pure normal gradient the recursion terminates, with
    #   ψ = −(K₁x²/2 + hK₁x³/3) + (K₁/2)(1+hx)y² + h²K₁y⁴/(24(1+hx)),
    # so the kick is ψ_x, ψ_y exactly.
    K1 = 0.42
    pb = SVector(0.0, K1, 0.0, 0.0); pa = SVector(0.0, 0.0, 0.0, 0.0)
    for (x, y) in PTS, h in (0.09, -0.21)
        oh = 1 + h * x
        dpx = -(K1 * x + h * K1 * x^2) + (h * K1 / 2) * y^2 - h^3 * K1 * y^4 / (24 * oh^2)
        dpy = K1 * oh * y + h^2 * K1 * y^3 / (6 * oh)
        c = curved(x, y, h; pa = pa, pb = pb)
        @test c[1] ≈ dpx rtol=1e-14
        @test c[2] ≈ dpy rtol=1e-14
        # The leading difference from the straight kick is a sextupole-like
        # feed-down of strength k₂ = h·K₁ — exactly that in the vertical plane,
        # and that plus an extra −hK₁x² horizontally. (The remainders are the
        # y⁴ and y³ terms above, relatively O(h y²) and O(h y²/x).)
        s = straight(x, y, h; pa = pa, pb = pb)
        @test c[1] - s[1] ≈ -h * K1 * x^2 + (h * K1 / 2) * y^2 rtol=1e-3
        @test c[2] - s[2] ≈ h * K1 * x * y rtol=1e-3
    end
end

@testset "The kick is the gradient of one scalar (so it is symplectic)" begin
    # dpx = ψ_x and dpy = ψ_y come from a single ψ, so ∂dpx/∂y must equal
    # ∂dpy/∂x exactly. This is what makes the thin kick symplectic, and it is
    # the property a hand-written "(1+hx) × straight kick" would break.
    for (x, y) in PTS, h in (0.0, 0.09, -0.21), f in (curved, straight)
        d = 1e-6
        dxy = (f(x, y + d, h)[1] - f(x, y - d, h)[1]) / 2d
        dyx = (f(x + d, y, h)[2] - f(x - d, y, h)[2]) / 2d
        @test dxy ≈ dyx atol=1e-9
    end
    # the naive scaling really does break it
    naive(x, y, h) = ((1 + h * x) .* straight(x, y, h))
    bad = maximum(PTS) do (x, y)
        d = 1e-6; h = -0.21
        abs((naive(x, y + d, h)[1] - naive(x, y - d, h)[1]) / 2d -
            (naive(x + d, y, h)[2] - naive(x - d, y, h)[2]) / 2d)
    end
    @test bad > 1e-4            # ~5e-4 here, against 1e-9 for both real models
end

@testset "Combined-function bends stay symplectic" begin
    beam = Beam(3.0e9)
    S = zeros(6, 6)
    for i in (1, 3, 5); S[i, i+1] = 1; S[i+1, i] = -1; end
    for elem in (SBend(1.5, 0.13, 0.02, 0.03; polynom_b = [0.0, 0.42, 1.1, 0.0],
                       polynom_a = [0.0, 0.07, 0.0, 0.0], num_int_steps = 12),
                 ExactSBend(1.5, 0.13, 0.02, 0.03; polynom_b = [0.0, 0.42, 1.1, 0.0],
                            polynom_a = [0.0, 0.07, 0.0, 0.0], num_int_steps = 12))
        # the map itself is symplectic by construction; this is the
        # finite-difference Jacobian, so the floor is the FD noise ~ eps/h
        M = transfer_map(Lattice(AbstractElement[elem]), beam)
        @test norm(M' * S * M - S, Inf) < 1e-7
    end
end

# A compact combined-function ring — 24 cells of gradient dipoles, ρ = 11.46 m —
# where the curvature of the gradient is a 220 % effect on ξx. Reference values
# from MAD-X 5.09.03 (`twiss, chrom`) and PTC (`model=1, method=6, nst=20`,
# `ptc_twiss, closed_orbit, icase=5, no=3`). 3 GeV electrons, e1 = e2 = 0.
const CF_ANG = 2π / 48
cf_ring(T) = Lattice(vcat([AbstractElement[
        T(1.5, CF_ANG; polynom_b = [0.0, 0.42, 0.0, 0.0], num_int_steps = 80), Drift(0.6),
        T(1.5, CF_ANG; polynom_b = [0.0, -0.52, 0.0, 0.0], num_int_steps = 80), Drift(0.6),
    ] for _ in 1:24]...); periodic = true)

@testset "Combined-function ring against MAD-X and PTC" begin
    beam = Beam(3.0e9)
    twe = twiss(cf_ring(ExactSBend), beam)
    tws = twiss(cf_ring(SBend), beam)

    # Linear optics are the same in every bend model — they were already right,
    # and are the control for the chromaticity comparison below. 80 integration
    # steps per 1.5 m bend; what is left is this integrator's truncation, which
    # falls to zero against MAD-X as the steps are refined.
    for tw in (twe, tws)
        @test tw.length ≈ 100.8 rtol=1e-12                 # MAD-X length
        @test tw.tunex ≈ 0.082254539 atol=5e-8             # MAD-X q1 = 3.082254539
        @test tw.tuney ≈ 0.429868774 atol=5e-8             # MAD-X q2 = 5.429868774
        @test tw.alphac ≈ 0.1154288029 rtol=1e-7           # MAD-X alfa, PTC alpha_c
        @test tw.betax[1] ≈ 7.009412077 rtol=1e-7          # MAD-X betx
        @test tw.dx[1] ≈ 2.041452399 rtol=1e-7             # MAD-X dx
    end

    # `ExactSBend` — exact curved body and curved-frame multipoles — is the one
    # that corresponds to PTC `exact=true` and to MAD-X. This is the assertion
    # that fails if the (1+hx) on the gradient is ever dropped again: without it
    # ξx is −3.8693 rather than −1.4737.
    @test twe.chromx ≈ -1.4737294 rtol=1e-5                # MAD-X dq1; PTC −1.4737341
    @test twe.chromy ≈ -4.5504344 rtol=1e-5                # MAD-X dq2; PTC −4.5504406

    # `SBend` keeps AT's expanded kinetic term but the same curved-frame field,
    # so it matches no external code exactly: it sits between the fully expanded
    # model (PTC exact=false: −4.7253694, −5.2655585, which is what TrackPad
    # itself produced before the field was corrected) and the exact one. What
    # remains is the h·x·(px²+py²)/2(1+δ) term of the kinetic expansion, which
    # the conventions page documents.
    @test tws.chromx ≈ -2.32978677 rtol=1e-6
    @test tws.chromy ≈ -4.72683114 rtol=1e-6
    @test -4.7253694 < tws.chromx < -1.4737294
    @test -5.2655585 < tws.chromy < -4.5504344
end
