"""
    tracking.jl

Core particle tracking functions for TrackPad.jl.
Implements allocation-free symplectic integrators using StaticArrays.

TrackPad canonical coordinate convention:
- r[1]: x  - Horizontal position
- r[2]: px - Canonical horizontal momentum (Px/P0)
- r[3]: y  - Vertical position  
- r[4]: py - Canonical vertical momentum (Py/P0)
- r[5]: z  - Canonical longitudinal coordinate s/β0-c*t = -c*(t-t0)
- r[6]: δE - Relative energy deviation (E-E0)/(P0*c)

The longitudinal canonical pair is (z, δE). See docs/src/conventions.md.
"""

using LinearAlgebra
using StaticArrays

export pass!, linepass!

# =============================================================================
# Physical Constants
# =============================================================================
const COORD_LIMIT = 1.0
const ANGLE_LIMIT = 1.0

# Yoshida 4th-order symplectic integrator coefficients
const DRIFT1 = 0.6756035959798286638
const DRIFT2 = -0.1756035959798286639
const KICK1 = 1.351207191959657328
const KICK2 = -1.702414383919314656

# =============================================================================
# Global Settings
# =============================================================================
# Use exact energy-coordinate Hamiltonian
# Ps/P0 = sqrt(1 + 2δE/β0 + δE² - px² - py²) versus its linearized map.
#
# This is a compile-time constant: `drift6` runs ~40 times per default thick
# element per particle, and a mutable global there (a) reloads on every call,
# (b) cannot be branch-folded, and (c) is treated by Enzyme as possibly-active
# memory, which forces runtime activity analysis. The linearized drift remains
# available for cross-code comparison via `drift6(r, L, beti, Val(false))`.
const USE_EXACT_HAMILTONIAN = true

"""
    SYMPLECTIC_BEND_EDGE

When `true` (the default) the AT-style dipole edge maps carry the longitudinal
term their Hamiltonian implies, making them exactly symplectic. `false`
reproduces AT/JuTrack bit-for-bit, which omits it. See
`edge_fringe_entrance` for the derivation.
"""
const SYMPLECTIC_BEND_EDGE = true

# =============================================================================
# Helper Functions (allocation-free)
# =============================================================================

# ---------------------------------------------------------------------------
# TPSA compatibility guards
#
# These allow CTPS{Float64} (or any non-Real) coordinates to flow through the
# tracking kernels unchanged for the few operations that require ordered-field
# semantics (comparisons, NaN checks, clamp).  For Real coordinates the
# original behaviour is preserved exactly.
# ---------------------------------------------------------------------------

"""Return true when pz² ≤ 0 (particle lost); always false for TPSA types."""
@inline _check_pz2(pz2::Real) = pz2 <= zero(pz2)
@inline _check_pz2(_) = false

"""Return NaN coords for real types; unreachable for TPSA types."""
@inline _nan_coords(::Type{T}) where T<:Real = SVector{6,T}(T(NaN),T(NaN),T(NaN),T(NaN),T(NaN),T(NaN))

"""
    _lost_coords(r, ::Type{T})

NaN coordinates for a lost particle, typed like the *normal* return of the
kernel that lost it: the promotion of the coordinate type `S` and the element
parameter type `T`. Returning `SVector{6,T}` (element type) here while the
regular path returns `SVector{6,promote_type(T,S)}` makes the kernel's return
type a `Union` whenever `S != T` (Dual coordinates, Dual element parameters,
Float32 coordinates, ...), and every integrator loop that reassigns `r` then
boxes on each step. For non-Real coordinates (TPSA) the loss branches are
unreachable, so `r` itself is returned to keep the inferred type stable.
"""
@inline _lost_coords(::SVector{6,S}, ::Type{T}) where {S<:Real,T<:Real} =
    _nan_coords(promote_type(S, T))
@inline _lost_coords(r::SVector{6}, ::Type) = r

"""Guard for `abs(val) < tol` tests (comparisons undefined on CTPS)."""
@inline _check_tiny(val::Real, tol::Real) = abs(val) < tol
@inline _check_tiny(_, _) = false

"""Guard for clamp in exact-bend helpers (comparisons undefined on CTPS)."""
@inline _safe_clamp(val::Real, lo::Real, hi::Real) = clamp(val, lo, hi)
@inline _safe_clamp(val, lo, hi) = val

"""Guard for pxyz sqrt domain check (only meaningful for Real coords)."""
@inline _safe_sqrt_pz(val::Real) = val > zero(val) ? sqrt(val) : zero(val)
@inline _safe_sqrt_pz(val) = sqrt(val)

"""Return `P/P0` for the stored energy deviation `delta_E`."""
@inline function _momentum_norm(delta_e, beti)
    return _safe_sqrt_pz(one(beti) + 2 * delta_e * beti + delta_e^2)
end

"""Return `d(delta_P)/d(delta_E)` for the exact energy-momentum conversion."""
@inline function _momentum_jacobian(delta_e, beti)
    return (beti + delta_e) / _momentum_norm(delta_e, beti)
end

"""
Return the reference momentum-energy product `P0*c = β0 E0` from the total
energy carried by an RF-type element and the beam's `beti = 1/β0`.
"""
@inline _reference_p0c(total_energy::T, beti::T) where T = abs(total_energy) / beti

"""
    check_lost(r::AbstractVector) -> Bool

Check whether a particle exceeds global transverse safety limits or contains a
NaN transverse coordinate. For TPSA/AD coordinate types, return `false` because
ordered comparisons are not generally defined.
"""
@inline function check_lost(r::AbstractVector{T}) where T<:Real
    return isnan(r[1]) || abs(r[1]) > COORD_LIMIT || abs(r[3]) > COORD_LIMIT ||
           abs(r[2]) > ANGLE_LIMIT || abs(r[4]) > ANGLE_LIMIT
end
@inline check_lost(r::AbstractVector) = false   # CTPS, Dual, etc.

"""
    apply_misalignment(r, t[, R])

Apply misalignment: translation `t` (element type) and optional rotation `R`
to coordinates `r`.  Supports mixed types: element params T, coordinates S.
"""
@inline function apply_misalignment(r::SVector{6,S}, t::SVector{6,T},
                                     R::SMatrix{6,6,T,36}) where {T,S}
    r_new = SVector(r[1]+t[1], r[2]+t[2], r[3]+t[3], r[4]+t[4], r[5]+t[5], r[6]+t[6])
    return R * r_new
end

@inline function apply_misalignment(r::SVector{6,S}, t::SVector{6,T}) where {T,S}
    return SVector(r[1]+t[1], r[2]+t[2], r[3]+t[3], r[4]+t[4], r[5]+t[5], r[6]+t[6])
end

# =============================================================================
# Drift Tracking
# =============================================================================

"""
    drift6(r, L, beti=1.0[, ::Val{exact}]) -> SVector{6}

Track particle through a drift space of length L.
Uses the exact Hamiltonian when `exact` is true (default:
`USE_EXACT_HAMILTONIAN`); `Val(false)` selects the linearized `pz ≈ 1 + δ` map
used for comparisons with codes that approximate the drift.

# Arguments
- `r::SVector{6,S}`: 6D phase space coordinates
- `L::T`: Drift length
- `beti::T`: 1/β (inverse relativistic velocity), default 1.0

# Returns
- Updated coordinates as `SVector{6,promote_type(S,T)}`
"""
@inline drift6(r::SVector{6,S}, L::T, beti::T=one(T)) where {T<:Real,S} =
    drift6(r, L, beti, Val(USE_EXACT_HAMILTONIAN))

"""
    _drift_dz(L, px, py, δE, beti, pz)

Change of `z = s/β0 − ct` over an exact drift of length `L`,
`L[(1/β0 + δE)/pz − 1/β0]`, evaluated without subtracting two O(L) numbers:
the numerator is expanded through its conjugate into
`(1 − β0⁻²) δE (2/β0 + δE) + β0⁻² (px² + py²)`, all terms of which are already
small. The direct form loses ~ε·L per drift, which random-walks to 1e-14 over a
ring and is the floor of finite-difference `M56` (momentum compaction) at the
1e-6 level.
"""
@inline function _drift_dz(L::T, px, py, δE, beti::T, pz) where T
    num = (one(T) - beti * beti) * δE * (2 * beti + δE) + beti * beti * (px * px + py * py)
    return L * num / (pz * ((beti + δE) + beti * pz))
end

@inline function drift6(r::SVector{6,S}, L::T, beti::T, ::Val{true}) where {T<:Real,S}
    # Exact longitudinal momentum for δE = (E-E0)/(P0*c).
    pz2 = one(T) + 2*r[6]*beti + r[6]^2 - r[2]^2 - r[4]^2
    if _check_pz2(pz2)
        return _lost_coords(r, T)
    end
    pz = sqrt(pz2)
    NormL = L / pz
    x_new = r[1] + NormL * r[2]
    y_new = r[3] + NormL * r[4]
    z_new = r[5] - _drift_dz(L, r[2], r[4], r[6], beti, pz)
    return SVector(x_new, r[2], y_new, r[4], z_new, r[6])
end

@inline function drift6(r::SVector{6,S}, L::T, beti::T, ::Val{false}) where {T<:Real,S}
    # Linearized approximation: pz ≈ 1 + δ
    NormL = L / (one(T) + r[6])
    x_new = r[1] + NormL * r[2]
    y_new = r[3] + NormL * r[4]
    z_new = r[5] - NormL * (r[2]^2 + r[4]^2) / (2*(one(T) + r[6]))
    return SVector(x_new, r[2], y_new, r[4], z_new, r[6])
end

"""
    pass!(elem::Drift, r::SVector{6,T}, beti=1.0) -> SVector{6,T}

Track particle through a Drift element.
"""
function pass!(elem::Drift{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    # Apply entrance misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    # Drift
    r = drift6(r, elem.L, beti)
    
    # Apply exit misalignment
    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    r = apply_misalignment(r, elem.t2)
    
    return r
end

"""
    pass!(elem::Marker, r::SVector{6,T}, beti=1.0) -> SVector{6,T}

Track particle through a Marker (no effect).
"""
# `beti` is unused (a Marker is the identity), but the default must not be
# derived from the coordinate type: `one(S)` needs a type-level `one`, which
# CTPS cannot provide because a polynomial needs a descriptor that the type
# does not carry. A plain `Real` default keeps the two-argument form usable
# for TPSA and AD coordinates alike.
function pass!(elem::Marker, r::SVector{6,S}, beti::Real=1.0) where S
    return r
end

# =============================================================================
# Dipole Edge Focusing (Fringe Fields)
# =============================================================================

# Edge-focusing strengths and the derivatives the symplectic correction needs.
# `sgn` is +1 at the entrance and -1 at the exit (the sign of the `px` term in
# the THOMX model). Returns `(fy, dfy_ddelta, dfy_dpx)`.
# True when `fy` reduces to the momentum-independent `fx`: the Brown model with
# no fringe-field integral. The edge is then already symplectic and the
# correction terms are identically zero, so the whole derivative path (and the
# `tan` on possibly-TPSA coordinates) is skipped.
@inline _edge_is_simple(fringecorr, method::Int) =
    iszero(fringecorr) && method != 2 && method != 3

@inline function _edge_fy(r::SVector{6,S}, inv_rho::T, edge_angle::T,
                          fringecorr::T, fx::T, method::Int, sgn::T) where {T<:Real,S}
    d = one(T) + r[6]
    if method == 2
        u = edge_angle - fringecorr / d
        t = tan(u)
        sec2 = t * t + one(T)
        fy = inv_rho * t / d
        return fy, inv_rho * (sec2 * fringecorr / d^3 - t / d^2), zero(T)
    elseif method == 3
        u = edge_angle - fringecorr + sgn * r[2] / d
        t = tan(u)
        sec2 = t * t + one(T)
        fy = inv_rho * t
        return fy, inv_rho * sec2 * (-sgn * r[2] / d^2), inv_rho * sec2 * (sgn / d)
    else  # method 1 (Brown) and the fallback
        u = edge_angle - fringecorr / d
        t = tan(u)
        sec2 = t * t + one(T)
        fy = inv_rho * t
        return fy, inv_rho * sec2 * fringecorr / d^2, zero(T)
    end
end

# The edge is the thin-lens flow of H = -fx*x^2/2 + fy(px, delta)*y^2/2, so a
# delta- or px-dependent `fy` also moves z and x. AT omits both, which is the
# entire source of its edge non-symplecticity.
@inline function _edge_apply(r::SVector{6,S}, fx::T, fy, dfy_dd, dfy_dpx,
                             ::Val{true}) where {T<:Real,S}
    half_y2 = r[3] * r[3] * T(0.5)
    return SVector(r[1] + half_y2 * dfy_dpx, r[2] + r[1] * fx, r[3],
                   r[4] - r[3] * fy, r[5] + half_y2 * dfy_dd, r[6])
end

@inline function _edge_apply(r::SVector{6,S}, fx::T, fy, _dfy_dd, _dfy_dpx,
                             ::Val{false}) where {T<:Real,S}
    return SVector(r[1], r[2] + r[1] * fx, r[3], r[4] - r[3] * fy, r[5], r[6])
end

@inline function _edge_fringecorr(inv_rho::T, edge_angle::T, fint::T, gap::T,
                                  method::Int) where T
    (iszero(fint) || iszero(gap) || method == 0) && return zero(T)
    sedge = sin(edge_angle)
    cedge = cos(edge_angle)
    return inv_rho * gap * fint * (one(T) + sedge^2) / cedge
end

"""
    edge_fringe_entrance(r, inv_rho, edge_angle, fint, gap, method) -> SVector{6,T}
    edge_fringe_entrance(r, inv_rho, edge_angle, fint, gap, method, Val(symplectic))

Apply dipole edge focusing at the entrance face.

# Arguments
- `r`: 6D coordinates
- `inv_rho`: Inverse bending radius (1/ρ)
- `edge_angle`: Pole face rotation angle (e1)
- `fint`: Fringe field integral
- `gap`: Magnet gap
- `method`: Fringe calculation method (0=none, 1=Brown, 2=SOLEIL, 3=THOMX)

# Symplecticity

The edge is a thin kick `px += fx*x`, `py -= fy*y`, so it is the flow of

    H = -fx*x^2/2 + fy*y^2/2,

with `dz/ds = ∂H/∂δ` and `dx/ds = ∂H/∂px`. Whenever `fy` depends on the
momentum — the `1/(1+δ)` of the Brown and SOLEIL models, the `px/(1+δ)` of
THOMX — the map must therefore also move `z` (and, for THOMX, `x`):

    z += y^2/2 * ∂fy/∂δ,      x += y^2/2 * ∂fy/∂px.

AT and JuTrack omit both terms, which is the sole source of their edge
non-symplecticity; it vanishes when `fint*gap == 0` under method 1, where `fy`
reduces to the momentum-independent `fx`. TrackPad includes them by default
(`SYMPLECTIC_BEND_EDGE`). The correction is `O(y²)`, so it leaves the linear
optics about `y = 0` — tunes, Twiss, chromaticity, dispersion — bit-for-bit
unchanged, and only affects off-axis and long-term tracking. Pass
`Val(false)` as a seventh argument for the AT-compatible map.
"""
@inline function edge_fringe_entrance(r::SVector{6,S}, inv_rho::T, edge_angle::T,
                                       fint::T, gap::T, method::Int) where {T<:Real,S}
    return edge_fringe_entrance(r, inv_rho, edge_angle, fint, gap, method,
                                Val(SYMPLECTIC_BEND_EDGE))
end

@inline function edge_fringe_entrance(r::SVector{6,S}, inv_rho::T, edge_angle::T,
                                       fint::T, gap::T, method::Int,
                                       sympl::Val) where {T<:Real,S}
    fringecorr = _edge_fringecorr(inv_rho, edge_angle, fint, gap, method)
    fx = inv_rho * tan(edge_angle)
    if _edge_is_simple(fringecorr, method)
        return SVector(r[1], r[2] + r[1] * fx, r[3], r[4] - r[3] * fx, r[5], r[6])
    end
    fy, dfy_dd, dfy_dpx = _edge_fy(r, inv_rho, edge_angle, fringecorr, fx, method, one(T))
    return _edge_apply(r, fx, fy, dfy_dd, dfy_dpx, sympl)
end

"""
    edge_fringe_exit(r, inv_rho, edge_angle, fint, gap, method) -> SVector{6,T}
    edge_fringe_exit(r, inv_rho, edge_angle, fint, gap, method, Val(symplectic))

Apply dipole edge focusing at the exit face. Identical to
[`edge_fringe_entrance`](@ref) except for the sign of the THOMX `px` term.
"""
@inline function edge_fringe_exit(r::SVector{6,S}, inv_rho::T, edge_angle::T,
                                   fint::T, gap::T, method::Int) where {T<:Real,S}
    return edge_fringe_exit(r, inv_rho, edge_angle, fint, gap, method,
                            Val(SYMPLECTIC_BEND_EDGE))
end

@inline function edge_fringe_exit(r::SVector{6,S}, inv_rho::T, edge_angle::T,
                                   fint::T, gap::T, method::Int,
                                   sympl::Val) where {T<:Real,S}
    fringecorr = _edge_fringecorr(inv_rho, edge_angle, fint, gap, method)
    fx = inv_rho * tan(edge_angle)
    if _edge_is_simple(fringecorr, method)
        return SVector(r[1], r[2] + r[1] * fx, r[3], r[4] - r[3] * fx, r[5], r[6])
    end
    fy, dfy_dd, dfy_dpx = _edge_fy(r, inv_rho, edge_angle, fringecorr, fx, method, -one(T))
    return _edge_apply(r, fx, fy, dfy_dd, dfy_dpx, sympl)
end

# =============================================================================
# Multipole Kick
# =============================================================================

"""
    bndthinkick(r, polynom_a, polynom_b, L, irho, max_order, beti) -> SVector{6,T}

Apply thin bend kick to particle including curvature (irho) term.
This is the bend-specific version of strthinkick that adds the dipole
focusing effect from the curved trajectory.

# Arguments
- `r`: 6D coordinates
- `polynom_a`: Skew multipole coefficients (SVector)
- `polynom_b`: Normal multipole coefficients (SVector)
- `L`: Effective kick length
- `irho`: Inverse bending radius (1/ρ = angle/length)
- `max_order`: Maximum multipole order to apply
- `beti`: 1/β (inverse relativistic velocity)
"""
@inline function bndthinkick(r::SVector{6,S},
                             polynom_a::SVector{N,T},
                             polynom_b::SVector{N,T},
                             L::T,
                             irho::T,
                             max_order::Int,
                             beti::T) where {T<:Real,N,S}
    # Start from highest order
    ReSum = polynom_b[max_order + 1]
    ImSum = polynom_a[max_order + 1]
    
    # Horner's method for polynomial evaluation
    @inbounds for i in max_order:-1:1
        ReSumTemp = ReSum * r[1] - ImSum * r[3] + polynom_b[i]
        ImSum = ImSum * r[1] + ReSum * r[3] + polynom_a[i]
        ReSum = ReSumTemp
    end
    
    # Apply kicks to momenta - note the irho term for the bend
    #   px -= L * (ReSum - (δP - x*irho) * irho)
    #   py += L * ImSum
    #   z  -= L * irho * x * dδP/dδE        for z = s/β0 - c*t.
    # δP = (P-P0)/P0 is taken from the stored δE through the exact relation
    # (1+δP)² = 1 + 2δE/β0 + δE², in the cancellation-free form
    # δP = δE(2/β0 + δE)/(2+δP); the linearised δE/β0 that AT and JuTrack use
    # is exact only for β0 = 1 and made the off-momentum optics of a finite-β0
    # ring depend on the species at O(δ²).
    pnorm = _momentum_norm(r[6], beti)
    dp = r[6] * (2 * beti + r[6]) / (one(T) + pnorm)
    px_new = r[2] - L * (ReSum - (dp - r[1] * irho) * irho)
    py_new = r[4] + L * ImSum
    z_new = r[5] - L * irho * r[1] * (beti + r[6]) / pnorm
    
    return SVector(r[1], px_new, r[3], py_new, z_new, r[6])
end

"""
    strthinkick(r, polynom_a, polynom_b, L, max_order) -> SVector{6,T}

Apply thin multipole kick to particle.

# Arguments
- `r`: 6D coordinates
- `polynom_a`: Skew multipole coefficients (SVector)
- `polynom_b`: Normal multipole coefficients (SVector)
- `L`: Effective kick length
- `max_order`: Maximum multipole order to apply
"""
@inline function strthinkick(r::SVector{6,S},
                             polynom_a::SVector{N,T},
                             polynom_b::SVector{N,T},
                             L::T,
                             max_order::Int) where {T<:Real,N,S}
    # Start from highest order
    ReSum = polynom_b[max_order + 1]
    ImSum = polynom_a[max_order + 1]
    
    # Horner's method for polynomial evaluation
    @inbounds for i in max_order:-1:1
        ReSumTemp = ReSum * r[1] - ImSum * r[3] + polynom_b[i]
        ImSum = ImSum * r[1] + ReSum * r[3] + polynom_a[i]
        ReSum = ReSumTemp
    end
    
    # Apply kicks to momenta
    px_new = r[2] - L * ReSum
    py_new = r[4] + L * ImSum
    
    return SVector(r[1], px_new, r[3], py_new, r[5], r[6])
end

# =============================================================================
# 4th-Order Symplectic Integrator
# =============================================================================

"""
    symplectic4_pass(r, L, polynom_a, polynom_b, max_order, num_steps, beti) -> SVector{6,T}

4th-order Yoshida symplectic integrator for thick multipole elements.
Drift-Kick-Drift-Kick-Drift-Kick-Drift pattern per step.
"""
@inline function symplectic4_pass(r::SVector{6,S},
                                   L::T,
                                   polynom_a::SVector{N,T},
                                   polynom_b::SVector{N,T},
                                   max_order::Int,
                                   num_steps::Int,
                                   beti::T) where {T<:Real,N,S}
    SL = L / num_steps
    L1 = SL * T(DRIFT1)
    L2 = SL * T(DRIFT2)
    K1 = SL * T(KICK1)
    K2 = SL * T(KICK2)
    
    @inbounds for _ in 1:num_steps
        r = drift6(r, L1, beti)
        r = strthinkick(r, polynom_a, polynom_b, K1, max_order)
        r = drift6(r, L2, beti)
        r = strthinkick(r, polynom_a, polynom_b, K2, max_order)
        r = drift6(r, L2, beti)
        r = strthinkick(r, polynom_a, polynom_b, K1, max_order)
        r = drift6(r, L1, beti)
    end
    
    return r
end

"""
    symplectic4_bend_pass(r, L, polynom_a, polynom_b, irho, max_order, num_steps, beti) -> SVector{6,T}

4th-order Yoshida symplectic integrator for bend elements.
Uses bndthinkick instead of strthinkick to include curvature effects.
Drift-Kick-Drift-Kick-Drift-Kick-Drift pattern per step.
"""
@inline function symplectic4_bend_pass(r::SVector{6,S},
                                        L::T,
                                        polynom_a::SVector{N,T},
                                        polynom_b::SVector{N,T},
                                        irho::T,
                                        max_order::Int,
                                        num_steps::Int,
                                        beti::T) where {T<:Real,N,S}
    SL = L / num_steps
    L1 = SL * T(DRIFT1)
    L2 = SL * T(DRIFT2)
    K1 = SL * T(KICK1)
    K2 = SL * T(KICK2)
    
    @inbounds for _ in 1:num_steps
        r = drift6(r, L1, beti)
        r = bndthinkick(r, polynom_a, polynom_b, K1, irho, max_order, beti)
        r = drift6(r, L2, beti)
        r = bndthinkick(r, polynom_a, polynom_b, K2, irho, max_order, beti)
        r = drift6(r, L2, beti)
        r = bndthinkick(r, polynom_a, polynom_b, K1, irho, max_order, beti)
        r = drift6(r, L1, beti)
    end
    
    return r
end

"""
    multipole_fringe(r, polynom_a, polynom_b, max_order, edge, skip_b0, beti) -> SVector{6}

Forest (13.29) entrance/exit fringe correction for thick multipoles, ported
from JuTrack's `multipole_fringe!`. `edge = ±1` selects entrance/exit and
`skip_b0 != 0` omits the dipole term (bends treat it separately). The
longitudinal update carries the opposite sign of JuTrack's because TrackPad's
`z` axis is negated; transverse dynamics are identical.
"""
@inline function multipole_fringe(r::SVector{6,S},
                                  polynom_a::SVector{4,T},
                                  polynom_b::SVector{4,T},
                                  max_order::Int,
                                  edge::T,
                                  skip_b0::Int,
                                  beti::T) where {T<:Real,S}
    # Accumulators are typed like the coordinates (not like the element
    # parameters) so the loop below is type-stable when S != T. The multiply by
    # zero is the only generic way to obtain a typed zero for TPSA coordinates.
    z0 = zero(T) * r[1]
    FX = z0; FY = z0
    FX_X = z0; FX_Y = z0; FY_X = z0; FY_Y = z0
    RX = one(T); IX = zero(T)
    @inbounds for n in 0:max_order
        B = polynom_b[n + 1]
        A = polynom_a[n + 1]
        j = T(n + 1)
        DRX = RX
        DIX = IX
        RX = DRX * r[1] - DIX * r[3]
        IX = DRX * r[3] + DIX * r[1]

        U = zero(T); V = zero(T); DU = zero(T); DV = zero(T)
        if n == 0 && skip_b0 != 0
            U -= A * IX
            V += A * RX
            DU -= A * DIX
            DV += A * DRX
        else
            U += B * RX - A * IX
            V += B * IX + A * RX
            DU += B * DRX - A * DIX
            DV += B * DIX + A * DRX
        end

        f1 = -edge / (T(4) * (j + one(T)))
        U *= f1
        V *= f1
        DU *= f1
        DV *= f1

        DUX = j * DU
        DVX = j * DV
        DUY = -j * DV
        DVY = j * DU

        nf = (j + T(2)) / j

        FX += U * r[1] + nf * V * r[3]
        FY += U * r[3] - nf * V * r[1]

        FX_X += DUX * r[1] + U + nf * r[3] * DVX
        FX_Y += DUY * r[1] + nf * V + nf * r[3] * DVY
        FY_X += DUX * r[3] - nf * V - nf * r[1] * DVX
        FY_Y += DUY * r[3] + U - nf * r[1] * DVY
    end

    DEL = one(T) / (beti + r[6])
    MA = one(T) - FX_X * DEL
    MB = -FY_X * DEL
    MD = one(T) - FY_Y * DEL
    MC = -FX_Y * DEL

    x_new = r[1] - FX * DEL
    y_new = r[3] - FY * DEL
    pxf = (MD * r[2] - MB * r[4]) / (MA * MD - MB * MC)
    pyf = (MA * r[4] - MC * r[2]) / (MA * MD - MB * MC)
    z_new = r[5] + (pxf * FX + pyf * FY) * DEL * DEL
    return SVector(x_new, pxf, y_new, pyf, z_new, r[6])
end

# =============================================================================
# Quadrupole Tracking
# =============================================================================

"""
    pass!(elem::Quadrupole, r::SVector{6,T}, beti=1.0) -> SVector{6,T}

Track particle through a Quadrupole using 4th-order symplectic integrator.
"""
function pass!(elem::Quadrupole{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    # Apply entrance misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    # Prepare polynomial coefficients (k1 goes to polynom_b[2])
    # polynom_b[1] = dipole (k0), polynom_b[2] = quadrupole (k1), etc.
    polynom_b = SVector{4,T}(elem.polynom_b[1], elem.k1, elem.polynom_b[3], elem.polynom_b[4])
    
    # Apply kick angle correction if L > 0
    if elem.L > zero(T)
        kick_correction_b = SVector{4,T}(
            polynom_b[1] - sin(elem.kick_angle[1]) / elem.L,
            polynom_b[2],
            polynom_b[3],
            polynom_b[4]
        )
        kick_correction_a = SVector{4,T}(
            elem.polynom_a[1] + sin(elem.kick_angle[2]) / elem.L,
            elem.polynom_a[2],
            elem.polynom_a[3],
            elem.polynom_a[4]
        )
    else
        kick_correction_b = polynom_b
        kick_correction_a = elem.polynom_a
    end

    # Forest (13.29) entrance/exit fringe when enabled
    if !iszero(elem.fringe_entrance)
        r = multipole_fringe(r, kick_correction_a, kick_correction_b,
                             elem.max_order, one(T), 1, beti)
    end

    # Symplectic integration
    r = symplectic4_pass(r, elem.L, kick_correction_a, kick_correction_b,
                         elem.max_order, elem.num_int_steps, beti)

    if !iszero(elem.fringe_exit)
        r = multipole_fringe(r, kick_correction_a, kick_correction_b,
                             elem.max_order, -one(T), 1, beti)
    end

    # Apply exit misalignment
    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    r = apply_misalignment(r, elem.t2)

    return r
end

# =============================================================================
# Sextupole Tracking
# =============================================================================

"""
    pass!(elem::Sextupole, r::SVector{6,T}, beti=1.0) -> SVector{6,T}

Track particle through a Sextupole using 4th-order symplectic integrator.
"""
function pass!(elem::Sextupole{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    # Apply entrance misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    # k2 goes to polynom_b[3] (with factor 1/2 for standard normalization)
    polynom_b = SVector{4,T}(elem.polynom_b[1], elem.polynom_b[2], elem.k2/2, elem.polynom_b[4])
    polynom_a = elem.polynom_a
    if elem.L > zero(T)
        polynom_b = SVector{4,T}(
            polynom_b[1] - sin(elem.kick_angle[1]) / elem.L,
            polynom_b[2], polynom_b[3], polynom_b[4])
        polynom_a = SVector{4,T}(
            polynom_a[1] + sin(elem.kick_angle[2]) / elem.L,
            polynom_a[2], polynom_a[3], polynom_a[4])
    end

    if !iszero(elem.fringe_entrance)
        r = multipole_fringe(r, polynom_a, polynom_b, elem.max_order, one(T), 1, beti)
    end

    # Symplectic integration
    r = symplectic4_pass(r, elem.L, polynom_a, polynom_b,
                         elem.max_order, elem.num_int_steps, beti)

    if !iszero(elem.fringe_exit)
        r = multipole_fringe(r, polynom_a, polynom_b, elem.max_order, -one(T), 1, beti)
    end
    
    # Apply exit misalignment
    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    r = apply_misalignment(r, elem.t2)
    
    return r
end

# =============================================================================
# Octupole Tracking
# =============================================================================

"""
    pass!(elem::Octupole, r::SVector{6,T}, beti=1.0) -> SVector{6,T}

Track particle through an Octupole using 4th-order symplectic integrator.
"""
function pass!(elem::Octupole{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    # Apply entrance misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    # k3 goes to polynom_b[4] (with factor 1/6 for standard normalization)
    polynom_b = SVector{4,T}(elem.polynom_b[1], elem.polynom_b[2], elem.polynom_b[3], elem.k3/6)
    polynom_a = elem.polynom_a
    if elem.L > zero(T)
        polynom_b = SVector{4,T}(
            polynom_b[1] - sin(elem.kick_angle[1]) / elem.L,
            polynom_b[2], polynom_b[3], polynom_b[4])
        polynom_a = SVector{4,T}(
            polynom_a[1] + sin(elem.kick_angle[2]) / elem.L,
            polynom_a[2], polynom_a[3], polynom_a[4])
    end

    if !iszero(elem.fringe_entrance)
        r = multipole_fringe(r, polynom_a, polynom_b, elem.max_order, one(T), 1, beti)
    end

    # Symplectic integration
    r = symplectic4_pass(r, elem.L, polynom_a, polynom_b,
                         elem.max_order, elem.num_int_steps, beti)

    if !iszero(elem.fringe_exit)
        r = multipole_fringe(r, polynom_a, polynom_b, elem.max_order, -one(T), 1, beti)
    end
    
    # Apply exit misalignment
    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    r = apply_misalignment(r, elem.t2)
    
    return r
end

# =============================================================================
# RF Cavity Tracking
# =============================================================================

"""
    pass!(elem::RFCavity, r::SVector{6,T}, beti=1.0) -> SVector{6,T}

Track particle through an RF Cavity using drift-kick-drift.
The energy kick is normalized by the reference `P0*c` reconstructed from
`elem.energy` (total energy) and `beti`.
"""
function pass!(elem::RFCavity{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    if elem.L > zero(T)
        r = drift6(r, elem.L / 2, beti)
    end
    if elem.energy > zero(T)
        kick = elem.charge * elem.volt / _reference_p0c(elem.energy, beti)
        phase = -T(2pi) * elem.freq * ((r[5] + elem.lag) / T(C_LIGHT)) - elem.philag
        delta_new = r[6] - kick * sin(phase)
        r = SVector(r[1], r[2], r[3], r[4], r[5], delta_new)
    end
    if elem.L > zero(T)
        r = drift6(r, elem.L / 2, beti)
    end
    return r
end

# =============================================================================
# Solenoid Tracking
# =============================================================================

"""
    pass!(elem::Solenoid, r::SVector{6,T}, beti=1.0) -> SVector{6,T}

Track a particle through a Solenoid using the linear hard-edge matrix with the
exact conversion from `δE` to normalized momentum.
"""
function pass!(elem::Solenoid{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    # Apply entrance misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    ks = elem.ks
    L = elem.L

    if iszero(ks)
        r = drift6(r, L, beti)
    else
        momentum = _momentum_norm(r[6], beti)
        p_norm = inv(momentum)
        momentum_jacobian = (beti + r[6]) * p_norm
        x = r[1]
        xpr = r[2] * p_norm
        y = r[3]
        ypr = r[4] * p_norm
        H = ks * p_norm / 2
        Sn = sin(L * H)
        Cn = cos(L * H)

        x_new = x * Cn * Cn + xpr * Cn * Sn / H + y * Cn * Sn + ypr * Sn * Sn / H
        px_new = (-x * H * Cn * Sn + xpr * Cn * Cn - y * H * Sn * Sn + ypr * Cn * Sn) / p_norm
        y_new = -x * Cn * Sn - xpr * Sn * Sn / H + y * Cn * Cn + ypr * Cn * Sn / H
        py_new = (x * H * Sn * Sn - xpr * Cn * Sn - y * Cn * Sn * H + ypr * Cn * Cn) / p_norm
        z_new = r[5] - momentum_jacobian * L *
                (H * H * (x * x + y * y) +
                 2 * H * (xpr * y - ypr * x) + xpr * xpr + ypr * ypr) / 2

        r = SVector(x_new, px_new, y_new, py_new, z_new, r[6])
    end
    
    # Apply exit misalignment
    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    r = apply_misalignment(r, elem.t2)
    
    return r
end

# =============================================================================
# Corrector Tracking
# =============================================================================

"""
    pass!(elem::Corrector, r::SVector{6,T}, beti=1.0) -> SVector{6,T}

Track particle through a Corrector (orbit correction kicks).
"""
function pass!(elem::Corrector{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    # Apply entrance misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    momentum = _momentum_norm(r[6], beti)
    p_norm = inv(momentum)
    momentum_jacobian = (beti + r[6]) * p_norm
    NormL = elem.L * p_norm
    z_new = r[5] - momentum_jacobian * NormL * p_norm *
            (elem.xkick^2 / 3 + elem.ykick^2 / 3 +
             r[2]^2 + r[4]^2 + r[2] * elem.xkick + r[4] * elem.ykick) / 2
    x_new = r[1] + NormL * (r[2] + elem.xkick / 2)
    px_new = r[2] + elem.xkick
    y_new = r[3] + NormL * (r[4] + elem.ykick / 2)
    py_new = r[4] + elem.ykick
    r = SVector(x_new, px_new, y_new, py_new, z_new, r[6])
    
    # Apply exit misalignment
    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    r = apply_misalignment(r, elem.t2)
    
    return r
end

# =============================================================================
# Thin Multipole Tracking
# =============================================================================

"""
    pass!(elem::ThinMultipole, r::SVector{6,T}, beti=1.0) -> SVector{6,T}

Track particle through a thin multipole element.
"""
function pass!(elem::ThinMultipole{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    # Apply entrance misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    # JuTrack thinMULTIPOLE uses an integrated-strength kick with unit kick length.
    # NOTE: JuTrack's Float64 thin-multipole pass applies no fringe field even
    # though the struct carries FringeQuad flags; TrackPad mirrors that here.
    r = strthinkick(r, elem.polynom_a, elem.polynom_b, one(T), elem.max_order)
    
    # Apply exit misalignment
    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    r = apply_misalignment(r, elem.t2)
    
    return r
end

# =============================================================================
# SBend (Sector Bend) Tracking
# =============================================================================

"""
    pass!(elem::SBend, r::SVector{6,T}, beti=1.0) -> SVector{6,T}

Track particle through a sector bending magnet using 4th-order symplectic integrator.

Includes:
- Entrance/exit edge focusing (Brown/SOLEIL/THOMX models)
- Curvature-dependent focusing (irho term)
- Multipole field components
- Fringe field corrections

The tracking sequence is:
1. Apply entrance misalignment (t1, r1)
2. Apply entrance edge focusing
3. Symplectic integration through body
4. Apply exit edge focusing
5. Apply exit misalignment (r2, t2)
"""
function pass!(elem::SBend{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    # Apply entrance misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    # Calculate inverse bending radius
    irho = elem.angle / elem.L
    
    # Prepare polynomial coefficients with kick angle correction
    polynom_b = elem.polynom_b
    polynom_a = elem.polynom_a
    if elem.L > zero(T)
        polynom_b = SVector{4,T}(
            polynom_b[1] - sin(elem.kick_angle[1]) / elem.L,
            polynom_b[2],
            polynom_b[3],
            polynom_b[4]
        )
        polynom_a = SVector{4,T}(
            polynom_a[1] + sin(elem.kick_angle[2]) / elem.L,
            polynom_a[2],
            polynom_a[3],
            polynom_a[4]
        )
    end
    
    # Apply entrance edge focusing
    if elem.fringe_bend_entrance != 0
        r = edge_fringe_entrance(r, irho, elem.e1, elem.fint1, elem.gap, elem.fringe_bend_entrance)
    end

    if !iszero(elem.fringe_quad_entrance)
        r = multipole_fringe(r, polynom_a, polynom_b, elem.max_order, one(T), 1, beti)
    end

    # Symplectic integration through body
    r = symplectic4_bend_pass(r, elem.L, polynom_a, polynom_b, irho,
                               elem.max_order, elem.num_int_steps, beti)

    if !iszero(elem.fringe_quad_exit)
        r = multipole_fringe(r, polynom_a, polynom_b, elem.max_order, -one(T), 1, beti)
    end

    # Apply exit edge focusing
    if elem.fringe_bend_exit != 0
        r = edge_fringe_exit(r, irho, elem.e2, elem.fint2, elem.gap, elem.fringe_bend_exit)
    end
    
    # Apply exit misalignment
    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    r = apply_misalignment(r, elem.t2)
    
    return r
end

# =============================================================================
# Exact Sector Bend Tracking
# =============================================================================

"""
    pxyz(momentum, px, py) -> T

Helper for exact bend Hamiltonian: `pz = sqrt(momentum^2 - px^2 - py^2)`.
Here `momentum` is `P/P0`, not `1/beta0 + delta_E`.
"""
@inline function pxyz(momentum::T, px::T, py::T) where T
    val = momentum^2 - px^2 - py^2
    return _safe_sqrt_pz(val)
end
# CTPS / non-Real overload (no comparison needed)
@inline function pxyz(momentum, px, py)
    return sqrt(momentum^2 - px^2 - py^2)
end

"""
    yrot!(r, phi, beti) -> SVector{6,T}

Rotation in free space (Forest 10.26).
"""
@inline function yrot(r::SVector{6,S}, phi::T, beti::T) where {T<:Real,S}
    if phi == zero(T)
        return r
    end
    
    energy_factor = beti + r[6]
    momentum = _momentum_norm(r[6], beti)
    c = cos(phi)
    s = sin(phi)
    pz = pxyz(momentum, r[2], r[4])
    
    p = c * pz - s * r[2]
    px_new = s * pz + c * r[2]
    x_new = r[1] * pz / p
    
    y_new = r[3] + r[1] * r[4] * s / p
    z_new = r[5] - energy_factor * r[1] * s / p
    
    return SVector(x_new, px_new, y_new, r[4], z_new, r[6])
end

"""
    bend_fringe(r, irho, gK, beti) -> SVector{6,T}

Hard-edge bend fringe (Forest 13.13).
"""
@inline function bend_fringe(r::SVector{6,S}, irho::T, gK::T, beti::T) where {T<:Real,S}
    b0 = irho
    energy_factor = beti + r[6]
    momentum = _momentum_norm(r[6], beti)
    pz = pxyz(momentum, r[2], r[4])
    px = r[2]
    py = r[4]
    
    xp = px / pz
    yp = py / pz
    
    term1 = 1 + xp^2 * (2 + yp^2)
    term2 = xp / (1 + yp^2)
    phi = -b0 * tan(b0 * gK * term1 * pz - atan(term2))
    
    px2 = px^2
    py2 = py^2
    pz2 = pz^2
    pz4 = pz2^2
    
    py2z2 = (py2 + pz2)
    arg_sec = (b0 * gK * (pz4 + px2 * (py2 + 2*pz2))) / (pz2 * pz) - atan(px * pz / (py2 + pz2))
    powsec = (one(T) / cos(arg_sec))^2   # sec² written so it also works on series types
    
    # Derivatives (simplified form matching JuTrack logic)
    # Note: These are complex derivatives. For TPSA matching we need exact form.
    # For now, implementing the core transformation for particles.
    
    # Using explicit derivatives from JuTrack source
    py4 = py2^2
    py6 = py4 * py2
    pz3 = pz2 * pz
    pz5 = pz4 * pz
    pz6 = pz4 * pz2
    
    py2z2_sq = py2z2^2
    denom = pz5 * (py4 + px2*pz2 + 2*py2*pz2 + pz4)
    
    dpx_num = -(b0 * (px2 * pz4 * (py2 - pz2) - pz6 * (py2 + pz2) + 
                b0 * gK * px * (pz2 * py2z2_sq * (2*py2 + 3*pz2) + 
                px2^2 * (3*py2*pz2 + 2*pz4) + 
                px2 * (3*py6 + 8*py4*pz2 + 9*py2*pz4 + 5*pz6)))) * powsec
    
    dpx = dpx_num / denom
    
    dpy_num = -(b0 * py * (px * pz4 * (py2 + pz2) + 
                b0 * gK * (-(pz4 * py2z2_sq) + px2^2 * (3*py2*pz2 + 4*pz4) + 
                px2 * (3*py6 + 10*py4*pz2 + 11*py2*pz4 + 3*pz6)))) * powsec
                
    dpy = dpy_num / denom
    
    dd_num = (b0 * energy_factor * (px * pz4 * (py2 - pz2) + b0 * gK *
              (-(pz4 * py2z2_sq) + px2^2 * (3*py2*pz2 + 2*pz4) + 
              px2 * (3*py6 + 8*py4*pz2 + 7*py2*pz4 + pz6)))) * powsec
              
    dd = dd_num / denom
    
    # symplectic correction
    disc = 1 - 2 * dpy * r[3]
    # A negative discriminant means the fringe map has no real solution at this
    # amplitude; report the particle as lost instead of throwing a DomainError.
    _check_pz2(disc) && return _lost_coords(r, T)
    yf = (2 * r[3]) / (1 + sqrt(disc))
    dxf = 0.5 * dpx * yf^2
    dct = 0.5 * dd * yf^2
    dpyf = phi * yf
    
    return SVector(r[1] + dxf, r[2], yf, r[4] - dpyf, r[5] + dct, r[6])
end

"""
    bend_edge(r, rhoinv, theta, beti) -> SVector{6,T}

Ideal wedge map (Forest 12.41).
"""
@inline function bend_edge(r::SVector{6,S}, rhoinv::T, theta::T, beti::T) where {T<:Real,S}
    if abs(rhoinv) < 1e-6
        return r
    end
    
    energy_factor = beti + r[6]
    momentum = _momentum_norm(r[6], beti)
    c = cos(theta)
    s = sin(theta)
    pz = pxyz(momentum, r[2], r[4])
    d2 = pxyz(momentum, zero(T), r[4])
    
    px_new = r[2] * c + (pz - rhoinv * r[1]) * s
    
    # dasin term
    val1 = r[2] / d2
    val2 = px_new / d2
    # clamp to [-1, 1] to avoid domain error
    val1 = _safe_clamp(val1, -one(T), one(T))
    val2 = _safe_clamp(val2, -one(T), one(T))
    dasin = asin(val1) - asin(val2)
    
    num = r[1] * (r[2] * sin(2*theta) + s^2 * (2*pz - rhoinv * r[1]))
    den = pxyz(momentum, px_new, r[4]) + pz * c - r[2] * s
    
    x_new = r[1] * c + num / den
    y_new = r[3] + r[4] * (theta / rhoinv + dasin / rhoinv)
    z_new = r[5] - energy_factor / rhoinv * (theta + dasin)
    
    return SVector(x_new, px_new, y_new, r[4], z_new, r[6])
end

"""
    exact_bend_body(r, irho, L, beti) -> SVector{6,T}

Exact bend body map (Forest 12.18).
"""
@inline function exact_bend_body(r::SVector{6,S}, irho::T, L::T, beti::T) where {T<:Real,S}
    energy_factor = beti + r[6]
    momentum = _momentum_norm(r[6], beti)
    pz = pxyz(momentum, r[2], r[4])
    
    if abs(irho) < 1e-6
        # Drift limit
        NormL = L / pz
        return SVector(
            r[1] + r[2] * NormL,
            r[2],
            r[3] + r[4] * NormL,
            r[4],
            r[5] - _drift_dz(L, r[2], r[4], r[6], beti, pz),
            r[6]
        )
    else
        # pz − 1 without cancellation: (p² − 1 − px² − py²)/(pz + 1) with
        # p² − 1 = δE(2/β0 + δE) exactly.
        p2m1 = r[6] * (2 * beti + r[6])
        pzm1 = (p2m1 - r[2] * r[2] - r[4] * r[4]) / (pz + one(T))
        pzmx = pzm1 - r[1] * irho
        cs = cos(irho * L)
        sn = sin(irho * L)
        
        px_new = r[2] * cs + pzmx * sn
        
        d2 = pxyz(momentum, zero(T), r[4])
        val1 = r[2] / d2
        val2 = px_new / d2
        val1 = _safe_clamp(val1, -one(T), one(T))
        val2 = _safe_clamp(val2, -one(T), one(T))
        
        dasin_rel = (asin(val1) - asin(val2)) / irho   # dasin − L
        dasin = L + dasin_rel
        
        # x_new = [(pz_new − 1) − pzmx cos + px sin]/h with both pz − 1 terms in
        # the cancellation-free form above. The direct Forest expression
        # (pz_new − pzmx cos + px sin − 1)/h subtracts O(1) quantities and
        # loses ~ε/h ≈ 1e-14 m per step, which random-walks to 1e-12 m over a
        # ring and corrupts finite-difference Jacobians (step 1e-8) at 1e-4.
        pz_new = pxyz(momentum, px_new, r[4])
        pzm1_new = (p2m1 - px_new * px_new - r[4] * r[4]) / (pz_new + one(T))
        x_new = (pzm1_new - pzmx * cs + r[2] * sn) / irho
        y_new = r[3] + r[4] * dasin
        # energy_factor·dasin − L/β0 = L δE + energy_factor·(dasin − L): no O(L) cancellation.
        z_new = r[5] - (L * r[6] + energy_factor * dasin_rel)
        
        return SVector(x_new, px_new, y_new, r[4], z_new, r[6])
    end
end

# The exact-bend kernels are `@inline` for `Real` coordinates, where that is
# free. For series or dual coordinates every arithmetic operation is itself a
# sizeable function, and inlining four bodies, two fringes and two wedges per
# `pass!` produced a single function LLVM took 30 s to compile; through this
# barrier the kernels compile once each (~2 s in total) and are called.
@inline _exact_kernel(f::F, r::SVector{6,<:Real}, args...) where F = f(r, args...)
@noinline _exact_kernel(f::F, r::SVector{6}, args...) where F = f(r, args...)

"""
    pass!(elem::ExactSBend, r, beti)

Track through Exact Sector Bend.
"""
function pass!(elem::ExactSBend{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    # Entrance Misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    irho = elem.angle / elem.L

    # Fold KickAngle into the multipoles once; fringes and kicks share them.
    polynom_b = elem.polynom_b
    polynom_a = elem.polynom_a
    if elem.L > zero(T)
        polynom_b = SVector{4,T}(
            polynom_b[1] - sin(elem.kick_angle[1]) / elem.L,
            polynom_b[2], polynom_b[3], polynom_b[4])
        polynom_a = SVector{4,T}(
            polynom_a[1] + sin(elem.kick_angle[2]) / elem.L,
            polynom_a[2], polynom_a[3], polynom_a[4])
    end

    # JuTrack/Forest ordering: rotation, bend fringe, multipole fringe, edge.
    r = _exact_kernel(yrot, r, elem.e1, beti)
    if elem.fringe_bend_entrance != 0
        r = _exact_kernel(bend_fringe, r, irho, elem.gk, beti)
    end
    if !iszero(elem.fringe_quad_entrance)
        r = _exact_kernel(multipole_fringe, r, polynom_a, polynom_b, elem.max_order, one(T), 1, beti)
    end
    r = _exact_kernel(bend_edge, r, irho, -elem.e1, beti)

    # Integrator (Body): drift-kick-drift using exact_bend_body.
    if elem.num_int_steps == 0
        r = _exact_kernel(exact_bend_body, r, irho, elem.L, beti)
    else
        SL = elem.L / elem.num_int_steps
        L1 = SL * T(DRIFT1)
        L2 = SL * T(DRIFT2)
        K1 = SL * T(KICK1)
        K2 = SL * T(KICK2)

        for _ in 1:elem.num_int_steps
            r = _exact_kernel(exact_bend_body, r, irho, L1, beti)
            r = _exact_kernel(strthinkick, r, polynom_a, polynom_b, K1, elem.max_order)
            r = _exact_kernel(exact_bend_body, r, irho, L2, beti)
            r = _exact_kernel(strthinkick, r, polynom_a, polynom_b, K2, elem.max_order)
            r = _exact_kernel(exact_bend_body, r, irho, L2, beti)
            r = _exact_kernel(strthinkick, r, polynom_a, polynom_b, K1, elem.max_order)
            r = _exact_kernel(exact_bend_body, r, irho, L1, beti)
        end
    end

    # JuTrack/Forest exit ordering reverses the entrance composition.
    r = _exact_kernel(bend_edge, r, irho, -elem.e2, beti)
    if !iszero(elem.fringe_quad_exit)
        r = _exact_kernel(multipole_fringe, r, polynom_a, polynom_b, elem.max_order, -one(T), 1, beti)
    end
    if elem.fringe_bend_exit != 0
        r = _exact_kernel(bend_fringe, r, -irho, elem.gk, beti)
    end
    
    # Coordinate Rotation (Exit)
    r = _exact_kernel(yrot, r, elem.e2, beti)
    
    # Exit Misalignment
    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    r = apply_misalignment(r, elem.t2)
    
    return r
end

# =============================================================================
# Space-Charge Canonical Element Wrappers
# =============================================================================


function pass!(elem::DriftSC{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    r = drift6(r, elem.L, beti)
    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    return apply_misalignment(r, elem.t2)
end

@inline function _sc_polynom_b(elem::Union{QuadrupoleSC{T,N}, SextupoleSC{T,N}, OctupoleSC{T,N}}) where {T,N}
    return SVector{4, T}(elem.k0, elem.k1, elem.k2 / 2, elem.k3 / 6)
end

function pass!(elem::QuadrupoleSC{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end

    polynom_b = _sc_polynom_b(elem)
    polynom_a = elem.polynom_a
    if elem.L > zero(T)
        polynom_b = SVector{4, T}(polynom_b[1] - sin(elem.kick_angle[1]) / elem.L, polynom_b[2], polynom_b[3], polynom_b[4])
        polynom_a = SVector{4, T}(polynom_a[1] + sin(elem.kick_angle[2]) / elem.L, polynom_a[2], polynom_a[3], polynom_a[4])
    end
    r = symplectic4_pass(r, elem.L, polynom_a, polynom_b, clamp(elem.max_order, 0, 3), elem.num_int_steps, beti)

    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    return apply_misalignment(r, elem.t2)
end

function pass!(elem::SextupoleSC{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    r = symplectic4_pass(r, elem.L, elem.polynom_a, _sc_polynom_b(elem), clamp(elem.max_order, 0, 3), elem.num_int_steps, beti)
    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    return apply_misalignment(r, elem.t2)
end

function pass!(elem::OctupoleSC{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    r = symplectic4_pass(r, elem.L, elem.polynom_a, _sc_polynom_b(elem), clamp(elem.max_order, 0, 3), elem.num_int_steps, beti)
    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    return apply_misalignment(r, elem.t2)
end

function pass!(elem::SBendSC{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end

    if iszero(elem.L)
        return _lost_coords(r, T)
    end
    irho = elem.angle / elem.L
    polynom_b = elem.polynom_b
    polynom_a = elem.polynom_a
    if elem.L > zero(T)
        polynom_b = SVector{4, T}(polynom_b[1] - sin(elem.kick_angle[1]) / elem.L, polynom_b[2], polynom_b[3], polynom_b[4])
        polynom_a = SVector{4, T}(polynom_a[1] + sin(elem.kick_angle[2]) / elem.L, polynom_a[2], polynom_a[3], polynom_a[4])
    end

    if elem.fringe_bend_entrance != 0
        r = edge_fringe_entrance(r, irho, elem.e1, elem.fint1, elem.gap, elem.fringe_bend_entrance)
    end
    r = symplectic4_bend_pass(r, elem.L, polynom_a, polynom_b, irho, clamp(elem.max_order, 0, 3), elem.num_int_steps, beti)
    if elem.fringe_bend_exit != 0
        r = edge_fringe_exit(r, irho, elem.e2, elem.fint2, elem.gap, elem.fringe_bend_exit)
    end

    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    return apply_misalignment(r, elem.t2)
end

# =============================================================================
# Linear Bend (LBend)
# =============================================================================

# The linear bend body branches on the sign of the momentum-dependent focusing
# G1, G2, so it is defined for Real coordinate types only (Float32/64, Dual);
# TPSA coordinates cannot flow through it.
@inline function _lbend_body(r::SVector{6,S}, L::T, grd::T, b_angle::T,
                             by_error::T, beti::T) where {T<:Real,S<:Real}
    if iszero(L)
        return r
    end

    momentum = _momentum_norm(r[6], beti)
    p_norm = inv(momentum)
    momentum_jacobian = (beti + r[6]) * p_norm
    Kx = b_angle / L
    G1 = (Kx * Kx + grd) * p_norm
    G2 = -grd * p_norm
    tol = sqrt(eps(T))

    # Matrix entries depend on the particle momentum: type them accordingly so
    # the branches below do not change the variable types.
    U = typeof(G1)
    MHD = one(U); M12 = U(L); M21 = zero(U)
    MVD = one(U); M34 = U(L); M43 = zero(U)
    arg1 = zero(U); arg2 = zero(U)
    sqrtG1 = zero(U); sqrtG2 = zero(U)

    if abs(G1) >= tol
        if G1 > zero(T)
            sqrtG1 = sqrt(G1)
            arg1 = L * sqrtG1
            MHD = cos(arg1); M12 = sin(arg1) / sqrtG1; M21 = -sin(arg1) * sqrtG1
        else
            sqrtG1 = sqrt(-G1)
            arg1 = L * sqrtG1
            MHD = cosh(arg1); M12 = sinh(arg1) / sqrtG1; M21 = sinh(arg1) * sqrtG1
        end
    end

    if abs(G2) >= tol
        if G2 > zero(T)
            sqrtG2 = sqrt(G2)
            arg2 = L * sqrtG2
            MVD = cos(arg2); M34 = sin(arg2) / sqrtG2; M43 = -sin(arg2) * sqrtG2
        else
            sqrtG2 = sqrt(-G2)
            arg2 = L * sqrtG2
            MVD = cosh(arg2); M34 = sinh(arg2) / sqrtG2; M43 = sinh(arg2) * sqrtG2
        end
    end

    x = r[1]
    xpr = r[2] * p_norm
    y = r[3]
    ypr = r[4] * p_norm
    delta_p = momentum - one(T)
    dterm = delta_p * p_norm - by_error

    x_new = MHD * x + M12 * xpr
    px_new = (M21 * x + MHD * xpr) / p_norm

    if abs(G1) < tol
        x_new += dterm * L * L * Kx / 2
        px_new += dterm * L * Kx / p_norm
    elseif G1 > zero(T)
        x_new += dterm * (one(T) - cos(arg1)) * Kx / G1
        px_new += dterm * sin(arg1) * Kx / (sqrtG1 * p_norm)
    else
        x_new += dterm * (one(T) - cosh(arg1)) * Kx / G1
        px_new += dterm * sinh(arg1) * Kx / (sqrtG1 * p_norm)
    end

    y_new = MVD * y + M34 * ypr
    py_new = (M43 * y + MVD * ypr) / p_norm

    longitudinal_increment = -xpr * xpr * (L + MHD * M12) / 4
    if abs(G1) >= tol
        longitudinal_increment -= (L - MHD * M12) * (x * x * G1 + dterm * dterm * Kx * Kx / G1 - 2 * x * Kx * dterm) / 4
        longitudinal_increment -= M12 * M21 * (x * xpr - xpr * dterm * Kx / G1) / 2
        longitudinal_increment -= Kx * x * M12 + xpr * (one(T) - MHD) * Kx / G1 + dterm * (L - M12) * Kx * Kx / G1
    end
    longitudinal_increment -= ((L - MVD * M34) * y * y * G2 + ypr * ypr * (L + MVD * M34)) / 4
    longitudinal_increment -= M34 * M43 * y * ypr / 2
    z_new = r[5] + momentum_jacobian * longitudinal_increment

    return SVector(x_new, px_new, y_new, py_new, z_new, r[6])
end

function pass!(elem::LBend{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end

    if iszero(elem.L)
        return _lost_coords(r, T)
    end
    irho = elem.angle / elem.L
    r = edge_fringe_entrance(r, irho, elem.e1, elem.fint1, elem.full_gap, 1)
    r = _lbend_body(r, elem.L, elem.K, elem.angle, elem.by_error, beti)
    r = edge_fringe_exit(r, irho, elem.e2, elem.fint2, elem.full_gap, 1)

    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    return apply_misalignment(r, elem.t2)
end

# =============================================================================
# Auxiliary Canonical Elements
# =============================================================================

"""
    outside_aperture(r, r_apertures, e_apertures) -> Bool

True when the coordinate vector `r` lies outside the element apertures,
mirroring JuTrack's `check_lost_aperture`. `r_apertures =
[xmin, xmax, ymin, ymax, 0, 0]` defines a rectangular aperture and
`e_apertures = [ax, ay, 0, 0, 0, 0]` an elliptical one with semi-axes
`ax`, `ay`; all-zero vectors disable the respective check.
"""
function outside_aperture(r::SVector{6,S}, r_apertures::AbstractVector{T},
                          e_apertures::AbstractVector{T}) where {S,T}
    if !iszero(r_apertures)
        (r[1] < r_apertures[1] || r[1] > r_apertures[2] ||
         r[3] < r_apertures[3] || r[3] > r_apertures[4]) && return true
    end
    if !iszero(e_apertures) && e_apertures[1] > zero(T) && e_apertures[2] > zero(T)
        r[1]^2 / e_apertures[1]^2 + r[3]^2 / e_apertures[2]^2 > one(T) && return true
    end
    return false
end

# Elements either carry aperture fields (magnets, drifts, cavities, ...) or
# none at all (collective elements, maps); resolve once per element so the
# hot multi-particle loop stays branch-free.
@inline function _elem_apertures(elem)
    hasfield(typeof(elem), :r_apertures) ?
        (getfield(elem, :r_apertures), getfield(elem, :e_apertures)) :
        (nothing, nothing)
end

@inline _apertures_active(rap, eap) =
    rap !== nothing && !(iszero(rap) && iszero(eap))

"""
    aperture_lost(r, rap, eap) -> Bool

True when the element carries an active aperture and `r` lies outside it.
Returns `false` for non-Real coordinate types (TPSA), where the ordered
comparisons an aperture test needs are not defined.
"""
@inline aperture_lost(r::SVector{6,S}, rap, eap) where {S<:Real} =
    _apertures_active(rap, eap) && outside_aperture(r, rap, eap)
@inline aperture_lost(::SVector{6}, _rap, _eap) = false



function pass!(elem::SpaceCharge{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    # TrackPad's single-particle API has no bunch moments/current.
    # Keep SPACECHARGE as a no-op here (JuTrack also gives zero kick at I=0).
    return r
end

"""
    _y_rotation(r, beti, angle) -> SVector{6}

Rotate the reference frame about the vertical axis by `angle` (Forest 10.26 in
the exact energy coordinates). Shared by `Patch` (pitches) and `YRotation`.
Generic in the coordinate type: the loss branches use `_check_pz2`/`_check_tiny`
so TPSA coordinates pass through unchanged.
"""
@inline function _y_rotation(r::SVector{6,S}, beti::T, angle::T) where {T<:Real,S}
    iszero(angle) && return r
    cosine = cos(angle)
    sine = sin(angle)
    tangent = tan(angle)
    pz_squared = one(T) + 2 * r[6] * beti + r[6]^2 - r[2]^2 - r[4]^2
    _check_pz2(pz_squared) && return _lost_coords(r, T)
    pz = sqrt(pz_squared)
    denominator = one(T) - tangent * r[2] / pz
    _check_tiny(denominator, sqrt(eps(T))) && return _lost_coords(r, T)
    x_new = r[1] / (cosine * denominator)
    px_new = cosine * r[2] + sine * pz
    y_new = r[3] + tangent * r[1] * r[4] / (pz * denominator)
    z_new = r[5] - tangent * r[1] * (beti + r[6]) / (pz * denominator)
    return SVector(x_new, px_new, y_new, r[4], z_new, r[6])
end

"""Rotation about the horizontal axis, expressed through `_y_rotation` with x<->y swapped."""
@inline function _x_rotation(r::SVector{6,S}, beti::T, angle::T) where {T<:Real,S}
    swapped = SVector(r[3], r[4], r[1], r[2], r[5], r[6])
    q = _y_rotation(swapped, beti, angle)
    return SVector(q[3], q[4], q[1], q[2], q[5], q[6])
end

function pass!(patch::Patch{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    pz_squared = one(T) + 2 * r[6] * beti + r[6]^2 - r[2]^2 - r[4]^2
    _check_pz2(pz_squared) && return _lost_coords(r, T)
    pz = sqrt(pz_squared)
    x = r[1] - (patch.x_offset + patch.z_offset * r[2] / pz)
    y = r[3] - (patch.y_offset + patch.z_offset * r[4] / pz)
    z = r[5] + patch.z_offset * (beti + r[6]) / pz + T(C_LIGHT) * patch.t_offset
    r = SVector(x, r[2], y, r[4], z, r[6])

    if !iszero(patch.x_pitch)
        r = _x_rotation(r, beti, patch.x_pitch)
    end
    r = _y_rotation(r, beti, -patch.y_pitch)

    if !iszero(patch.tilt)
        cosine = cos(patch.tilt)
        sine = sin(patch.tilt)
        x_t = cosine * r[1] + sine * r[3]
        y_t = -sine * r[1] + cosine * r[3]
        px_t = cosine * r[2] + sine * r[4]
        py_t = -sine * r[2] + cosine * r[4]
        r = SVector(x_t, px_t, y_t, py_t, r[5], r[6])
    end
    return r
end

function pass!(elem::Translation{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    # A rigid frame displacement by (dx, dy, ds): lateral origin offsets are
    # pure coordinate redefinitions, while the longitudinal ds acts like a
    # drift slice of length ds and therefore shares drift6's exact update,
    # including the reference-path compensation that keeps a synchronous
    # particle's z fixed. JuTrack's TRANSLATION instead applies
    # +/- ds*(1/beta + delta)/pz without compensation under its own
    # c(t-t0) axis; TrackPad deliberately uses the drift-consistent form.
    pz2 = one(T) + 2 * r[6] * beti + r[6]^2 - r[2]^2 - r[4]^2
    _check_pz2(pz2) && return _lost_coords(r, T)
    norm_ds = elem.ds / sqrt(pz2)
    x_new = r[1] - elem.dx + norm_ds * r[2]
    y_new = r[3] - elem.dy + norm_ds * r[4]
    z_new = r[5] - (norm_ds * (beti + r[6]) - elem.ds * beti)
    return SVector(x_new, r[2], y_new, r[4], z_new, r[6])
end

function pass!(elem::YRotation{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    return _y_rotation(r, beti, -elem.angle)
end

# =============================================================================
# Wiggler
# =============================================================================

@inline function _sinc_taylor(x::T) where T
    x2 = x * x
    return one(T) - x2 / 6 * (one(T) - x2 / 20 * (one(T) - x2 / 42 * (one(T) - x2 / 72)))
end

@inline function _wig_ax_axpy(elem::Wiggler{T,N,V}, r::SVector{6,S}, Zw::T, Aw::T, Po::T) where {T,N,V,S}
    x = r[1]
    y = r[3]
    kw = T(2pi) / elem.lw
    ax = zero(T) * x   # typed like the coordinates
    axpy = zero(T) * x   # typed like the coordinates

    @inbounds for i in 1:elem.NHharm
        if 6 * i > length(elem.By)
            break
        end
        base = (i - 1) * 6
        HCw = T(elem.By[base + 2]) * Aw / Po
        kx = T(elem.By[base + 3]) * kw
        ky = T(elem.By[base + 4]) * kw
        kz = T(elem.By[base + 5]) * kw
        tz = T(elem.By[base + 6])

        cx = cos(kx * x)
        chy = cosh(ky * y)
        sz = sin(kz * Zw + tz)
        ax += HCw * (kw / kz) * cx * chy * sz

        shy = sinh(ky * y)
        sxkx = abs(kx / kw) > T(1e-6) ? sin(kx * x) / kx : x * _sinc_taylor(kx * x)
        axpy += HCw * (kw / kz) * ky * sxkx * shy * sz
    end

    @inbounds for i in 1:elem.NVharm
        if 6 * i > length(elem.Bx)
            break
        end
        base = (i - 1) * 6
        VCw = T(elem.Bx[base + 2]) * Aw / Po
        kx = T(elem.Bx[base + 3]) * kw
        ky = T(elem.Bx[base + 4]) * kw
        kz = T(elem.Bx[base + 5]) * kw
        tz = T(elem.Bx[base + 6])

        shx = sinh(kx * x)
        sy = sin(ky * y)
        sz = sin(kz * Zw + tz)
        ax += VCw * (kw / kz) * (ky / kx) * shx * sy * sz

        chx = cosh(kx * x)
        cy = cos(ky * y)
        axpy += VCw * (kw / kz) * (ky / kx)^2 * chx * cy * sz
    end

    return ax, axpy
end

@inline function _wig_ay_aypx(elem::Wiggler{T,N,V}, r::SVector{6,S}, Zw::T, Aw::T, Po::T) where {T,N,V,S}
    x = r[1]
    y = r[3]
    kw = T(2pi) / elem.lw
    ay = zero(T) * x   # typed like the coordinates
    aypx = zero(T) * x   # typed like the coordinates

    @inbounds for i in 1:elem.NHharm
        if 6 * i > length(elem.By)
            break
        end
        base = (i - 1) * 6
        HCw = T(elem.By[base + 2]) * Aw / Po
        kx = T(elem.By[base + 3]) * kw
        ky = T(elem.By[base + 4]) * kw
        kz = T(elem.By[base + 5]) * kw
        tz = T(elem.By[base + 6])

        sx = sin(kx * x)
        shy = sinh(ky * y)
        sz = sin(kz * Zw + tz)
        ay += HCw * (kw / kz) * (kx / ky) * sx * shy * sz

        cx = cos(kx * x)
        chy = cosh(ky * y)
        aypx += HCw * (kw / kz) * (kx / ky)^2 * cx * chy * sz
    end

    @inbounds for i in 1:elem.NVharm
        if 6 * i > length(elem.Bx)
            break
        end
        base = (i - 1) * 6
        VCw = T(elem.Bx[base + 2]) * Aw / Po
        kx = T(elem.Bx[base + 3]) * kw
        ky = T(elem.Bx[base + 4]) * kw
        kz = T(elem.Bx[base + 5]) * kw
        tz = T(elem.Bx[base + 6])

        chx = cosh(kx * x)
        cy = cos(ky * y)
        sz = sin(kz * Zw + tz)
        ay += VCw * (kw / kz) * chx * cy * sz

        shx = sinh(kx * x)
        syky = abs(ky / kw) > T(1e-6) ? sin(ky * y) / ky : y * _sinc_taylor(ky * y)
        aypx += VCw * (kw / kz) * kx * shx * syky * sz
    end

    return ay, aypx
end

@inline function _wig_map_2nd(elem::Wiggler{T,N,V}, r::SVector{6,S}, dl::T,
                              Zw::T, Aw::T, Po::T, beti::T) where {T,N,V,S}
    delta = r[6]
    momentum = _momentum_norm(delta, beti)
    inv_momentum = inv(momentum)
    momentum_jacobian = (beti + delta) * inv_momentum
    dld = dl * inv_momentum
    dl2 = dl / 2
    dl2d = dl2 * inv_momentum

    Zw += dl2
    x = r[1]
    px = r[2]
    y = r[3]
    py = r[4]
    z = r[5]

    ay, aypx = _wig_ay_aypx(elem, SVector(x, px, y, py, z, delta), Zw, Aw, Po)
    px -= aypx
    py -= ay
    y += dl2d * py
    z -= momentum_jacobian * (dl2d / 2) * py^2 * inv_momentum

    ay, aypx = _wig_ay_aypx(elem, SVector(x, px, y, py, z, delta), Zw, Aw, Po)
    px += aypx
    py += ay

    ax, axpy = _wig_ax_axpy(elem, SVector(x, px, y, py, z, delta), Zw, Aw, Po)
    px -= ax
    py -= axpy
    x += dld * px
    z -= momentum_jacobian * (dld / 2) * px^2 * inv_momentum

    ax, axpy = _wig_ax_axpy(elem, SVector(x, px, y, py, z, delta), Zw, Aw, Po)
    px += ax
    py += axpy

    ay, aypx = _wig_ay_aypx(elem, SVector(x, px, y, py, z, delta), Zw, Aw, Po)
    px -= aypx
    py -= ay
    y += dl2d * py
    z -= momentum_jacobian * (dl2d / 2) * py^2 * inv_momentum

    ay, aypx = _wig_ay_aypx(elem, SVector(x, px, y, py, z, delta), Zw, Aw, Po)
    px += aypx
    py += ay
    Zw += dl2

    return SVector(x, px, y, py, z, delta), Zw
end

@inline function _wig_pass_4th(elem::Wiggler{T,N,V}, r::SVector{6,S}, beti::T) where {T,N,V,S}
    PN = elem.Nsteps
    Nw = round(Int, elem.L / elem.lw)
    Nstep = PN * max(1, Nw)
    dl = elem.lw / T(PN)
    dl1 = dl * T(1.3512071919596573)
    dl0 = dl * T(-1.7024143839193146)

    Po = sqrt((elem.energy / elem.mass)^2 - one(T))
    Aw = T(1e-9) * T(C_LIGHT) / (elem.mass * T(1e-9)) / T(2pi) * elem.lw * elem.Bmax * abs(elem.charge)
    Zw = zero(T)

    @inbounds for _ in 1:Nstep
        r, Zw = _wig_map_2nd(elem, r, dl1, Zw, Aw, Po, beti)
        r, Zw = _wig_map_2nd(elem, r, dl0, Zw, Aw, Po, beti)
        r, Zw = _wig_map_2nd(elem, r, dl1, Zw, Aw, Po, beti)
    end
    return r
end

@inline function _wig_B(elem::Wiggler{T,N,V}, r::SVector{6,S}, Zw::T) where {T,N,V,S}
    x = r[1]
    y = r[3]
    kw = T(2pi) / elem.lw
    PB0 = elem.Bmax
    Bxv = zero(T) * x   # typed like the coordinates
    Byv = zero(T) * x   # typed like the coordinates
    Bzv = zero(T) * x   # typed like the coordinates

    @inbounds for i in 1:elem.NHharm
        if 6 * i > length(elem.By)
            break
        end
        base = (i - 1) * 6
        HCw = T(elem.By[base + 2])
        kx = T(elem.By[base + 3]) * kw
        ky = T(elem.By[base + 4]) * kw
        kz = T(elem.By[base + 5]) * kw
        tz = T(elem.By[base + 6])

        sx = sin(kx * x)
        cx = cos(kx * x)
        chy = cosh(ky * y)
        shy = sinh(ky * y)
        cz = cos(kz * Zw + tz)
        sz = sin(kz * Zw + tz)

        Bxv += PB0 * HCw * (kx / ky) * sx * shy * cz
        Byv -= PB0 * HCw * cx * chy * cz
        Bzv += PB0 * HCw * (kz / ky) * cx * shy * sz
    end

    @inbounds for i in 1:elem.NVharm
        if 6 * i > length(elem.Bx)
            break
        end
        base = (i - 1) * 6
        VCw = T(elem.Bx[base + 2])
        kx = T(elem.Bx[base + 3]) * kw
        ky = T(elem.Bx[base + 4]) * kw
        kz = T(elem.Bx[base + 5]) * kw
        tz = T(elem.Bx[base + 6])

        shx = sinh(kx * x)
        chx = cosh(kx * x)
        sy = sin(ky * y)
        cy = cos(ky * y)
        cz = cos(kz * Zw + tz)
        sz = sin(kz * Zw + tz)

        Bxv += PB0 * VCw * chx * cy * cz
        Byv -= PB0 * VCw * (ky / kx) * shx * sy * cz
        Bzv -= PB0 * VCw * (kz / kx) * cy * shx * sz
    end

    return Bxv, Byv, Bzv
end

@inline function _wig_radiation_kicks(r::SVector{6,S}, Bxv, Byv, Po::T, srCoef::T, dl::T,
                                      mass::T) where {T,S}
    B2 = Bxv^2 + Byv^2
    if iszero(B2)
        return r
    end
    H = Po * mass / T(C_LIGHT)
    irho2 = B2 / (H * H)
    dFactor = (one(T) + r[6])^2
    dDelta = -srCoef * dFactor * irho2 * dl
    scale = one(T) + dDelta
    return SVector(r[1], r[2] * scale, r[3], r[4] * scale, r[5], r[6] + dDelta)
end

@inline function _wig_pass_4th_rad(elem::Wiggler{T,N,V}, r::SVector{6,S}, beti::T) where {T,N,V,S}
    PN = elem.Nsteps
    Nw = round(Int, elem.L / elem.lw)
    Nstep = PN * max(1, Nw)
    SL = elem.lw / T(PN)
    dl1 = SL * T(1.3512071919596573)
    dl0 = SL * T(-1.7024143839193146)

    gamma = elem.energy / elem.mass
    Po = sqrt(gamma^2 - one(T))
    Aw = T(1e-9) * T(C_LIGHT) / (elem.mass * T(1e-9)) / T(2pi) * elem.lw * elem.Bmax * abs(elem.charge)
    srCoef = T(2.0 / 3.0) * T(classical_radius(elem.mass, elem.charge)) * gamma^3
    Zw = zero(T)

    ax, _ = _wig_ax_axpy(elem, r, Zw, Aw, Po)
    ay, _ = _wig_ay_aypx(elem, r, Zw, Aw, Po)
    r = SVector(r[1], r[2] - ax, r[3], r[4] - ay, r[5], r[6])
    Bxv, Byv, _ = _wig_B(elem, r, Zw)
    r = _wig_radiation_kicks(r, Bxv, Byv, Po, srCoef, SL, elem.mass)
    r = SVector(r[1], r[2] + ax, r[3], r[4] + ay, r[5], r[6])

    @inbounds for _ in 1:Nstep
        r, Zw = _wig_map_2nd(elem, r, dl1, Zw, Aw, Po, beti)
        r, Zw = _wig_map_2nd(elem, r, dl0, Zw, Aw, Po, beti)
        r, Zw = _wig_map_2nd(elem, r, dl1, Zw, Aw, Po, beti)

        ax, _ = _wig_ax_axpy(elem, r, Zw, Aw, Po)
        ay, _ = _wig_ay_aypx(elem, r, Zw, Aw, Po)
        r = SVector(r[1], r[2] - ax, r[3], r[4] - ay, r[5], r[6])
        Bxv, Byv, _ = _wig_B(elem, r, Zw)
        r = _wig_radiation_kicks(r, Bxv, Byv, Po, srCoef, SL, elem.mass)
        r = SVector(r[1], r[2] + ax, r[3], r[4] + ay, r[5], r[6])
    end

    return r
end

function pass!(elem::Wiggler{T,N,V}, r::SVector{6,S}, beti::T=one(T)) where {T,N,V,S}
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end

    if elem.rad == 1
        r = _wig_pass_4th_rad(elem, r, beti)
    else
        r = _wig_pass_4th(elem, r, beti)
    end

    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    return apply_misalignment(r, elem.t2)
end

# =============================================================================
# Crab / Accelerating Cavities
# =============================================================================

function pass!(elem::CrabCavity{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    volt = elem.volt * (one(T) + elem.errors[1])
    phi = elem.phi + elem.errors[2]
    p0c = max(_reference_p0c(elem.energy, beti), eps(T))
    ang = -elem.k * r[5] + phi

    if elem.L > zero(T)
        r = drift6(r, elem.L / 2, beti)
    end
    kick = elem.charge * volt / p0c
    px_new = r[2] + kick * sin(ang)
    delta_new = r[6] - elem.k * kick * r[1] * cos(ang)
    r = SVector(r[1], px_new, r[3], r[4], r[5], delta_new)
    if elem.L > zero(T)
        r = drift6(r, elem.L / 2, beti)
    end
    return r
end

function pass!(elem::AccelCavity{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    p0c = max(_reference_p0c(elem.energy, beti), eps(T))
    sv = sin(-elem.k * r[5] + elem.phis) - sin(elem.phis)
    delta_new = r[6] + (elem.charge * elem.volt / p0c) * sv
    return SVector(r[1], r[2], r[3], r[4], r[5], delta_new)
end

# =============================================================================
# Longitudinal Map / Lorentz transforms
# =============================================================================

@inline _rf_h(rf::RFCavity{T,N}) where {T,N} = rf.h
@inline _rf_h(rf::AccelCavity{T,N}) where {T,N} = rf.h
@inline _rf_h(rf::CrabCavity{T,N}) where {T,N} = one(T)
@inline _rf_h(::AbstractElement) = 1.0

@inline _rf_k(rf::RFCavity{T,N}) where {T,N} = T(2pi) * rf.freq / T(C_LIGHT)
@inline _rf_k(rf::AccelCavity{T,N}) where {T,N} = rf.k
@inline _rf_k(rf::CrabCavity{T,N}) where {T,N} = rf.k
@inline _rf_k(::AbstractElement) = 0.0

function pass!(elem::LongitudinalRFMap{T,E}, r::SVector{6,S}, beti::T=one(T)) where {T,E,S}
    k = T(_rf_k(elem.rf))
    if abs(k) <= eps(T)
        return r
    end
    h = T(_rf_h(elem.rf))
    beta = inv(beti)
    eta = elem.alphac - (one(T) - beta * beta)          # slip factor η = αc − 1/γ0²
    # One-turn slip in the stored coordinates. With z = s/β0 − ct the closed
    # orbit at relative momentum δP arrives with Δz = −(C/β0) η δP per turn
    # (C = hλ_rf = 2πh/k), and δP = δE/β0 to first order, hence the β0⁻²:
    # Δz = −C η δE / β0². Verified against the path length of the off-momentum
    # closed orbit and against ∮D/ρ ds on a β0 = 0.875 proton ring.
    z_new = r[5] - (T(2pi) * h * eta * beti * beti / k) * r[6]
    return SVector(r[1], r[2], r[3], r[4], z_new, r[6])
end

function pass!(elem::LorentzBoost{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    if elem.mode != 0
        return r
    end
    x_new = r[1] - elem.tanang * r[5]
    z_new = r[5] / elem.cosang
    delta_new = elem.tanang * elem.cosang * r[2] + elem.cosang * r[6]
    return SVector(x_new, r[2], r[3], r[4], z_new, delta_new)
end

function pass!(elem::InvLorentzBoost{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    if elem.mode != 0
        return r
    end
    x_new = r[1] + elem.sinang * r[5]
    delta_new = (r[6] - elem.sinang * r[2]) / elem.cosang
    return SVector(x_new, r[2], r[3], r[4], r[5] * elem.cosang, delta_new)
end

# =============================================================================
# Strong Beam-Beam
#
# Transverse field of a bi-Gaussian charge distribution, evaluated with the
# Bassetti-Erskine formula through the Faddeeva function w(z) = exp(-z^2) erfc(-iz).
# Everything here is plain arithmetic on real/complex numbers (no special-function
# library calls), so the same code runs in the packed KernelAbstractions kernel.
# =============================================================================

"""
    _weideman_coefficients(N) -> NTuple{N,Float64}

Coefficients of Weideman's rational approximation to the Faddeeva function
(J. A. C. Weideman, SIAM J. Numer. Anal. 31 (1994) 1497), computed once at load
time by a direct DFT so no FFT dependency is needed.
"""
function _weideman_coefficients(N::Int)
    M = 2N
    M2 = 2M
    L = sqrt(N / sqrt(2.0))
    # Samples of f(theta) = exp(-t^2) (L^2 + t^2), t = L tan(theta/2), on the
    # uniform grid theta_j = j*pi/M, j = 0..M2-1 (f(pi) = 0).
    f = zeros(M2)
    for j in 0:M2-1
        theta = j * pi / M
        theta > pi && (theta -= 2pi)
        if abs(abs(theta) - pi) < 1e-12
            f[j+1] = 0.0
        else
            t = L * tan(theta / 2)
            f[j+1] = exp(-t^2) * (L^2 + t^2)
        end
    end
    a = zeros(N)
    for n in 1:N
        acc = 0.0
        for j in 0:M2-1
            acc += f[j+1] * cos(2pi * n * j / M2)
        end
        a[n] = acc / M2
    end
    return ntuple(i -> a[i], N)
end

const _FADDEEVA_N = 48
const _FADDEEVA_L = sqrt(_FADDEEVA_N / sqrt(2.0))
const _FADDEEVA_A = _weideman_coefficients(_FADDEEVA_N)

"""
    faddeeva_w(z::Complex) -> Complex

Faddeeva function `w(z) = exp(-z^2) * erfc(-i z)` for any complex `z`.
Weideman's N=48 rational approximation in the upper half-plane (relative error
~1e-14 in Float64), reflected to the lower half-plane with
`w(z) = 2 exp(-z^2) - w(-z)`. Allocation-free and GPU-safe.
"""
@inline function faddeeva_w(z::Complex{T}) where {T<:Real}
    if imag(z) < zero(T)
        return 2 * exp(-z * z) - _faddeeva_upper(-z)
    end
    return _faddeeva_upper(z)
end

@inline function _faddeeva_upper(z::Complex{T}) where {T<:Real}
    L = T(_FADDEEVA_L)
    iz = im * z
    d = L - iz
    Z = (L + iz) / d
    # p = sum_{n=1}^{N} a_n Z^(n-1), Horner from the highest degree.
    p = Complex{T}(T(_FADDEEVA_A[_FADDEEVA_N]))
    @inbounds for n in (_FADDEEVA_N - 1):-1:1
        p = p * Z + T(_FADDEEVA_A[n])
    end
    return 2 * p / (d * d) + T(1 / sqrt(pi)) / d
end

"""
    gaussian_beam_field(x, y, sigmax, sigmay) -> (Ex, Ey)

Normalized transverse field of a bi-Gaussian charge distribution centred at the
origin, at transverse offset `(x, y)`. Normalization: for a round beam
`Ex = 2 x (1 - exp(-r^2/2sigma^2)) / r^2`, so the linear-regime focusing is
`Ex ≈ x / sigma^2`. This is the same normalization as JuTrack's
`Bassetti_Erskine!`: the kick on a particle of the weak beam is
`Δpx = N r0 q_w q_s / γ_w * Ex`, with `N` the number of strong-beam particles,
`r0` the classical radius of the weak-beam species, `q_w, q_s` the signed charges
and `γ_w` the weak-beam Lorentz factor. Elliptical beams use Bassetti-Erskine;
`sigmax == sigmay` (to 1e-10 relative) uses the closed round-beam form.
"""
@inline gaussian_beam_field(x::Real, y::Real, sigmax::Real, sigmay::Real) =
    gaussian_beam_field(promote(x, y, sigmax, sigmay)...)

@inline function gaussian_beam_field(x::T, y::T, sigmax::T, sigmay::T) where {T<:Real}
    sx2 = sigmax * sigmax
    sy2 = sigmay * sigmay
    if abs(sx2 - sy2) <= T(1e-10) * (sx2 + sy2)
        return _round_beam_field(x, y, (sx2 + sy2) / 2)
    elseif sx2 > sy2
        return _bassetti_erskine(x, y, sigmax, sigmay)
    else
        # x <-> y symmetry of the problem.
        Ey, Ex = _bassetti_erskine(y, x, sigmay, sigmax)
        return Ex, Ey
    end
end

@inline function _round_beam_field(x::T, y::T, sigma2::T) where {T<:Real}
    r2 = x * x + y * y
    if r2 <= T(1e-30) * sigma2
        # Linear limit 2 x (1 - e^{-u}) / r^2 -> x / sigma^2.
        return x / sigma2, y / sigma2
    end
    g = -2 * expm1(-r2 / (2 * sigma2)) / r2
    return g * x, g * y
end

"""
    _round_beam_field_series(x, y, sigma2) -> (Ex, Ey)

The round-beam field written as its everywhere-convergent power series in
`u = r^2 / (2 sigma^2)`: `2 (1 - e^{-u}) / r^2 = (1/sigma^2) Σ_k (-u)^k / (k+1)!`.
This form has no division by `r^2`, so it is the one used for series (TPSA)
coordinates, whose reference orbit is usually on axis. 41 terms give full
double precision for `u ≲ 12`, i.e. reference amplitudes below ~5 σ.
"""
# c[k+1] = (-1)^k / (k+1)!, k = 0..40
const _ROUND_SERIES_C = ntuple(i -> Float64((-1)^(i - 1) // factorial(big(i))), 41)

@inline function _round_beam_field_series(x, y, sigma2)
    u = (x * x + y * y) / (2 * sigma2)
    # Horner in u from the highest term.
    g = _ROUND_SERIES_C[end] * u
    @inbounds for k in (length(_ROUND_SERIES_C) - 1):-1:2
        g = (g + _ROUND_SERIES_C[k]) * u
    end
    g = (g + _ROUND_SERIES_C[1]) / sigma2
    return g * x, g * y
end

# Bassetti & Erskine, CERN-ISR-TH/80-06, for sigmax > sigmay. The formula holds
# for y >= 0; y < 0 follows from Ey(x,-y) = -Ey(x,y), Ex(x,-y) = Ex(x,y).
@inline function _bassetti_erskine(x::T, y::T, sigmax::T, sigmay::T) where {T<:Real}
    if y < zero(T)
        Ex, Ey = _bassetti_erskine(x, -y, sigmax, sigmay)
        return Ex, -Ey
    end
    sx2 = sigmax * sigmax
    sy2 = sigmay * sigmay
    s = sqrt(2 * (sx2 - sy2))
    z1 = Complex(x, y) / s
    z2 = Complex(x * sigmay / sigmax, y * sigmax / sigmay) / s
    expo = exp(-(x * x) / (2 * sx2) - (y * y) / (2 * sy2))
    F = faddeeva_w(z1) - expo * faddeeva_w(z2)
    # E_y + i E_x = C F  with C = 2 sqrt(pi) / s  (round limit: 2 (1-e^{-u}) / r)
    C = 2 * T(sqrt(pi)) / s
    return C * imag(F), C * real(F)
end

"""
    pass!(elem::StrongThinGaussianBeam, r, beti)

Thin strong-beam kick: `Δp = amplitude * gaussian_beam_field(x - xoffset, y - yoffset,
rmssizex, rmssizey)`. `amplitude = N r0 q_w q_s / γ_w` (see
[`beambeam_amplitude`](@ref)); positive for like charges, giving a defocusing
kick. `zloc` is descriptive metadata of the strong slice and does not enter the
thin kick; use [`StrongGaussianBeam`](@ref) for the synchro-beam mapping.
"""
function pass!(elem::StrongThinGaussianBeam{T,N}, r::SVector{6,S}, beti::T=one(T)) where {T,N,S}
    dx = r[1] - elem.xoffset
    dy = r[3] - elem.yoffset
    Ex, Ey = gaussian_beam_field(dx, dy, elem.rmssizex, elem.rmssizey)
    return SVector(r[1], r[2] + elem.amplitude * Ex, r[3], r[4] + elem.amplitude * Ey, r[5], r[6])
end

"""
    pass!(elem::StrongGaussianBeam, r, beti)

Synchro-beam mapping (Hirata, Moshammer & Ruggiero 1993) through the strong
beam's longitudinal slices. Slice `i`, holding `zslice_npar[i]` particles at
`zslice_center[i]` (positive toward the head of the strong bunch), collides
with a weak particle of coordinate `z` at `s* = (z - zslice_center[i]) / 2`
from the interaction point (TrackPad's `z` is positive for an early particle).
The transverse coordinates are drifted to `s*`, kicked with
`Δp = kick_scale * zslice_npar[i] * gaussian_beam_field(...)`, and drifted
back with the kicked momenta, which keeps the transverse map symplectic. The
beam size is taken constant along the slices (no hourglass), and the weak
particle's energy is not updated, as in JuTrack.
"""
function pass!(elem::StrongGaussianBeam{T,N,V}, r::SVector{6,S}, beti::T=one(T)) where {T,N,V,S}
    elem.nzslice <= 0 && return r
    x = r[1]; px = r[2]; y = r[3]; py = r[4]
    @inbounds for i in 1:elem.nzslice
        npar = i <= length(elem.zslice_npar) ? elem.zslice_npar[i] : T(elem.num_particle) / T(elem.nzslice)
        zc = i <= length(elem.zslice_center) ? elem.zslice_center[i] : zero(T)
        xoff = i <= length(elem.xoffsets) ? elem.xoffsets[i] : zero(T)
        yoff = i <= length(elem.yoffsets) ? elem.yoffsets[i] : zero(T)
        sstar = (r[5] - zc) / 2
        xc = x + px * sstar
        yc = y + py * sstar
        Ex, Ey = gaussian_beam_field(xc - xoff, yc - yoff, elem.beamsize[1], elem.beamsize[2])
        px = px + elem.kick_scale * npar * Ex
        py = py + elem.kick_scale * npar * Ey
        x = xc - px * sstar
        y = yc - py * sstar
    end
    return SVector(x, px, y, py, r[5], r[6])
end

# =============================================================================
# Longitudinal Wake models
# =============================================================================

# The RLC and tabulated wake functions are Green functions W(t) of the delay
# t = (z_test - z_source)/c. They vanish for t > 0: a particle only feels
# sources ahead of it. The longitudinal wake potential is the discrete
# convolution of the Green function with the bunch profile, obtained from a
# cloud-in-cell histogram of r[5] over all macroparticles:
#
#     V_k = sum_j N_j * W((z_k - z_j)/c),   z_k = bin-center coordinates
#
# As in JuTrack, V is averaged onto bin edges and every macroparticle receives
# the edge-interpolated potential at its own coordinate. TrackPad uses a
# translation-invariant padded range and linear cloud-in-cell deposition rather
# than JuTrack's zero-centered range and quadratic neighbor weights:
#
#     delta_i -= scale * V(z_i)
#
# The cloud-in-cell deposition and sub-bin interpolation remove the binning
# noise and staircase artifacts of a nearest-bin, piecewise-constant kick.
# Each macroparticle carries equal charge; `scale` absorbs q_macro/(P0*c) and
# any additional user normalization (`physical_wake_scale` computes the
# physically normalized value).

# Single-particle application (scalar CPU or TPSA) cannot represent a
# convolution over the bunch, so it is rejected outright.
function pass!(::Union{LongitudinalRLCWake, LongitudinalWake}, ::SVector{6},
               ::Any=one(Float64))
    throw(ArgumentError(
        "longitudinal wake elements are collective: the kick requires the " *
        "convolution of the wake Green function with the histogram of r[5] " *
        "over all macroparticles. Use `track!`/`linepass!` with an N x 6 " *
        "particle matrix instead of single-particle tracking."))
end

# Per-bin potential for one concrete element type (function barrier).
# `hist` holds cloud-in-cell weights (fractional counts) on a uniform grid with
# spacing `dz`.
function _wake_potential!(potential::AbstractVector{T}, hist::AbstractVector{T},
                          dz::T, elem::LongitudinalRLCWake{T}) where T
    fill!(potential, zero(T))
    nb = length(potential)
    dt = dz / T(C_LIGHT)
    @inbounds for k in 1:nb
        acc = zero(T)
        for j in k:nb
            hist[j] == 0 && continue
            acc += hist[j] * wakefieldfunc_RLCWake(elem, T(k - j) * dt)
        end
        potential[k] = acc
    end
    return nothing
end

function _wake_potential!(potential::AbstractVector{T}, hist::AbstractVector{T},
                          dz::T, elem::LongitudinalWake{T}) where T
    fill!(potential, zero(T))
    nb = length(potential)
    dt = dz / T(C_LIGHT)
    @inbounds for k in 1:nb
        acc = zero(T)
        for j in k:nb
            hist[j] == 0 && continue
            acc += hist[j] * wakefieldfunc(elem, T(k - j) * dt)
        end
        potential[k] = acc
    end
    return nothing
end

"""
Apply the collective longitudinal wake kick to all alive particles in
`coords`.

The bunch profile is deposited into `elem.nbins` uniform bins spanning the
alive-particle range padded by one nominal bin width on each side, using
cloud-in-cell (linear two-point) weights. The convolved potential is averaged
onto bin edges and each particle receives the linearly interpolated value at
its own coordinate; `delta -= elem.scale * V(z)`.
"""
function _apply_longitudinal_wake!(coords::Matrix{T}, lost_flags::AbstractVector{<:Integer},
                                   nparticles::Int, elem) where {T}
    iszero(elem.scale) && return nothing

    zmin = typemax(T)
    zmax = typemin(T)
    nalive = 0
    @inbounds for i in 1:nparticles
        lost_flags[i] == 1 && continue
        z = coords[i, 5]
        z < zmin && (zmin = z)
        z > zmax && (zmax = z)
        nalive += 1
    end
    nalive > 0 || return nothing  # no alive particles

    nb = elem.nbins

    if zmax > zmin
        span = zmax - zmin
        # Pad the grid by one nominal bin width on each side so cloud-in-cell
        # neighbor deposits and edge extrapolation stay inside the grid.
        dz_pad = span / nb
        z0 = zmin - dz_pad
        dz = (span + 2 * dz_pad) / nb

        hist = zeros(T, nb)
        if nb == 1
            hist[1] = T(nalive)
        else
            # Cloud-in-cell deposition about the nearest bin center: each
            # macro splits its unit weight linearly between adjacent centers.
            @inbounds for i in 1:nparticles
                lost_flags[i] == 1 && continue
                s = (coords[i, 5] - z0) / dz
                m = clamp(round(Int, s - T(0.5)) + 1, 1, nb)
                d = s - (T(m) - T(0.5))
                hist[m] += one(T) - abs(d)
                if d > zero(T)
                    hist[m+1] += d
                elseif d < zero(T)
                    hist[m-1] -= d
                end
            end
        end

        potential = zeros(T, nb)
        _wake_potential!(potential, hist, dz, elem)

        # Potential at bin edges: averages of adjacent bin-center values,
        # linearly extrapolated through both boundary edges.
        edges = Vector{T}(undef, nb + 1)
        if nb == 1
            edges[1] = edges[2] = potential[1]
        elseif nb == 2
            edges[1] = (T(3) * potential[1] - potential[2]) / T(2)
            edges[2] = (potential[1] + potential[2]) / T(2)
            edges[3] = (T(3) * potential[2] - potential[1]) / T(2)
        else
            @inbounds for k in 2:nb
                edges[k] = (potential[k - 1] + potential[k]) / T(2)
            end
            edges[1] = 2 * edges[2] - edges[3]
            edges[nb+1] = 2 * edges[nb] - edges[nb - 1]
        end

        @inbounds for i in 1:nparticles
            lost_flags[i] == 1 && continue
            s = (coords[i, 5] - z0) / dz
            b = clamp(floor(Int, s) + 1, 1, nb)
            w = edges[b] + (edges[b + 1] - edges[b]) *
                (coords[i, 5] - (z0 + (b - 1) * dz)) / dz
            coords[i, 6] -= elem.scale * w
        end
    else
        # Degenerate bunch (all alive particles share one z): every particle
        # feels N * W(0) exactly, independent of any grid choice.
        hist = T[nalive]
        potential = zeros(T, 1)
        _wake_potential!(potential, hist, one(T), elem)
        kick = elem.scale * potential[1]
        @inbounds for i in 1:nparticles
            lost_flags[i] == 1 && continue
            coords[i, 6] -= kick
        end
    end
    return nothing
end
