"""
    optics.jl

Linear optics utilities migrated from JuTrack concepts:
- one-turn map estimation
- tune extraction
- chromaticity (finite-difference)
- uncoupled 2D Twiss propagation along a lattice

These routines are numerical (finite-difference Jacobians) and intentionally
backend-agnostic for the current migration stage.
"""

using LinearAlgebra
using StaticArrays

export AbstractOptics, AbstractOptics2D, AbstractOptics4D, optics2D, optics4DUC
export TwissResult, transition_gamma
export transfer_map, one_turn_map, gettune, getchrom
export deltap_from_deltae, deltae_from_deltap
export twiss, periodic_twiss, transport_twiss, twissline
export find_closed_orbit_4d, find_closed_orbit_6d
export findm66, fastfindm66, findm66_refpts, fastfindm66_refpts
export periodicEdwardsTengTwiss, twissring, twissPropagate

"""Abstract supertype for TrackPad optics data."""
abstract type AbstractOptics end

"""Abstract supertype for one-plane optics data."""
abstract type AbstractOptics2D <: AbstractOptics end

"""Abstract supertype for transverse four-dimensional optics data."""
abstract type AbstractOptics4D <: AbstractOptics end

"""
    optics2D(beta, alpha, phase=0, eta=0, etap=0)

Courant-Snyder and dispersion data for one uncoupled transverse plane.
`gamma` is calculated as `(1 + alpha^2)/beta`.
"""
struct optics2D{T} <: AbstractOptics2D
    beta::T
    alpha::T
    gamma::T
    phase::T
    eta::T
    etap::T
end

function optics2D(beta::Real, alpha::Real, phase::Real, eta::Real, etap::Real)
    T = promote_type(typeof(beta), typeof(alpha), typeof(phase), typeof(eta), typeof(etap))
    b = T(beta)
    a = T(alpha)
    g = (one(T) + a^2) / b
    return optics2D{T}(b, a, g, T(phase), T(eta), T(etap))
end

optics2D(beta::Real, alpha::Real) = optics2D(beta, alpha, 0.0, 0.0, 0.0)

"""
    optics4DUC(bx, ax, by, ay)

Uncoupled transverse optics containing horizontal and vertical [`optics2D`](@ref)
values.
"""
struct optics4DUC{T} <: AbstractOptics4D
    optics_x::optics2D{T}
    optics_y::optics2D{T}
end

function optics4DUC(bx::Real, ax::Real, by::Real, ay::Real)
    T = promote_type(typeof(bx), typeof(ax), typeof(by), typeof(ay))
    ox = optics2D(T(bx), T(ax), zero(T), zero(T), zero(T))
    oy = optics2D(T(by), T(ay), zero(T), zero(T), zero(T))
    return optics4DUC(ox, oy)
end

"""
    TwissResult

Uncoupled optics along a lattice, returned by [`twiss`](@ref) (and its aliases
[`periodic_twiss`](@ref) and [`transport_twiss`](@ref)) at every boundary of
the sampled lattice (arrays have length `number of pieces + 1`; see the
`slices`, `max_step` and `sample_integrator_steps` keywords).

`periodic` records whether the result is the periodic solution of a ring
(evaluated about its closed orbit) or the propagation of supplied entrance
optics through an open line; `method` how the maps were obtained (`:tpsa`,
Taylor maps through PolySeries, the default, or `:fd`, finite differences of
tracking).

Linear optics: `s`, `betax`, `alphax`, `betay`, `alphay`, `mux`, `muy`
(accumulated phase, rad), `tunex`, `tuney` (the fractional tunes of a ring;
the total phase advance over 2π of a line) and `length` (the path length; the
circumference of a ring).

Dispersion `dx`, `dpx`, `dy`, `dpy` along `s`, as derivatives with respect to
the relative momentum `δP` (the `wrt` keyword selects the stored `δE`
instead). For a line it is propagated from the entrance dispersion.

Ring-only quantities are `nothing` for a line: the momentum compaction
`alphac = (1/C) dC/dδP`, the slip factor `slip = alphac − 1/γ0²`
([`transition_gamma`](@ref) derives γ_tr), the chromaticities `chromx`,
`chromy` `= dQ/dδP` about the off-momentum closed orbit, and — when requested —
`chrom2x`, `chrom2y` (the `δP²` Taylor coefficients of `Q(δP)`, i.e.
`½ d²Q/dδP²`; `second_order=true`) and `detuning`, the 2×2 amplitude-dependent
tune shift `∂Q_i/∂J_j` (`detuning=true`).

`radiation` holds the synchrotron-radiation integrals `(I1, I2, I3, I4, I5)`
over the bend bodies when `radiation_integrals=true` (ring or line), otherwise
`nothing`.
"""
struct TwissResult{T}
    periodic::Bool
    method::Symbol
    s::Vector{T}
    betax::Vector{T}
    alphax::Vector{T}
    betay::Vector{T}
    alphay::Vector{T}
    mux::Vector{T}
    muy::Vector{T}
    tunex::T
    tuney::T
    dx::Vector{T}
    dpx::Vector{T}
    dy::Vector{T}
    dpy::Vector{T}
    length::T
    alphac::Union{Nothing,T}
    slip::Union{Nothing,T}
    chromx::Union{Nothing,T}
    chromy::Union{Nothing,T}
    chrom2x::Union{Nothing,T}
    chrom2y::Union{Nothing,T}
    radiation::Union{Nothing,NamedTuple{(:I1, :I2, :I3, :I4, :I5),NTuple{5,T}}}
    detuning::Union{Nothing,SMatrix{2,2,T,4}}
end

"""
    transition_gamma(tw::TwissResult) -> γ_tr

`1/sqrt(alphac)` of a periodic result; `Inf` when the momentum compaction is
zero and `NaN` when it is negative (imaginary transition energy). Throws for
the optics of an open line, which has no momentum compaction.
"""
function transition_gamma(tw::TwissResult{T}) where T
    a = tw.alphac
    a === nothing && throw(ArgumentError("transition_gamma: an open line has no momentum compaction"))
    a > zero(T) && return one(T) / sqrt(a)
    return iszero(a) ? T(Inf) : T(NaN)
end

@inline _unit6(::Type{T}, i::Int) where T = SVector{6,T}(ntuple(j -> (j == i ? one(T) : zero(T)), 6))
@inline _as_lattice(lat::Lattice) = lat
@inline _as_lattice(seq::AbstractVector{<:AbstractElement}) = Lattice(seq)

@inline function _check_wrt(wrt::Symbol)
    wrt === :deltap || wrt === :deltae || throw(ArgumentError(
        "wrt must be :deltap (relative momentum (P-P0)/P0, the default) or " *
        ":deltae (TrackPad's stored sixth coordinate), got :$wrt"))
    return wrt
end

# Convert a momentum offset supplied at the interface into the stored sixth
# coordinate. `deltae_from_deltap` is defined below, next to `getchrom`.
@inline function _deltae_input(dp::Real, β0::Real, wrt::Symbol)
    _check_wrt(wrt)
    return wrt === :deltae ? dp : deltae_from_deltap(dp, β0)
end

function _reference6(::Type{T}, dp::Real, orb::AbstractVector, β0::Real,
                     wrt::Symbol=:deltap) where T
    if length(orb) != 6
        throw(ArgumentError("orb must have length 6"))
    end
    # `dp` is an interface momentum offset (δP unless `wrt=:deltae`); `orb` is
    # a coordinate vector already in the stored δE, so only `dp` is converted.
    dpeff = (dp == 0 && orb[6] != 0) ? T(orb[6]) : T(_deltae_input(dp, β0, wrt))
    return SVector{6,T}(T(orb[1]), T(orb[2]), T(orb[3]), T(orb[4]), T(orb[5]), dpeff)
end

function _beam_from_energy_mass(E0::Real, m0::Real)
    T = promote_type(typeof(E0), typeof(m0))
    return Beam(T(E0); mass = T(m0))
end

function _validate_refpts(refpts::AbstractVector{<:Integer}, n::Int)
    isempty(refpts) && return
    prev = 0
    for rp in refpts
        if rp < 1 || rp > n
            throw(ArgumentError("refpts entries must be in [1, $n]"))
        end
        if rp < prev
            throw(ArgumentError("refpts must be non-decreasing"))
        end
        prev = rp
    end
end

"""
    transfer_map(lat, beam; reference=zeros, h=1e-8)

Finite-difference estimate of the 6x6 map through an open line or periodic
ring, evaluated about `reference`.
"""
function transfer_map(lat::Lattice, beam::Beam{T};
                      reference::SVector{6,T}=zero(SVector{6,T}),
                      h::T=T(1e-8)) where T
    M = Matrix{T}(undef, 6, 6)
    for i in 1:6
        ei = _unit6(T, i)
        rp = linepass(lat, reference + h * ei, beam)
        rm = linepass(lat, reference - h * ei, beam)
        @inbounds M[:, i] = (rp - rm) / (2h)
    end
    return M
end

"""
    one_turn_map(lat, beam; reference=zeros, h=1e-8)

Finite-difference estimate of a periodic lattice's 6x6 one-turn Jacobian.
Use [`transfer_map`](@ref) for an open line.
"""
function one_turn_map(lat::Lattice, beam::Beam{T}; kwargs...) where T
    _require_periodic(lat, "one_turn_map")
    return transfer_map(lat, beam; kwargs...)
end

function _twiss_from_2x2(M::AbstractMatrix{T}) where T
    tr = (M[1,1] + M[2,2]) / 2
    if abs(tr) >= one(T)
        throw(DomainError(tr, "Unstable 2x2 map: |trace/2| >= 1"))
    end

    mu = acos(clamp(tr, -one(T), one(T)))
    sinmu = sin(mu)

    if abs(sinmu) < sqrt(eps(T))
        throw(DomainError(sinmu, "Degenerate map: sin(mu) too small"))
    end

    beta = M[1,2] / sinmu
    alpha = (M[1,1] - M[2,2]) / (2sinmu)

    if beta <= zero(T)
        mu = T(2π) - mu
        sinmu = sin(mu)
        beta = M[1,2] / sinmu
        alpha = (M[1,1] - M[2,2]) / (2sinmu)
    end

    tune = mod(mu / T(2π), one(T))
    return beta, alpha, mu, tune
end

@inline function _propagate_twiss(beta::T, alpha::T, A::AbstractMatrix{T}) where T
    gamma = (one(T) + alpha^2) / beta
    beta2 = A[1,1]^2 * beta - 2A[1,1]*A[1,2]*alpha + A[1,2]^2 * gamma
    alpha2 = -A[1,1]*A[2,1]*beta + (A[1,1]*A[2,2] + A[1,2]*A[2,1])*alpha - A[1,2]*A[2,2]*gamma

    dmu = atan(A[1,2], A[1,1]*beta - A[1,2]*alpha)
    if dmu < zero(T)
        dmu += T(2π)
    end

    return beta2, alpha2, dmu
end

"""
    twissPropagate(tin, M)

Propagate uncoupled optics through a 4x4 or 6x6 transfer map.
"""
function twissPropagate(tin::optics4DUC{T}, M::AbstractMatrix{T}) where T
    size(M, 1) >= 4 && size(M, 2) >= 4 || throw(ArgumentError("M must be at least 4x4"))
    bx2, ax2, dmx = _propagate_twiss(tin.optics_x.beta, tin.optics_x.alpha, @view M[1:2, 1:2])
    by2, ay2, dmy = _propagate_twiss(tin.optics_y.beta, tin.optics_y.alpha, @view M[3:4, 3:4])
    ox = optics2D(bx2, ax2, tin.optics_x.phase + dmx, tin.optics_x.eta, tin.optics_x.etap)
    oy = optics2D(by2, ay2, tin.optics_y.phase + dmy, tin.optics_y.eta, tin.optics_y.etap)
    return optics4DUC(ox, oy)
end

function twissPropagate(tin::optics4DUC, M::AbstractMatrix)
    T = promote_type(typeof(tin.optics_x.beta), eltype(M))
    tin_T = optics4DUC(
        optics2D(T(tin.optics_x.beta), T(tin.optics_x.alpha), T(tin.optics_x.phase), T(tin.optics_x.eta), T(tin.optics_x.etap)),
        optics2D(T(tin.optics_y.beta), T(tin.optics_y.alpha), T(tin.optics_y.phase), T(tin.optics_y.eta), T(tin.optics_y.etap)),
    )
    return twissPropagate(tin_T, Matrix{T}(M))
end

"""
    fastfindm66(lat, dp=0.0; E0=3e9, m0=M_ELECTRON, orb=zeros(6), h=3e-8,
                wrt=:deltap)

JuTrack-compatible finite-difference 6x6 map API. `h` is the full separation
between the positive and negative perturbations, matching JuTrack's `scaling`.
`E0` is the reference **total** energy in eV (JuTrack's `energy` is kinetic;
pass `E0 = K + m0` to match a JuTrack call).

`dp` is the relative momentum `δP = (P-P0)/P0` (the interface convention of
MAD-X, elegant and AT); pass `wrt=:deltae` to supply TrackPad's stored sixth
coordinate instead. `orb` is a coordinate vector and is always in the stored
`δE`. The returned matrix is the Jacobian in tracking coordinates, so its sixth
row and column refer to `δE` regardless of `wrt`.
"""
function fastfindm66(lat_in, dp::Real=0.0;
                     E0::Real=3.0e9,
                     m0::Real=M_ELECTRON,
                     orb::AbstractVector=zeros(6),
                     h::Real=3e-8,
                     wrt::Symbol=:deltap)
    lat = _as_lattice(lat_in)
    beam = _beam_from_energy_mass(E0, m0)
    T = typeof(beam.energy)
    ref = _reference6(T, dp, orb, beam.beta, wrt)
    return transfer_map(lat, beam; reference=ref, h=T(h) / 2)
end

"""
    findm66(lat, dp, order; E0=3e9, m0=M_ELECTRON, orb=zeros(6), h=3e-8)

JuTrack-compatible map API. `order > 0` (TPSA path) is deferred and falls back to finite-difference.
"""
function findm66(lat_in, dp::Real, order::Integer;
                 E0::Real=3.0e9,
                 m0::Real=M_ELECTRON,
                 orb::AbstractVector=zeros(6),
                 h::Real=3e-8,
                 wrt::Symbol=:deltap)
    if order != 0
        @warn "findm66: order > 0 TPSA map is deferred; using finite-difference fallback."
    end
    return fastfindm66(lat_in, dp; E0=E0, m0=m0, orb=orb, h=h, wrt=wrt)
end

"""
    fastfindm66_refpts(lat, dp, refpts; E0=3e9, m0=M_ELECTRON, orb=zeros(6), h=3e-8)

Return 6x6 segment maps at each reference point index. `h` is the full
positive-to-negative perturbation separation.
"""
function fastfindm66_refpts(lat_in, dp::Real, refpts::AbstractVector{<:Integer};
                            E0::Real=3.0e9,
                            m0::Real=M_ELECTRON,
                            orb::AbstractVector=zeros(6),
                            h::Real=3e-8,
                            wrt::Symbol=:deltap)
    lat = _as_lattice(lat_in)
    _validate_refpts(refpts, length(lat))
    beam = _beam_from_energy_mass(E0, m0)
    T = typeof(beam.energy)
    ref = _reference6(T, dp, orb, beam.beta, wrt)

    maps = zeros(T, 6, 6, length(refpts))
    prev = 0
    for (i, rp) in enumerate(refpts)
        seg = prev == 0 ? lat.elements[1:rp] : lat.elements[prev+1:rp]
        seg_lat = Lattice(seg)
        maps[:, :, i] = transfer_map(seg_lat, beam; reference=ref, h=T(h) / 2)
        prev = rp
    end
    return maps
end

"""
    findm66_refpts(lat, dp, order, refpts; E0=3e9, m0=M_ELECTRON, orb=zeros(6), h=3e-8)

JuTrack-compatible map-at-refpts API. `order > 0` TPSA path is deferred.
"""
function findm66_refpts(lat_in, dp::Real, order::Integer, refpts::AbstractVector{<:Integer};
                        E0::Real=3.0e9,
                        m0::Real=M_ELECTRON,
                        orb::AbstractVector=zeros(6),
                        h::Real=3e-8,
                        wrt::Symbol=:deltap)
    if order != 0
        @warn "findm66_refpts: order > 0 TPSA map is deferred; using finite-difference fallback."
    end
    return fastfindm66_refpts(lat_in, dp, refpts; E0=E0, m0=m0, orb=orb, h=h, wrt=wrt)
end

@inline function _tune_from_map(M::AbstractMatrix{T}) where T
    M44 = @view M[1:4, 1:4]
    cos_mu_x = (M44[1,1] + M44[2,2]) / 2
    cos_mu_y = (M44[3,3] + M44[4,4]) / 2
    sin_mu_x = sign(M44[1,2]) * sqrt(abs(-M44[1,2] * M44[2,1] - (M44[1,1] - M44[2,2])^2 / 4))
    sin_mu_y = sign(M44[3,4]) * sqrt(abs(-M44[3,4] * M44[4,3] - (M44[3,3] - M44[4,4])^2 / 4))
    qx = mod(atan(sin_mu_x, cos_mu_x) / T(2π), one(T))
    qy = mod(atan(sin_mu_y, cos_mu_y) / T(2π), one(T))
    return qx, qy
end

"""
    gettune(lat, beam; reference=zeros, h=3e-8)

Return `(Qx, Qy)` from the uncoupled blocks of the one-turn map.
"""
function gettune(lat::Lattice, beam::Beam{T};
                 reference::SVector{6,T}=zero(SVector{6,T}),
                 h::T=T(3e-8)) where T
    _require_periodic(lat, "gettune")
    M = findm66(lat, reference[6], 0; E0=beam.energy, m0=beam.mass, orb=collect(reference), h=h)
    return _tune_from_map(M)
end

function _unwrap_tune_delta(q1::T, q0::T) where T
    dq = q1 - q0
    if dq > 0.5
        dq -= one(T)
    elseif dq < -0.5
        dq += one(T)
    end
    return dq
end

"""
    deltap_from_deltae(δE, β0) -> δP

Exact conversion from TrackPad's stored sixth coordinate `δE = (E - E0)/(P0*c)`
to the relative momentum `δP = (P - P0)/P0`, from

    (1 + δP)^2 = 1 + 2δE/β0 + δE^2.

Written in rationalized form, so it avoids the cancellation of
`sqrt(1 + small) - 1` and stays accurate to full precision at
finite-difference-sized arguments (1e-8 and below). To first order
`δE = β0 δP`; the relation is *not* linear away from the reference energy.
See also [`deltae_from_deltap`](@ref).
"""
@inline function deltap_from_deltae(δE::Real, β0::Real)
    t = δE * (2 / β0 + δE)                      # (1 + δP)^2 - 1
    return t / (sqrt(one(t) + t) + one(t))
end

"""
    deltae_from_deltap(δP, β0) -> δE

Exact inverse of [`deltap_from_deltae`](@ref): the stored sixth coordinate for a
relative momentum offset `δP`, also in cancellation-free form. This is the
conversion applied to every `dp`/`dpp` argument of the optics functions before
tracking, so that the interface speaks `δP` while the state vector stores `δE`.
"""
@inline function deltae_from_deltap(δP::Real, β0::Real)
    t = δP * (2 + δP)                           # (1 + δP)^2 - 1
    bi = inv(β0)
    return t / (sqrt(bi * bi + t) + bi)
end

# dδE/dδP at the point whose stored coordinate is δE. This is the local β
# there (β0 at the reference energy), and is the factor that converts any
# first derivative with respect to δE into one with respect to δP.
@inline function _beta_at_deltae(δE::Real, β0::Real)
    δP = deltap_from_deltae(δE, β0)
    return (one(δP) + δP) / (inv(β0) + δE)
end

# ── Optics building blocks, dispatched on the map method ──────────────────────
# `:tpsa` (default) uses Taylor maps through PolySeries CTPS coordinates and is
# implemented in src/tpsa_polyseries.jl; `:fd` uses finite differences of
# tracking (implemented here).

@inline function _check_optics_method(method::Symbol)
    method === :fd || method === :tpsa || throw(ArgumentError(
        "method must be :tpsa (Taylor map through PolySeries, the default) or " *
        ":fd (finite differences of tracking), got :$method"))
    return method
end

# 4-D closed orbit at stored momentum δE, seeded by x0.
_closed_orbit_4d(lat::Lattice, beam::Beam{T}, δE::T, x0::SVector{4,T}, ::Val{:fd}) where T =
    find_closed_orbit_4d(lat, beam; dp=δE, wrt=:deltae, x0=x0)

# One-turn Jacobian about the coordinate vector `ref` (stored coordinates).
_one_turn_jacobian(lat::Lattice, beam::Beam{T}, ref::SVector{6,T}, h::T, ::Val{:fd}) where T =
    transfer_map(lat, beam; reference=ref, h=h / 2)

# Per-element Jacobians along the trajectory starting at `ref`.
_optics_jacobians(lat::Lattice, beam::Beam{T}, ref::SVector{6,T}, h::T, ::Val{:fd}) where T =
    _element_jacobians(lat, beam, ref, h)

# The `Val{:tpsa}` methods of these primitives, and `_getchrom_tpsa`, live in
# src/tpsa_polyseries.jl.

"""
Tunes of the ring at stored momentum `δE`, about the off-momentum closed orbit
seeded by `reference[1:4]` with `z = reference[5]`. Returns `(qx, qy), ref`.
"""
function _closed_orbit_tunes(lat::Lattice, beam::Beam{T}, δE::T,
                             reference::SVector{6,T}, h::T, method::Val) where T
    M, ref = _closed_orbit_jacobian(lat, beam, δE, reference, h, method)
    return _tune_from_map(M), ref
end

# Closed orbit at δE (seeded by `reference`, which also supplies z) together
# with the one-turn Jacobian about it. The Taylor-map version (tpsa_polyseries.jl)
# gets the Jacobian for free from the last Newton iteration.
function _closed_orbit_jacobian(lat::Lattice, beam::Beam{T}, δE::T,
                                reference::SVector{6,T}, h::T, method::Val{:fd}) where T
    co = _closed_orbit_4d(lat, beam, δE,
                          SVector{4,T}(reference[1], reference[2], reference[3], reference[4]),
                          method)
    ref = SVector{6,T}(co[1], co[2], co[3], co[4], reference[5], δE)
    return _one_turn_jacobian(lat, beam, ref, h, method), ref
end

# Seed for the closed orbit at a nearby momentum: the linear dispersion of the
# one-turn Jacobian `M` about `ref` extrapolated by ΔδE. Saves one Newton
# iteration (one Taylor-map turn) per off-momentum evaluation.
function _dispersion_seed(M::AbstractMatrix{T}, ref::SVector{6,T}, ΔδE::T) where T
    D = (I - @view(M[1:4, 1:4])) \ @view(M[1:4, 6])
    return SVector{6,T}(ref[1] + D[1] * ΔδE, ref[2] + D[2] * ΔδE,
                        ref[3] + D[3] * ΔδE, ref[4] + D[4] * ΔδE, ref[5], ref[6] + ΔδE)
end

# Local β and γ at stored momentum δE (β0, γ0 at the reference).
function _local_beta_gamma(beam::Beam{T}, δE::T) where T
    E = beam.energy + δE * p0c(beam)
    pc = sqrt(max(E * E - beam.mass * beam.mass, zero(T)))
    return pc / E, E / beam.mass
end

"""
    getchrom(lat, beam; dp=0, reference=zeros, method=:tpsa, wrt=:deltap,
             h=3e-8, dpp=1e-8, centered=false) -> (ξx, ξy)

Chromaticity of a periodic lattice, `ξ = dQ/dδP`, measured about the
off-momentum closed orbit at each momentum. This is the only meaningful
definition for a ring: on the closed orbit every sextupole sits at `x = D·δP`
and acts as a quadrupole of gradient `k2·D·δP`, which *is* the sextupole
chromaticity correction. A trajectory launched on-axis instead (JuTrack's
convention) samples that feed-down along a betatron-oscillating path and is off
by a few percent of the sextupole contribution; TrackPad does not offer it.
An open line has no closed orbit and no chromaticity; `getchrom` requires
`periodic=true`.

`wrt` selects the variable the tune is differentiated with respect to, and is
also the variable `dp` (evaluation point) and `dpp` (step) are expressed in:

  * `:deltap` (default) — `δP = (P-P0)/P0`, the definition MAD-X, elegant and
    AT use;
  * `:deltae` — TrackPad's stored sixth coordinate `δE = (E-E0)/(P0*c)`.

Offsets are converted exactly through `(1+δP)^2 = 1 + 2δE/β0 + δE^2`.

`method` selects how the tunes are obtained:

  * `:tpsa` (default) — the derivative is read off an order-2 Taylor map about
    the exact closed orbit: `ξ = ∂Q/∂δ + Σ_k D_k ∂Q/∂x_k` with no
    finite-difference step at all (agreement with PTC to 1e-6 relative on the
    reference ring of the test suite). `dpp`, `h` and `centered` are unused.
  * `:fd` — finite differences of tracking: a 4-D Newton closed-orbit solve
    and a finite-difference Jacobian (`h`) at `dp` and `dp + dpp`
    (`centered=true`: at `dp ± dpp`). The forward difference has error
    ≈ (truncation ∝ `dpp`) + (roundoff ≈ 2e-15/`dpp`) and `dpp = 1e-8` sits at
    that optimum (~7 correct digits); `centered=true` has error ≈ `dpp²` +
    1e-15/`dpp`, optimum near `dpp = 1e-6`. About 3× faster than `:tpsa` and
    the only method for elements without a series map (`LBend`).

`reference` is a tracking coordinate vector (its sixth entry is the stored
`δE`); its transverse part seeds the closed-orbit search and `reference[5]`
sets `z`. A lattice with no stable off-momentum closed orbit throws (pass
`strict=false` to [`find_closed_orbit_4d`](@ref) directly if you need the last
iterate).

See also [`periodic_twiss`](@ref), which returns the chromaticity together with
the dispersion and momentum compaction, and [`gettune`](@ref).
"""
function getchrom(lat::Lattice, beam::Beam{T};
                  dp::T=zero(T),
                  reference::SVector{6,T}=zero(SVector{6,T}),
                  h::T=T(3e-8),
                  dpp::T=T(1e-8),
                  centered::Bool=false,
                  wrt::Symbol=:deltap,
                  method::Symbol=:tpsa) where T
    _require_periodic(lat, "getchrom")
    dpp > zero(T) || throw(ArgumentError("dpp must be positive"))
    _check_wrt(wrt)
    _check_optics_method(method)
    if method === :tpsa
        return _getchrom_tpsa(lat, beam, Val(:tpsa); dp=dp, reference=reference, wrt=wrt)
    end

    # `dp` and `dpp` are offsets and steps in the variable being differentiated;
    # tracking always needs the stored coordinate δE.
    to_deltae(x::T) = wrt === :deltae ? x : T(deltae_from_deltap(x, beam.beta))

    # Chromaticity is a property of the ring: the tune is measured about the
    # off-momentum closed orbit at each momentum, never about a trajectory
    # launched on-axis (which samples the sextupole feed-down along a
    # betatron-oscillating path and is off by a few percent of it).
    tune_at(momentum::T) = _closed_orbit_tunes(lat, beam, momentum, reference, h, Val(:fd))[1]

    if centered
        qminus = tune_at(to_deltae(dp - dpp))
        qplus = tune_at(to_deltae(dp + dpp))
        return (
            _unwrap_tune_delta(qplus[1], qminus[1]) / (2dpp),
            _unwrap_tune_delta(qplus[2], qminus[2]) / (2dpp),
        )
    end

    q0 = tune_at(to_deltae(dp))
    q1 = tune_at(to_deltae(dp + dpp))
    return (
        _unwrap_tune_delta(q1[1], q0[1]) / dpp,
        _unwrap_tune_delta(q1[2], q0[2]) / dpp,
    )
end

function _numerical_jacobian(f::Function, x::Vector{T}; h::T=T(1e-6)) where T
    n = length(x)
    y0 = f(x)
    J = Matrix{T}(undef, n, n)
    for j in 1:n
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[j] += h
        x_minus[j] -= h
        y_plus = f(x_plus)
        y_minus = f(x_minus)
        @inbounds J[:, j] = (y_plus .- y_minus) ./ (2h)
    end
    return J, y0
end

"""
    _newton_closed_orbit(f, x0, tol, maxiter, h, reg, strict, label) -> Vector

Newton iteration on the fixed-point residual `f(x) - x` with finite-difference
Jacobians. Returns the converged iterate.

The final iterate is always re-tested: the loop updates `x` after evaluating
the residual, so the value returned when the iteration limit is reached had
never been checked. A non-converged result is an error rather than a silently
returned vector, because an unconverged "orbit" (a ring with no fixed point
returns whatever the last Newton step produced, often metres off) is
indistinguishable downstream from a real closed orbit.
"""
function _newton_closed_orbit(f::Function, x0::Vector{T}, tol::T, maxiter::Int,
                              h::T, reg::T, strict::Bool,
                              label::AbstractString) where T
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
    tol > zero(T) || throw(ArgumentError("tol must be positive"))
    x = copy(x0)
    eye = Matrix{T}(I, length(x), length(x))

    for _ in 1:maxiter
        J, x_out = _numerical_jacobian(f, x; h=h)
        Δ = x_out .- x
        residual = norm(Δ)
        if residual < tol
            return x
        end
        Δx = (eye - J + reg * eye) \ Δ
        all(isfinite, Δx) || break
        x .+= Δx
    end

    residual = norm(f(x) .- x)
    residual < tol && return x

    message = "$label did not converge: |x_out - x| = $residual after " *
              "$maxiter iterations (tol = $tol). The lattice may have no " *
              "closed orbit, or the search needs a different starting point, " *
              "a larger maxiter, or a looser tol. Pass strict=false to accept " *
              "the last iterate."
    strict && error(message)
    @warn message
    return x
end

"""
    find_closed_orbit_6d(lat, beam; x0=zeros, tol=1e-10, maxiter=20, h=1e-6,
                         reg=1e-12, strict=true)

Find a 6-D closed orbit using Newton iterations with finite-difference Jacobians.

Throws when the iteration does not reach `tol`; pass `strict=false` to warn and
return the last iterate instead.
"""
function find_closed_orbit_6d(lat::Lattice, beam::Beam{T};
                              x0::SVector{6,T}=zero(SVector{6,T}),
                              tol::T=T(1e-10),
                              maxiter::Int=20,
                              h::T=T(1e-6),
                              reg::T=T(1e-12),
                              strict::Bool=true) where T
    _require_periodic(lat, "find_closed_orbit_6d")

    f(θ::Vector{T}) = collect(linepass(lat, SVector{6,T}(θ...), beam))

    x = _newton_closed_orbit(f, collect(x0), tol, maxiter, h, reg, strict,
                             "find_closed_orbit_6d")
    return SVector{6,T}(x...)
end

"""
    find_closed_orbit_4d(lat, beam; dp=0, x0=zeros, tol=1e-10, maxiter=20,
                         h=1e-6, reg=1e-12, strict=true, wrt=:deltap)

Find a 4-D closed orbit `(x, px, y, py)` at the fixed momentum offset `dp`.
`dp` is the relative momentum `δP = (P-P0)/P0` by default, matching the
interface convention of MAD-X, elegant and AT; pass `wrt=:deltae` to give it in
TrackPad's stored sixth coordinate instead. The returned orbit is transverse
only, so it carries no convention.

Throws when the iteration does not reach `tol`; pass `strict=false` to warn and
return the last iterate instead.
"""
function find_closed_orbit_4d(lat::Lattice, beam::Beam{T};
                              dp::T=zero(T),
                              x0::SVector{4,T}=zero(SVector{4,T}),
                              tol::T=T(1e-10),
                              maxiter::Int=20,
                              h::T=T(1e-6),
                              reg::T=T(1e-12),
                              strict::Bool=true,
                              wrt::Symbol=:deltap) where T
    _require_periodic(lat, "find_closed_orbit_4d")
    δE = T(_deltae_input(dp, beam.beta, wrt))

    function f(θ::Vector{T})
        rin = SVector{6,T}(θ[1], θ[2], θ[3], θ[4], zero(T), δE)
        rout = linepass(lat, rin, beam)
        return T[rout[1], rout[2], rout[3], rout[4]]
    end

    x = _newton_closed_orbit(f, collect(x0), tol, maxiter, h, reg, strict,
                             "find_closed_orbit_4d")
    return SVector{4,T}(x...)
end

function _element_jacobian(elem::AbstractElement, r0::SVector{6,T}, β_inv::T, h::T) where T
    J = Matrix{T}(undef, 6, 6)
    for i in 1:6
        ei = _unit6(T, i)
        rp = pass!(elem, r0 + h * ei, β_inv)
        rm = pass!(elem, r0 - h * ei, β_inv)
        @inbounds J[:, i] = (rp - rm) / (2h)
    end
    return J
end

function _element_jacobians(lat::Lattice, beam::Beam{T},
                            reference::SVector{6,T}, h::T) where T
    n = length(lat)
    β_inv = beti(beam)
    s = zeros(T, n + 1)
    jacobians = Vector{Matrix{T}}(undef, n)
    r = reference
    # Optics are evaluated at the same (time, turn) = (0, 0) that `linepass`
    # uses by default; time-varying elements are resolved here so that a
    # lattice containing `timed(...)` elements works in the Twiss/dispersion
    # functions exactly as it does in tracking.
    ctx = TimeContext(zero(T))
    for (i, elem_raw) in enumerate(lat.elements)
        elem = _resolve_for_time(elem_raw, ctx)
        s[i + 1] = s[i] + T(get_length(elem))
        jacobians[i] = _element_jacobian(elem, r, β_inv, h)
        r = pass!(elem, r, β_inv)
    end
    return s, jacobians, r
end

"""
    twiss(lat, beam; entrance=nothing, reference=zeros, method=:tpsa, wrt=:deltap,
          slices=1, max_step=nothing, sample_integrator_steps=false,
          second_order=false, radiation_integrals=false, detuning=false,
          detuning_actions=(1e-8, 2e-8, 3e-8, 4e-8), detuning_turns=1024,
          h=1e-8, dpp=1e-6, dpp2=1e-4) -> TwissResult

Uncoupled optics of a lattice as a [`TwissResult`](@ref).

Without `entrance` the lattice must be periodic (`Lattice(...; periodic=true)`)
and the result is the periodic solution of the ring, evaluated about its closed
orbit: Twiss functions, dispersion, momentum compaction and slip factor,
chromaticity, and optionally the second-order chromaticity, the radiation
integrals and the amplitude-dependent tune shift. Every momentum derivative is
taken about the off-momentum closed orbit, so sextupole feed-down is included.

With `entrance` the supplied optics are propagated once through `lat` (open
line or a single pass through a ring) from the coordinate `reference`.
`entrance` is either an [`optics4DUC`](@ref) — `beta`, `alpha`, `phase` of
both planes and the horizontal dispersion `eta`, `etap` (in the `wrt`
convention) — or another `TwissResult`, whose exit values (β, α, phase,
dispersion) become the entrance, so that consecutive lines chain. The
ring-only fields of the result are `nothing`, and `second_order` and
`detuning` are errors.

Sampling — the arrays are returned at every boundary of the sampled lattice:

- `slices=n`: split every element of finite length (drifts, thick multipoles,
  bends) into at least `n` pieces. A cheap way to get smooth curves inside
  thick elements for plotting without touching the lattice.
- `max_step=Δs`: split them into pieces no longer than `Δs` (uniform sampling
  in `s`).
- `sample_integrator_steps=true`: one piece per configured integration step.

The three combine (the largest count wins), and none of them replaces a
configured integrator by fewer, larger steps: a bend with `num_int_steps=10`
split into 4 slices still integrates in 10 steps. Entrance pole-face, fringe,
offset and rotation maps sit on the first piece, their exit counterparts on
the last ([`refine_lattice`](@ref) exposes the sampled lattice itself).

Other keywords:

- `reference`: 6-vector in the stored coordinates. For a ring its transverse
  part seeds the closed-orbit search and its sixth component is the momentum
  (stored `δE`) at which the optics are evaluated; for a line it is the launch
  coordinate.
- `method`: `:tpsa` (default; Taylor maps through PolySeries) or `:fd`
  (finite differences of tracking). `:tpsa` has no step parameters and no
  noise floor — the closed orbit, Jacobians, chromaticities and detuning are
  exact derivatives of the truncated maps; `:fd` reproduces it to the
  finite-difference truncation (tunes 1e-12, ξ ~1e-6 absolute, ξ₂ ~1e-3) and
  is 1.5–4× faster for the linear and chromatic quantities but 5–20× slower
  for the detuning, which it obtains by tracking. `:fd` is the only method
  for lattices with elements that have no series map (`LBend`). Series orders
  are the minimum each quantity needs: 1 for the closed orbit and Jacobians, 2
  for the chromaticity, 3 for the detuning.
- `wrt`: `:deltap` (default) reports every momentum derivative — dispersion,
  chromaticities, `alphac`, `slip` — with respect to `δP = (P − P0)/P0`;
  `:deltae` reports them with respect to the stored `δE = (E − E0)/(P0 c)`.
  An `optics4DUC` entrance dispersion is read in the same convention.
- `second_order=true`: also fill `chrom2x`, `chrom2y`, the `δP²` Taylor
  coefficients of the tune, i.e. `½ d²Q/dδP²` (three-point stencil, `dpp2`).
- `radiation_integrals=true`: also fill `radiation = (I1, I2, I3, I4, I5)`,
  integrated over the bend bodies with the dispersion of the result. Each
  bend is sliced internally (64 slices or its integration-step count) and
  integrated with composite Simpson, independently of the output sampling;
  `I4` omits the pole-face contribution of dipole edges.
- `detuning=true`: also fill `detuning`, the 2×2 matrix `∂Q_i/∂J_j` of the
  amplitude-dependent tune shift. With `method=:fd` it is obtained from
  `detuning_turns` turns of tracking at the actions `detuning_actions`
  (m·rad) and a NAFF tune estimate; with `method=:tpsa` from the order-3
  one-turn map.
- `h`: finite-difference step for the Jacobians (`method=:fd` only);
  `dpp`: momentum step for the first-order chromaticity (`method=:fd` only);
  `dpp2`: step of the second-order chromaticity stencil (both methods). Steps
  are in the `wrt` variable.

[`periodic_twiss`](@ref) and [`transport_twiss`](@ref) are the same function
with the two use cases spelled out.

See also [`getchrom`](@ref), [`transition_gamma`](@ref).
"""
function twiss(lat::Lattice, beam::Beam{T};
               entrance::Union{Nothing,optics4DUC,TwissResult}=nothing,
               reference::SVector{6,T}=zero(SVector{6,T}),
               h::T=T(1e-8),
               slices::Int=1,
               max_step::Union{Nothing,Real}=nothing,
               sample_integrator_steps::Bool=false,
               dpp::T=T(1e-6),
               dpp2::T=T(1e-4),
               second_order::Bool=false,
               radiation_integrals::Bool=false,
               detuning::Bool=false,
               detuning_actions=(1.0e-8, 2.0e-8, 3.0e-8, 4.0e-8),
               detuning_turns::Int=1024,
               wrt::Symbol=:deltap,
               method::Symbol=:tpsa) where T
    periodic = entrance === nothing
    if periodic
        lat.periodic || throw(ArgumentError(
            "twiss: the lattice is an open line; pass `entrance` (an optics4DUC or a " *
            "TwissResult) to propagate optics through it, or construct it with " *
            "`periodic=true` for the periodic solution"))
    else
        second_order && throw(ArgumentError("twiss: second_order chromaticity needs a periodic lattice, not an entrance"))
        detuning && throw(ArgumentError("twiss: amplitude detuning needs a periodic lattice, not an entrance"))
    end
    _check_wrt(wrt)
    _check_optics_method(method)
    slices >= 1 || throw(ArgumentError("slices must be at least 1"))
    mth = Val(method)
    β0 = beam.beta
    δE = reference[6]
    scale = wrt === :deltap ? T(_beta_at_deltae(δE, β0)) : one(T)

    sampled_lat = if sample_integrator_steps || max_step !== nothing || slices > 1
        refine_lattice(lat; sample_integrator_steps, max_step, slices)
    else
        lat
    end
    n = length(sampled_lat)

    ref = SVector{6,T}(reference[1], reference[2], reference[3], reference[4], reference[5], δE)
    if periodic
        # Everything is evaluated about the closed orbit at this momentum; the
        # transverse part of `reference` only seeds the search. The per-element
        # Jacobian pass also returns the orbit after one turn, so when `reference`
        # already closes (a ring without orbit errors, or an orbit found earlier)
        # no separate Newton solve is run; the closed-orbit solve is on `lat`, so
        # the shortcut applies only when the optics are not resampled.
        s, Jlist, rend = _optics_jacobians(sampled_lat, beam, ref, h, mth)
        if sampled_lat !== lat || maximum(abs(rend[i] - ref[i]) for i in 1:4) >= T(1e-13)
            _, ref = _closed_orbit_jacobian(lat, beam, δE, reference, h, mth)
            s, Jlist, _ = _optics_jacobians(sampled_lat, beam, ref, h, mth)
        end
    else
        s, Jlist, _ = _optics_jacobians(sampled_lat, beam, ref, h, mth)
    end

    M = Matrix{T}(I, 6, 6)
    for J in Jlist
        M = J * M
    end

    # ── entrance values ──
    DE = Matrix{T}(undef, 4, n + 1)          # dispersion as the stored-δE response
    betax = zeros(T, n + 1); alphax = zeros(T, n + 1)
    betay = zeros(T, n + 1); alphay = zeros(T, n + 1)
    mux = zeros(T, n + 1); muy = zeros(T, n + 1)
    if periodic
        betax[1], alphax[1], _, _ = _twiss_from_2x2(@view M[1:2, 1:2])
        betay[1], alphay[1], _, _ = _twiss_from_2x2(@view M[3:4, 3:4])
        DE[:, 1] .= (I - @view(M[1:4, 1:4])) \ @view(M[1:4, 6])
    elseif entrance isa optics4DUC
        betax[1] = T(entrance.optics_x.beta); alphax[1] = T(entrance.optics_x.alpha)
        betay[1] = T(entrance.optics_y.beta); alphay[1] = T(entrance.optics_y.alpha)
        mux[1] = T(entrance.optics_x.phase);  muy[1] = T(entrance.optics_y.phase)
        DE[:, 1] .= (T(entrance.optics_x.eta) / scale, T(entrance.optics_x.etap) / scale,
                     T(entrance.optics_y.eta) / scale, T(entrance.optics_y.etap) / scale)
    else
        e = entrance::TwissResult
        betax[1] = T(e.betax[end]); alphax[1] = T(e.alphax[end])
        betay[1] = T(e.betay[end]); alphay[1] = T(e.alphay[end])
        mux[1] = T(e.mux[end]);     muy[1] = T(e.muy[end])
        DE[:, 1] .= (T(e.dx[end]) / scale, T(e.dpx[end]) / scale,
                     T(e.dy[end]) / scale, T(e.dpy[end]) / scale)
    end

    # ── propagation ──
    for i in 1:n
        J = Jlist[i]
        betax[i+1], alphax[i+1], dmx = _propagate_twiss(betax[i], alphax[i], @view J[1:2, 1:2])
        betay[i+1], alphay[i+1], dmy = _propagate_twiss(betay[i], alphay[i], @view J[3:4, 3:4])
        mux[i+1] = mux[i] + dmx
        muy[i+1] = muy[i] + dmy
        DE[:, i + 1] .= @view(J[1:4, 1:4]) * @view(DE[:, i]) .+ @view(J[1:4, 6])
    end
    dx  = DE[1, :] .* scale; dpx = DE[2, :] .* scale
    dy  = DE[3, :] .* scale; dpy = DE[4, :] .* scale
    C0 = s[end]

    rad = radiation_integrals ?
        _radiation_integrals(sampled_lat, beam, ref, h, mth, betax, alphax, DE, scale) : nothing

    if !periodic
        return TwissResult{T}(false, method, s, betax, alphax, betay, alphay, mux, muy,
                              (mux[end] - mux[1]) / T(2π), (muy[end] - muy[1]) / T(2π),
                              dx, dpx, dy, dpy, C0,
                              nothing, nothing, nothing, nothing, nothing, nothing, rad, nothing)
    end

    qx, qy = _tune_from_map(M)

    # ── momentum compaction ──
    # Along the off-momentum closed orbit Δz per turn = M56 δE + Σ M5k D_k δE,
    # and with z = s/β0 − ct this equals −(C/β) η δP, δP = δE/β. Hence
    # η = −(β²/C) dΔz/dδE and αc = η + 1/γ².  (Verified against the path length
    # of the closed orbit and ∮D/ρ ds on a β0 = 0.875 ring.)
    βl, γl = _local_beta_gamma(beam, δE)
    dz_dδE = M[5, 6] + dot(@view(M[5, 1:4]), @view(DE[:, 1]))
    slip = -(βl * βl / C0) * dz_dδE
    alphac = slip + one(T) / (γl * γl)

    # ── chromaticity ──
    dp_wrt = wrt === :deltae ? δE : T(deltap_from_deltae(δE, β0))
    chromx, chromy = getchrom(lat, beam; dp=dp_wrt, reference=ref, h=h, dpp=dpp,
                              centered=true, wrt=wrt, method=method)
    # (the on-momentum tune must come from `lat` itself, like the ±dpp2 ones;
    # a resampled lattice integrates slightly differently)
    chrom2x, chrom2y = if second_order
        _second_order_chromaticity(lat, beam, ref, h, dpp2, wrt, mth;
                                   M0=(sampled_lat === lat ? M : nothing))
    else
        nothing, nothing
    end

    det = detuning ? _amplitude_detuning(lat, beam, ref, betax[1], alphax[1], betay[1], alphay[1],
                                         qx, qy, detuning_actions, detuning_turns, mth) : nothing

    return TwissResult{T}(true, method, s, betax, alphax, betay, alphay, mux, muy, qx, qy,
                          dx, dpx, dy, dpy, C0, alphac, slip, chromx, chromy,
                          chrom2x, chrom2y, rad, det)
end

"""
    periodic_twiss(lat, beam; kwargs...) -> TwissResult

The periodic optics of a ring: [`twiss`](@ref) without an `entrance`. The
lattice must be periodic.
"""
function periodic_twiss(lat::Lattice, beam::Beam; kwargs...)
    _require_periodic(lat, "periodic_twiss")
    return twiss(lat, beam; kwargs...)
end

"""
    transport_twiss(lat, beam, entrance; kwargs...) -> TwissResult

Optics propagated from `entrance` (an [`optics4DUC`](@ref) or a
[`TwissResult`](@ref) to continue from) through `lat`: [`twiss`](@ref) with
that `entrance`. Valid for an open line and for a single pass through a ring.
"""
transport_twiss(lat::Lattice, beam::Beam, entrance::Union{optics4DUC,TwissResult}; kwargs...) =
    twiss(lat, beam; entrance=entrance, kwargs...)

"""
    twissline(lat, beam; kwargs...)

Historic name of [`periodic_twiss`](@ref).
"""
function twissline(lat::Lattice, beam::Beam; kwargs...)
    _require_periodic(lat, "twissline(lat, beam)")
    return twiss(lat, beam; kwargs...)
end

# Second-order chromaticity as the δP² Taylor coefficient of Q(δP): a centered
# three-point second difference of the closed-orbit tunes. `dpp2` is a step in
# the `wrt` variable; 1e-4 balances the ~2e-15 tune noise against truncation.
# `M0`, when given, is the one-turn Jacobian about `ref` (already the closed
# orbit at ref[6]); otherwise both are computed here. The ±dpp2 closed orbits
# are seeded from its dispersion.
function _second_order_chromaticity(lat::Lattice, beam::Beam{T}, ref::SVector{6,T},
                                    h::T, dpp2::T, wrt::Symbol, mth::Val;
                                    M0::Union{Nothing,AbstractMatrix{T}}=nothing) where T
    β0 = beam.beta
    δE0 = ref[6]
    x0 = wrt === :deltae ? δE0 : T(deltap_from_deltae(δE0, β0))
    to_δE(x) = wrt === :deltae ? x : T(deltae_from_deltap(x, β0))
    if M0 === nothing
        M0, ref = _closed_orbit_jacobian(lat, beam, δE0, ref, h, mth)
    end
    q0 = _tune_from_map(M0)
    δEp = to_δE(x0 + dpp2); δEm = to_δE(x0 - dpp2)
    qp, _ = _closed_orbit_tunes(lat, beam, δEp, _dispersion_seed(M0, ref, δEp - δE0), h, mth)
    qm, _ = _closed_orbit_tunes(lat, beam, δEm, _dispersion_seed(M0, ref, δEm - δE0), h, mth)
    d2(i) = (_unwrap_tune_delta(qp[i], q0[i]) + _unwrap_tune_delta(qm[i], q0[i])) / (dpp2 * dpp2)
    return d2(1) / 2, d2(2) / 2
end

# Curvature 1/ρ and body gradient k1 of a bend piece; `nothing` for anything else.
_bend_curvature(el::Union{SBend,ExactSBend,SBendSC}) = (el.angle / el.L, el.polynom_b[2])
_bend_curvature(el::LBend) = (el.angle / el.L, el.K)
_bend_curvature(::AbstractElement) = nothing

# Synchrotron-radiation integrals over the bend bodies. Each bend is sliced
# internally into `_RAD_INT_SLICES` (or its integration-step count, if larger)
# pieces, the closed-orbit Jacobians of the pieces propagate β, α and the
# stored-δE dispersion from the bend entrance, and D/ρ, D(1/ρ² + 2k1)/ρ and
# H/|ρ|³ are integrated with composite Simpson. The end nodes are quadratic
# (cubic) extrapolations of the interior nodes, so that the entrance/exit pole-face maps
# (which sit on the first/last piece) do not contaminate the body integrand;
# their own contribution to I4 is not included. Sampling only at element
# boundaries, as before, overestimated I1, I4 and I5 by (hL)²/12·D''/D ≈ 4 %
# on a 7.5° bend.
const _RAD_INT_SLICES = 64

@inline function _simpson_weights(::Type{T}, m::Int) where T
    w = fill(T(2), m + 1)
    w[2:2:m] .= T(4)
    w[1] = w[end] = one(T)
    return w ./ T(3)
end

# Cubic extrapolation of equally spaced samples to the node before f1 (O(h⁴),
# matching Simpson; a quadratic one dominated the error at 1e-6).
@inline _extrap_left(f1, f2, f3, f4) = 4 * f1 - 6 * f2 + 4 * f3 - f4   # (not `4f1`: a Float32 literal)

function _radiation_integrals(lat::Lattice, beam::Beam{T}, ref::SVector{6,T}, h::T, mth::Val,
                              betax::Vector{T}, alphax::Vector{T},
                              DE::AbstractMatrix{T}, scale::T) where T
    I1 = I2 = I3 = I4 = I5 = zero(T)
    β_inv = beti(beam)
    ctx = TimeContext(zero(T))
    r = ref
    for (i, elem_raw) in enumerate(lat.elements)
        elem = _resolve_for_time(elem_raw, ctx)
        c = _bend_curvature(elem)
        if c !== nothing
            irho, k1 = c
            L = T(get_length(elem))
            if !iszero(irho) && !iszero(L)
                m = max(_RAD_INT_SLICES, _has_tracking_steps(elem) ? getfield(elem, :num_int_steps) : 0)
                isodd(m) && (m += 1)
                pieces = _slice_for_optics(elem, m)
                _, Jsub, _ = _optics_jacobians(Lattice(pieces; periodic=false), beam, r, h, mth)
                βs = Vector{T}(undef, m + 1); αs = similar(βs); Ds = similar(βs); Dps = similar(βs)
                βs[1] = betax[i]; αs[1] = alphax[i]; Ds[1] = DE[1, i]; Dps[1] = DE[2, i]
                for j in 1:m
                    J = Jsub[j]
                    βs[j+1], αs[j+1], _ = _propagate_twiss(βs[j], αs[j], @view J[1:2, 1:2])
                    Ds[j+1]  = J[1, 1] * Ds[j] + J[1, 2] * Dps[j] + J[1, 6]
                    Dps[j+1] = J[2, 1] * Ds[j] + J[2, 2] * Dps[j] + J[2, 6]
                end
                fD = Vector{T}(undef, m + 1); fH = similar(fD)
                for j in 1:m+1
                    D = Ds[j] * scale; Dp = Dps[j] * scale
                    γ = (one(T) + αs[j]^2) / βs[j]
                    fD[j] = D
                    fH[j] = γ * D * D + 2αs[j] * D * Dp + βs[j] * Dp * Dp
                end
                # Replace the nodes that sit outside the pole-face maps.
                fD[1] = _extrap_left(fD[2], fD[3], fD[4], fD[5]);   fD[end] = _extrap_left(fD[end-1], fD[end-2], fD[end-3], fD[end-4])
                fH[1] = _extrap_left(fH[2], fH[3], fH[4], fH[5]);   fH[end] = _extrap_left(fH[end-1], fH[end-2], fH[end-3], fH[end-4])
                w = _simpson_weights(T, m) .* (L / m)
                ∫D = dot(w, fD); ∫H = dot(w, fH)
                I1 += ∫D * irho
                I2 += irho^2 * L
                I3 += abs(irho)^3 * L
                I4 += ∫D * irho * (irho^2 + 2k1)
                I5 += ∫H * abs(irho)^3
            end
        end
        r = pass!(elem, r, β_inv)
    end
    return (I1=I1, I2=I2, I3=I3, I4=I4, I5=I5)
end

# Amplitude-dependent tune shift; implemented below (tracking) and in the
# PolySeries extension (Taylor-map iteration).
# ── Amplitude-dependent tune shift ──────────────────────────────────────────
# Both methods launch particles at a few actions about the closed orbit, take
# the turn-by-turn tune of the normalized coordinate with a NAFF-style
# estimator, and fit Q linearly in J. `:fd` tracks; `:tpsa` (tpsa_polyseries.jl)
# iterates the order-3 Taylor map. `_detuning_from_turns` is the shared part.

"""
Fundamental frequency in turns⁻¹, in ``[0, 1)``, of a complex turn-by-turn
signal: Hann-windowed DFT maximum on the grid, refined by golden-section search
(Laskar's NAFF for the leading line). Resolution ≈ 1e-4/N² … 1e-9 for N = 1024.
"""
function _naff_tune(z::AbstractVector{<:Complex})
    N = length(z)
    N >= 8 || throw(ArgumentError("need at least 8 turns"))
    zw = [z[n] * (0.5 - 0.5cospi(2(n - 1) / (N - 1))) for n in 1:N]
    amp(ν) = abs(sum(zw[n] * cispi(-2ν * (n - 1)) for n in 1:N))
    best, bestν = -1.0, 0.0
    for k in 0:N-1
        a = amp(k / N)
        a > best && (best = a; bestν = k / N)
    end
    lo, hi = bestν - 1 / N, bestν + 1 / N
    φ = (sqrt(5.0) - 1) / 2
    c = hi - φ * (hi - lo); d = lo + φ * (hi - lo)
    fc, fd = amp(c), amp(d)
    for _ in 1:60
        if fc > fd
            hi, d, fd = d, c, fc
            c = hi - φ * (hi - lo); fc = amp(c)
        else
            lo, c, fc = c, d, fd
            d = lo + φ * (hi - lo); fd = amp(d)
        end
    end
    return mod((lo + hi) / 2, 1.0)
end

# Choose between ν and 1−ν the branch closest to the linear tune q.
@inline _tune_branch(ν, q) = abs(ν - q) <= abs((1 - ν) - q) ? ν : 1 - ν

"""
    _detuning_from_turns(turns!, ref, βx, αx, βy, αy, qx, qy, actions, nturns)

`turns!(buf, r0)` must fill `buf::Matrix` (nturns × 4) with `x, px, y, py`
after each turn starting from the 6-vector `r0`. Returns the 2×2 matrix
`[∂Qx/∂Jx ∂Qx/∂Jy; ∂Qy/∂Jx ∂Qy/∂Jy]` from linear fits of the tunes against the
action, scanning one plane at a time with the other held at the smallest
action.
"""
function _detuning_from_turns(turns!, ref::SVector{6,T}, βx::T, αx::T, βy::T, αy::T,
                              qx::T, qy::T, actions, nturns::Int) where T
    J = collect(T, actions)
    length(J) >= 2 || throw(ArgumentError("detuning needs at least two actions"))
    buf = Matrix{T}(undef, nturns, 4)
    launch(Jx, Jy) = SVector{6,T}(ref[1] + sqrt(2Jx * βx), ref[2] - αx * sqrt(2Jx / βx),
                                  ref[3] + sqrt(2Jy * βy), ref[4] - αy * sqrt(2Jy / βy),
                                  ref[5], ref[6])
    function tunes(r0)
        turns!(buf, r0)
        ux = [complex((buf[n, 1] - ref[1]) / sqrt(βx),
                      αx * (buf[n, 1] - ref[1]) / sqrt(βx) + sqrt(βx) * (buf[n, 2] - ref[2])) for n in 1:nturns]
        uy = [complex((buf[n, 3] - ref[3]) / sqrt(βy),
                      αy * (buf[n, 3] - ref[3]) / sqrt(βy) + sqrt(βy) * (buf[n, 4] - ref[4])) for n in 1:nturns]
        return _tune_branch(_naff_tune(ux), qx), _tune_branch(_naff_tune(uy), qy)
    end
    slope(x, y) = (n = length(x); (n * dot(x, y) - sum(x) * sum(y)) / (n * dot(x, x) - sum(x)^2))
    Jmin = minimum(J)
    tx = [tunes(launch(j, Jmin)) for j in J]      # x scan
    ty = [tunes(launch(Jmin, j)) for j in J]      # y scan
    return SMatrix{2,2,T,4}(slope(J, first.(tx)), slope(J, last.(tx)),
                            slope(J, first.(ty)), slope(J, last.(ty)))
end

function _amplitude_detuning(lat::Lattice, beam::Beam{T}, ref::SVector{6,T},
                             βx::T, αx::T, βy::T, αy::T, qx::T, qy::T,
                             actions, nturns::Int, ::Val{:fd}) where T
    function turns!(buf, r0)
        r = r0
        @inbounds for n in 1:size(buf, 1)
            r = linepass(lat, r, beam)
            check_lost(r) && throw(ArgumentError("detuning: particle lost during tracking; reduce `detuning_actions`"))
            buf[n, 1] = r[1]; buf[n, 2] = r[2]; buf[n, 3] = r[3]; buf[n, 4] = r[4]
        end
    end
    return _detuning_from_turns(turns!, ref, βx, αx, βy, αy, qx, qy, actions, nturns)
end

"""
    periodicEdwardsTengTwiss(seq_or_lat, dp, order; E0=3e9, m0=M_ELECTRON, orb=zeros(6), h=3e-8)

JuTrack-style periodic optics interface (uncoupled 4D projection).
"""
function periodicEdwardsTengTwiss(lat_in, dp::Real, order::Integer;
                                  E0::Real=3.0e9,
                                  m0::Real=M_ELECTRON,
                                  orb::AbstractVector=zeros(6),
                                  h::Real=3e-8,
                                  wrt::Symbol=:deltap)
    lat_in isa Lattice && _require_periodic(lat_in, "periodicEdwardsTengTwiss")
    M = findm66(lat_in, dp, order; E0=E0, m0=m0, orb=orb, h=h, wrt=wrt)
    bx, ax, _, _ = _twiss_from_2x2(@view M[1:2, 1:2])
    by, ay, _, _ = _twiss_from_2x2(@view M[3:4, 3:4])
    return optics4DUC(bx, ax, by, ay)
end

"""
    twissring(seq_or_lat, dp, order; E0=3e9, m0=M_ELECTRON, h=1e-8)

JuTrack-style ring Twiss API mapped to [`TwissResult`](@ref) (`dp` is δP unless `wrt=:deltae`).
"""
function twissring(lat_in, dp::Real, order::Integer;
                   E0::Real=3.0e9,
                   m0::Real=M_ELECTRON,
                   h::Real=1e-8,
                   wrt::Symbol=:deltap)
    if order != 0
        @warn "twissring: order > 0 TPSA optics is deferred; using finite-difference fallback."
    end
    lat = _as_lattice(lat_in)
    _require_periodic(lat, "twissring")
    beam = _beam_from_energy_mass(E0, m0)
    T = typeof(beam.energy)
    ref = SVector{6,T}(zero(T), zero(T), zero(T), zero(T), zero(T),
                       T(_deltae_input(dp, beam.beta, wrt)))
    return twissline(lat, beam; reference=ref, h=T(h))
end

"""
    twissring(seq_or_lat, dp, order, refpts; E0=3e9, m0=M_ELECTRON, h=1e-8)

JuTrack-style Twiss-at-reference-points interface.
"""
function twissring(lat_in, dp::Real, order::Integer, refpts::AbstractVector{<:Integer};
                   E0::Real=3.0e9,
                   m0::Real=M_ELECTRON,
                   h::Real=1e-8,
                   wrt::Symbol=:deltap)
    lat = _as_lattice(lat_in)
    _validate_refpts(refpts, length(lat))
    tw = twissring(lat, dp, order; E0=E0, m0=m0, h=h, wrt=wrt)
    T = eltype(tw.s)
    out = Vector{optics4DUC{T}}(undef, length(refpts))
    for (i, rp) in enumerate(refpts)
        k = rp + 1
        out[i] = optics4DUC(
            optics2D(tw.betax[k], tw.alphax[k], tw.mux[k], tw.dx[k], tw.dpx[k]),
            optics2D(tw.betay[k], tw.alphay[k], tw.muy[k], tw.dy[k], tw.dpy[k]),
        )
    end
    return out
end

"""
    twissline(tin, seq_or_lat, dp, order, endindex; E0=3e9, m0=M_ELECTRON, h=3e-8)

JuTrack-style optics propagation through the first `endindex` elements.
"""
function twissline(tin::optics4DUC, lat_in, dp::Real, order::Integer, endindex::Integer;
                   E0::Real=3.0e9,
                   m0::Real=M_ELECTRON,
                   h::Real=3e-8,
                   wrt::Symbol=:deltap)
    lat = _as_lattice(lat_in)
    1 <= endindex <= length(lat) || throw(ArgumentError("endindex must be in [1, $(length(lat))]"))
    used = Lattice(lat.elements[1:endindex])
    M = findm66(used, dp, order; E0=E0, m0=m0, h=h, wrt=wrt)
    return twissPropagate(tin, M)
end

"""
    twissline(tin, seq_or_lat, dp, order, refpts; E0=3e9, m0=M_ELECTRON, h=3e-8)

JuTrack-style optics propagation at specified reference points.
"""
function twissline(tin::optics4DUC, lat_in, dp::Real, order::Integer, refpts::AbstractVector{<:Integer};
                   E0::Real=3.0e9,
                   m0::Real=M_ELECTRON,
                   h::Real=3e-8,
                   wrt::Symbol=:deltap)
    lat = _as_lattice(lat_in)
    _validate_refpts(refpts, length(lat))
    Mlist = findm66_refpts(lat, dp, order, refpts; E0=E0, m0=m0, h=h, wrt=wrt)
    out = Vector{typeof(tin)}(undef, length(refpts))
    cur = tin
    for i in eachindex(refpts)
        cur = twissPropagate(cur, @view Mlist[:, :, i])
        out[i] = cur
    end
    return out
end
