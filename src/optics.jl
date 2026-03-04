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
export TwissLineResult, one_turn_map, gettune, getchrom, twissline
export find_closed_orbit_4d, find_closed_orbit_6d
export findm66, fastfindm66, findm66_refpts, fastfindm66_refpts
export periodicEdwardsTengTwiss, twissring, twissPropagate

abstract type AbstractOptics end
abstract type AbstractOptics2D <: AbstractOptics end
abstract type AbstractOptics4D <: AbstractOptics end

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

struct TwissLineResult{T}
    s::Vector{T}
    betax::Vector{T}
    alphax::Vector{T}
    betay::Vector{T}
    alphay::Vector{T}
    mux::Vector{T}
    muy::Vector{T}
    tunex::T
    tuney::T
end

@inline _unit6(::Type{T}, i::Int) where T = SVector{6,T}(ntuple(j -> (j == i ? one(T) : zero(T)), 6))
@inline _as_lattice(lat::Lattice) = lat
@inline _as_lattice(seq::AbstractVector{<:AbstractElement}) = Lattice(seq)

function _dp_with_orbit(dp::Real, orb::AbstractVector)
    if length(orb) != 6
        throw(ArgumentError("orb must have length 6"))
    end
    return (dp == 0 && orb[6] != 0) ? orb[6] : dp
end

function _reference6(::Type{T}, dp::Real, orb::AbstractVector) where T
    dpeff = _dp_with_orbit(dp, orb)
    return SVector{6,T}(T(orb[1]), T(orb[2]), T(orb[3]), T(orb[4]), T(orb[5]), T(dpeff))
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
    one_turn_map(lat, beam; reference=zeros, h=1e-8)

Finite-difference estimate of the 6x6 one-turn Jacobian about `reference`.
"""
function one_turn_map(lat::Lattice, beam::Beam{T};
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
    fastfindm66(lat, dp=0.0; E0=3e9, m0=M_ELECTRON, orb=zeros(6), h=3e-8)

JuTrack-compatible finite-difference 6x6 map API.
"""
function fastfindm66(lat_in, dp::Real=0.0;
                     E0::Real=3.0e9,
                     m0::Real=M_ELECTRON,
                     orb::AbstractVector=zeros(6),
                     h::Real=3e-8)
    lat = _as_lattice(lat_in)
    beam = _beam_from_energy_mass(E0, m0)
    T = typeof(beam.energy)
    ref = _reference6(T, dp, orb)
    return one_turn_map(lat, beam; reference=ref, h=T(h))
end

"""
    findm66(lat, dp, order; E0=3e9, m0=M_ELECTRON, orb=zeros(6), h=3e-8)

JuTrack-compatible map API. `order > 0` (TPSA path) is deferred and falls back to finite-difference.
"""
function findm66(lat_in, dp::Real, order::Integer;
                 E0::Real=3.0e9,
                 m0::Real=M_ELECTRON,
                 orb::AbstractVector=zeros(6),
                 h::Real=3e-8)
    if order != 0
        @warn "findm66: order > 0 TPSA map is deferred; using finite-difference fallback."
    end
    return fastfindm66(lat_in, dp; E0=E0, m0=m0, orb=orb, h=h)
end

"""
    fastfindm66_refpts(lat, dp, refpts; E0=3e9, m0=M_ELECTRON, orb=zeros(6), h=3e-8)

Return 6x6 segment maps at each reference point index.
"""
function fastfindm66_refpts(lat_in, dp::Real, refpts::AbstractVector{<:Integer};
                            E0::Real=3.0e9,
                            m0::Real=M_ELECTRON,
                            orb::AbstractVector=zeros(6),
                            h::Real=3e-8)
    lat = _as_lattice(lat_in)
    _validate_refpts(refpts, length(lat))
    beam = _beam_from_energy_mass(E0, m0)
    T = typeof(beam.energy)
    ref = _reference6(T, dp, orb)

    maps = zeros(T, 6, 6, length(refpts))
    prev = 0
    for (i, rp) in enumerate(refpts)
        seg = prev == 0 ? lat.elements[1:rp] : lat.elements[prev+1:rp]
        seg_lat = Lattice(seg)
        maps[:, :, i] = one_turn_map(seg_lat, beam; reference=ref, h=T(h))
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
                        h::Real=3e-8)
    if order != 0
        @warn "findm66_refpts: order > 0 TPSA map is deferred; using finite-difference fallback."
    end
    return fastfindm66_refpts(lat_in, dp, refpts; E0=E0, m0=m0, orb=orb, h=h)
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
    gettune(lat, beam; reference=zeros, h=1e-8)

Return `(Qx, Qy)` from the uncoupled blocks of the one-turn map.
"""
function gettune(lat::Lattice, beam::Beam{T};
                 reference::SVector{6,T}=zero(SVector{6,T}),
                 h::T=T(1e-8)) where T
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
    getchrom(lat, beam; dp=1e-6, reference=zeros, h=1e-8)

Finite-difference chromaticity `(ξx, ξy)` from tune variation with momentum offset.
"""
function getchrom(lat::Lattice, beam::Beam{T};
                  dp::T=zero(T),
                  reference::SVector{6,T}=zero(SVector{6,T}),
                  h::T=T(1e-8)) where T
    ref0 = SVector{6,T}(reference[1], reference[2], reference[3], reference[4], reference[5], T(dp))
    M0 = findm66(lat, dp, 0; E0=beam.energy, m0=beam.mass, orb=collect(ref0), h=h)
    qx0, qy0 = _tune_from_map(M0)
    dpp = T(1e-8)
    ref1 = SVector{6,T}(reference[1], reference[2], reference[3], reference[4], reference[5], T(dp) + dpp)
    M1 = findm66(lat, dp + dpp, 0; E0=beam.energy, m0=beam.mass, orb=collect(ref1), h=h)
    qx1, qy1 = _tune_from_map(M1)
    return (qx1 - qx0) / dpp, (qy1 - qy0) / dpp
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
    find_closed_orbit_6d(lat, beam; x0=zeros, tol=1e-10, maxiter=20, h=1e-6, reg=1e-12)

Find a 6-D closed orbit using Newton iterations with finite-difference Jacobians.
"""
function find_closed_orbit_6d(lat::Lattice, beam::Beam{T};
                              x0::SVector{6,T}=zero(SVector{6,T}),
                              tol::T=T(1e-10),
                              maxiter::Int=20,
                              h::T=T(1e-6),
                              reg::T=T(1e-12)) where T
    x = collect(x0)
    eye = Matrix{T}(I, 6, 6)

    f(θ::Vector{T}) = collect(linepass(lat, SVector{6,T}(θ...), beam))

    for _ in 1:maxiter
        J, x_out = _numerical_jacobian(f, x; h=h)
        Δ = x_out .- x
        if norm(Δ) < tol
            return SVector{6,T}(x...)
        end
        Δx = (eye - J + reg * eye) \ Δ
        x .+= Δx
    end
    return SVector{6,T}(x...)
end

"""
    find_closed_orbit_4d(lat, beam; dp=0, x0=zeros, tol=1e-10, maxiter=20, h=1e-6, reg=1e-12)

Find a 4-D closed orbit `(x, px, y, py)` at fixed momentum offset `dp`.
"""
function find_closed_orbit_4d(lat::Lattice, beam::Beam{T};
                              dp::T=zero(T),
                              x0::SVector{4,T}=zero(SVector{4,T}),
                              tol::T=T(1e-10),
                              maxiter::Int=20,
                              h::T=T(1e-6),
                              reg::T=T(1e-12)) where T
    x = collect(x0)
    eye = Matrix{T}(I, 4, 4)

    function f(θ::Vector{T})
        rin = SVector{6,T}(θ[1], θ[2], θ[3], θ[4], zero(T), dp)
        rout = linepass(lat, rin, beam)
        return T[rout[1], rout[2], rout[3], rout[4]]
    end

    for _ in 1:maxiter
        J, x_out = _numerical_jacobian(f, x; h=h)
        Δ = x_out .- x
        if norm(Δ) < tol
            return SVector{4,T}(x...)
        end
        Δx = (eye - J + reg * eye) \ Δ
        x .+= Δx
    end
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

"""
    twissline(lat, beam; reference=zeros, h=1e-8)

Compute uncoupled periodic Twiss functions at each element boundary.
"""
function twissline(lat::Lattice, beam::Beam{T};
                   reference::SVector{6,T}=zero(SVector{6,T}),
                   h::T=T(1e-8)) where T
    n = length(lat)
    β_inv = beti(beam)

    s = zeros(T, n + 1)
    Jlist = Vector{Matrix{T}}(undef, n)
    r = reference
    for (i, elem) in enumerate(lat.elements)
        s[i+1] = s[i] + T(get_length(elem))
        Jlist[i] = _element_jacobian(elem, r, β_inv, h)
        r = pass!(elem, r, β_inv)
    end

    M = Matrix{T}(I, 6, 6)
    for J in Jlist
        M = J * M
    end

    βx0, αx0, _, qx = _twiss_from_2x2(@view M[1:2, 1:2])
    βy0, αy0, _, qy = _twiss_from_2x2(@view M[3:4, 3:4])

    betax = zeros(T, n + 1); betax[1] = βx0
    alphax = zeros(T, n + 1); alphax[1] = αx0
    betay = zeros(T, n + 1); betay[1] = βy0
    alphay = zeros(T, n + 1); alphay[1] = αy0
    mux = zeros(T, n + 1)
    muy = zeros(T, n + 1)

    for i in 1:n
        A = @view Jlist[i][1:2, 1:2]
        B = @view Jlist[i][3:4, 3:4]

        betax[i+1], alphax[i+1], dmx = _propagate_twiss(betax[i], alphax[i], A)
        betay[i+1], alphay[i+1], dmy = _propagate_twiss(betay[i], alphay[i], B)

        mux[i+1] = mux[i] + dmx
        muy[i+1] = muy[i] + dmy
    end

    return TwissLineResult{T}(s, betax, alphax, betay, alphay, mux, muy, qx, qy)
end

"""
    periodicEdwardsTengTwiss(seq_or_lat, dp, order; E0=3e9, m0=M_ELECTRON, orb=zeros(6), h=1e-8)

JuTrack-style periodic optics interface (uncoupled 4D projection).
"""
function periodicEdwardsTengTwiss(lat_in, dp::Real, order::Integer;
                                  E0::Real=3.0e9,
                                  m0::Real=M_ELECTRON,
                                  orb::AbstractVector=zeros(6),
                                  h::Real=1e-8)
    M = findm66(lat_in, dp, order; E0=E0, m0=m0, orb=orb, h=h)
    bx, ax, _, _ = _twiss_from_2x2(@view M[1:2, 1:2])
    by, ay, _, _ = _twiss_from_2x2(@view M[3:4, 3:4])
    return optics4DUC(bx, ax, by, ay)
end

"""
    twissring(seq_or_lat, dp, order; E0=3e9, m0=M_ELECTRON, h=1e-8)

JuTrack-style ring Twiss API mapped to `TwissLineResult`.
"""
function twissring(lat_in, dp::Real, order::Integer;
                   E0::Real=3.0e9,
                   m0::Real=M_ELECTRON,
                   h::Real=1e-8)
    if order != 0
        @warn "twissring: order > 0 TPSA optics is deferred; using finite-difference fallback."
    end
    lat = _as_lattice(lat_in)
    beam = _beam_from_energy_mass(E0, m0)
    T = typeof(beam.energy)
    ref = SVector{6,T}(zero(T), zero(T), zero(T), zero(T), zero(T), T(dp))
    return twissline(lat, beam; reference=ref, h=T(h))
end

"""
    twissring(seq_or_lat, dp, order, refpts; E0=3e9, m0=M_ELECTRON, h=1e-8)

JuTrack-style Twiss-at-reference-points interface.
"""
function twissring(lat_in, dp::Real, order::Integer, refpts::AbstractVector{<:Integer};
                   E0::Real=3.0e9,
                   m0::Real=M_ELECTRON,
                   h::Real=1e-8)
    lat = _as_lattice(lat_in)
    _validate_refpts(refpts, length(lat))
    tw = twissring(lat, dp, order; E0=E0, m0=m0, h=h)
    T = eltype(tw.s)
    out = Vector{optics4DUC{T}}(undef, length(refpts))
    for (i, rp) in enumerate(refpts)
        k = rp + 1
        out[i] = optics4DUC(
            optics2D(tw.betax[k], tw.alphax[k], tw.mux[k], zero(T), zero(T)),
            optics2D(tw.betay[k], tw.alphay[k], tw.muy[k], zero(T), zero(T)),
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
                   h::Real=3e-8)
    lat = _as_lattice(lat_in)
    1 <= endindex <= length(lat) || throw(ArgumentError("endindex must be in [1, $(length(lat))]"))
    used = Lattice(lat.elements[1:endindex])
    M = findm66(used, dp, order; E0=E0, m0=m0, h=h)
    return twissPropagate(tin, M)
end

"""
    twissline(tin, seq_or_lat, dp, order, refpts; E0=3e9, m0=M_ELECTRON, h=3e-8)

JuTrack-style optics propagation at specified reference points.
"""
function twissline(tin::optics4DUC, lat_in, dp::Real, order::Integer, refpts::AbstractVector{<:Integer};
                   E0::Real=3.0e9,
                   m0::Real=M_ELECTRON,
                   h::Real=3e-8)
    lat = _as_lattice(lat_in)
    _validate_refpts(refpts, length(lat))
    Mlist = findm66_refpts(lat, dp, order, refpts; E0=E0, m0=m0, h=h)
    out = Vector{typeof(tin)}(undef, length(refpts))
    cur = tin
    for i in eachindex(refpts)
        cur = twissPropagate(cur, @view Mlist[:, :, i])
        out[i] = cur
    end
    return out
end
