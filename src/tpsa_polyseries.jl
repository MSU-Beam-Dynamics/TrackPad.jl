# =============================================================================
# TPSA backend: PolySeries CTPS coordinates through the generic kernels
#
# TrackPad's element kernels and `pass!` methods are generic in the coordinate
# type (loss branches are guarded by `_check_pz2`/`_check_tiny`, `check_lost`
# and `aperture_lost` are no-ops for non-`Real` coordinates), so
# `SVector{6,CTPS{T}}` coordinates flow through the core `pass!`, `linepass`
# and `track` unchanged. This file only provides
#
#   * `pass!` methods for elements whose Float64 kernel uses an operation CTPS
#     does not have (Faddeeva function, ordered comparisons), replaced by a
#     series form or rejected with an explicit error;
#   * the `method=:tpsa` implementations of the optics primitives (closed
#     orbit, Jacobians, chromaticity, amplitude detuning), which are the
#     default `method` of `periodic_twiss` and `getchrom`;
#   * the TPSA utilities `polyseries_variables`, `polyseries_one_turn_map` and
#     `tpsa_map`.
# =============================================================================


# ============================================================================
# Elements needing a CTPS-specific map
# ============================================================================

# ── StrongThinGaussianBeam / StrongGaussianBeam ──
# The Bassetti-Erskine field needs the Faddeeva function of a complex series and
# ordered comparisons, neither of which CTPS provides. Round beams use the
# everywhere-convergent power series in r^2/(2σ^2) from the core (no division by
# r^2, so the on-axis reference orbit is fine); elliptical beams are rejected.

@inline function _tpsa_beam_field(x, y, sigmax::T, sigmay::T, what::AbstractString) where T
    sx2 = sigmax * sigmax
    sy2 = sigmay * sigmay
    abs(sx2 - sy2) <= T(1e-10) * (sx2 + sy2) || throw(ArgumentError(
        "$what TPSA map: only round strong beams (rmssizex == rmssizey) have a " *
        "series form of the beam-beam kick; the elliptical Bassetti-Erskine field " *
        "needs the Faddeeva function of a complex series, which PolySeries does not provide."))
    return _round_beam_field_series(x, y, (sx2 + sy2) / 2)
end

function pass!(elem::StrongThinGaussianBeam{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    dx = r[1] - elem.xoffset
    dy = r[3] - elem.yoffset
    Ex, Ey = _tpsa_beam_field(dx, dy, elem.rmssizex, elem.rmssizey, "StrongThinGaussianBeam")
    return SVector{6,CTPS{T}}(r[1], r[2] + elem.amplitude * Ex, r[3], r[4] + elem.amplitude * Ey, r[5], r[6])
end

function pass!(elem::StrongGaussianBeam{T,N,V}, r::SVector{6,CTPS{T}}, beti::T) where {T,N,V}
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
        Ex, Ey = _tpsa_beam_field(xc - xoff, yc - yoff, elem.beamsize[1], elem.beamsize[2], "StrongGaussianBeam")
        px = px + (elem.kick_scale * npar) * Ex
        py = py + (elem.kick_scale * npar) * Ey
        x = xc - px * sstar
        y = yc - py * sstar
    end
    return SVector{6,CTPS{T}}(x, px, y, py, r[5], r[6])
end

# ── LongitudinalRLCWake / LongitudinalWake ──
# Collective elements: the wake potential is the convolution of the wake Green
# function with the histogram of r[5] over all macroparticles. A single TPSA
# particle carries no bunch distribution, so TPSA wake tracking is rejected
# (the base pass! methods throw ArgumentError).

# ── Elements served by TrackPad's generic kernels ──
# Every other element (Drift, Quadrupole, Sextupole, Octupole, ThinMultipole,
# SBend, RFCavity, Solenoid, Corrector, the *SC variants, CrabCavity,
# AccelCavity, LongitudinalRFMap, LorentzBoost/InvLorentzBoost, Marker, Patch,
# Translation, YRotation, SpaceCharge, ...) has no CTPS-specific method: the
# core `pass!` implementations are written for any coordinate type and their
# loss branches are guarded (`_check_pz2`, `_check_tiny`), so CTPS coordinates
# flow through them unchanged. The same holds for `linepass`/`track`
# (`check_lost` and `aperture_lost` are `false` for non-`Real` coordinates, so a
# TPSA map is always computed as if the apertures were absent).

# ── LBend ──
# The linear bend body branches on the sign of the momentum-dependent focusing
# strengths, which has no series analogue.

function pass!(::LBend{T,N}, ::SVector{6,CTPS{T}}, ::T) where {T,N}
    throw(ArgumentError(
        "LBend has no TPSA map: its body selects the trigonometric or hyperbolic " *
        "branch from the sign of the momentum-dependent focusing. Use SBend or " *
        "ExactSBend for Taylor maps."))
end

# ============================================================================
# Optics through Taylor maps  (method = :tpsa)
# ============================================================================
# These implement the `Val{:tpsa}` methods of the optics building blocks in
# src/optics.jl. Jacobians are exact (order-1 series), the closed orbit is a
# Newton iteration on exact Jacobians, and the chromaticity is read off an
# order-2 map about the closed orbit with no finite-difference step at all.

const _E6 = ntuple(j -> [k == j ? 1 : 0 for k in 1:6], 6)

# Constant term and exact Jacobian of `f` (a map on SVector{6,CTPS}) about `ref`.
function _tpsa_linear(f, ref::SVector{6,T}) where T
    set_descriptor!(6, 1)
    r = SVector{6,CTPS{T}}(ntuple(i -> CTPS(ref[i], i, 6, 1), 6))
    out = f(r)
    M = Matrix{T}(undef, 6, 6)
    c = Vector{T}(undef, 6)
    @inbounds for i in 1:6
        c[i] = cst(out[i])
        for j in 1:6
            M[i, j] = element(out[i], _E6[j])
        end
    end
    return c, M
end

function _one_turn_jacobian(lat::Lattice, beam::Beam{T}, ref::SVector{6,T},
                                     h::T, ::Val{:tpsa}) where T
    return _tpsa_linear(r -> linepass(lat, r, beam), ref)[2]
end

function _optics_jacobians(lat::Lattice, beam::Beam{T}, ref::SVector{6,T},
                                    h::T, ::Val{:tpsa}) where T
    n = length(lat)
    β_inv = beti(beam)
    s = zeros(T, n + 1)
    jacobians = Vector{Matrix{T}}(undef, n)
    r = ref
    ctx = TimeContext(zero(T))
    for (i, elem_raw) in enumerate(lat.elements)
        elem = _resolve_for_time(elem_raw, ctx)
        s[i + 1] = s[i] + T(get_length(elem))
        c, J = _tpsa_linear(q -> pass!(elem, q, β_inv), r)
        jacobians[i] = J
        r = SVector{6,T}(c...)
    end
    return s, jacobians, r
end

# Newton iteration on the exact order-1 map. Returns the closed orbit (with the
# z of `reference`) and the one-turn Jacobian about it, which the last
# iteration has already computed.
function _closed_orbit_jacobian(lat::Lattice, beam::Beam{T}, δE::T,
                                reference::SVector{6,T}, h::T, ::Val{:tpsa}) where T
    r = SVector{6,T}(reference[1], reference[2], reference[3], reference[4], reference[5], δE)
    tol = T(1e-14)
    for _ in 1:30
        c, M = _tpsa_linear(q -> linepass(lat, q, beam), r)
        res = SVector{4,T}(c[1] - r[1], c[2] - r[2], c[3] - r[3], c[4] - r[4])
        maximum(abs, res) < tol && return M, r
        A = SMatrix{4,4,T}(@view M[1:4, 1:4]) - I
        dx = A \ res
        r = SVector{6,T}(r[1] - dx[1], r[2] - dx[2], r[3] - dx[3], r[4] - dx[4], reference[5], δE)
    end
    throw(ArgumentError("TPSA closed-orbit search did not converge; the lattice may have no stable off-momentum closed orbit"))
end

function _closed_orbit_4d(lat::Lattice, beam::Beam{T}, δE::T, x0::SVector{4,T},
                          mth::Val{:tpsa}) where T
    _, r = _closed_orbit_jacobian(lat, beam, δE, SVector{6,T}(x0[1], x0[2], x0[3], x0[4], zero(T), δE), zero(T), mth)
    return SVector{4,T}(r[1], r[2], r[3], r[4])
end

# dQ/dp for a tune Q = acos((M11+M22)/2)/2π given ∂M/∂p of the 2×2 block.
@inline function _dtune(M::AbstractMatrix{T}, dM::AbstractMatrix{T}) where T
    qx, qy = _tune_from_map(M)
    return (-(dM[1, 1] + dM[2, 2]) / (4T(π) * sin(2T(π) * qx)),
            -(dM[3, 3] + dM[4, 4]) / (4T(π) * sin(2T(π) * qy)))
end

function _getchrom_tpsa(lat::Lattice, beam::Beam{T}, ::Val{:tpsa};
                                 dp::T=zero(T),
                                 reference::SVector{6,T}=zero(SVector{6,T}),
                                 wrt::Symbol=:deltap) where T
    _require_periodic(lat, "getchrom")
    β0 = beam.beta
    δE = wrt === :deltae ? dp : T(deltae_from_deltap(dp, β0))
    ref = SVector{6,T}(reference[1], reference[2], reference[3], reference[4], reference[5], δE)

    # Order-2 map about the closed orbit: M_ij = ∂f_i/∂x_j and ∂M_ij/∂x_k from
    # the mixed second-order coefficients (a diagonal x_j² coefficient is ½∂²f).
    # `reference` is usually already the closed orbit (periodic_twiss passes
    # it, and a ring without orbit errors has x_co = 0), in which case the
    # constant term of this map confirms it and no Newton solve is needed.
    m = tpsa_map(lat, beam; order=2, closed_orbit=collect(ref))
    if maximum(abs(cst(m[i]) - ref[i]) for i in 1:4) >= T(1e-14)
        _, ref = _closed_orbit_jacobian(lat, beam, δE, ref, zero(T), Val(:tpsa))
        m = tpsa_map(lat, beam; order=2, closed_orbit=collect(ref))
    end
    M = [element(m[i], _E6[j]) for i in 1:6, j in 1:6]
    dM = [Matrix{T}(undef, 6, 6) for _ in 1:6]
    for k in 1:6, i in 1:6, j in 1:6
        dM[k][i, j] = j == k ? 2 * element(m[i], 2 .* _E6[j]) : element(m[i], _E6[j] .+ _E6[k])
    end

    # Total derivative along the off-momentum closed orbit x_co(δE) = D δE:
    # dQ/dδE = ∂Q/∂δE + Σ_k D_k ∂Q/∂x_k, with D the periodic δE-dispersion.
    D = (I - SMatrix{4,4,T}(@view M[1:4, 1:4])) \ SVector{4,T}(@view M[1:4, 6])
    ξx, ξy = _dtune(M, dM[6])
    for k in 1:4
        gx, gy = _dtune(M, dM[k])
        ξx += D[k] * gx
        ξy += D[k] * gy
    end
    # Convert the δE derivative to the requested variable (dδE/dδP = local β).
    scale = wrt === :deltae ? one(T) : T(_beta_at_deltae(δE, β0))
    return ξx * scale, ξy * scale
end

function _amplitude_detuning(lat::Lattice, beam::Beam{T}, ref::SVector{6,T},
                                      βx::T, αx::T, βy::T, αy::T, qx::T, qy::T,
                                      actions, nturns::Int, ::Val{:tpsa}) where T
    # Iterate the order-3 Taylor map about the closed orbit instead of tracking.
    # The truncated map is not exactly symplectic, which is harmless for the
    # ~1e3 turns and small actions used here, and it is what makes the result
    # a property of the Taylor map rather than of the integrator.
    m = tpsa_map(lat, beam; order=3, closed_orbit=collect(ref))
    function turns!(buf, r0)
        d = collect(r0 .- ref)                       # map variables are offsets from the closed orbit
        @inbounds for n in 1:size(buf, 1)
            d = [m[i](d...) for i in 1:6]
            buf[n, 1] = d[1] + ref[1]; buf[n, 2] = d[2] + ref[2]
            buf[n, 3] = d[3] + ref[3]; buf[n, 4] = d[4] + ref[4]
        end
    end
    return _detuning_from_turns(turns!, ref, βx, αx, βy, αy, qx, qy, actions, nturns)
end

# ============================================================================
# TPSA-specific utilities (no Float64 counterpart)
# ============================================================================

function polyseries_variables(::Type{T}; order::Int=1) where T
    nv = 6
    set_descriptor!(nv, order)
    return SVector{nv,CTPS{T}}(ntuple(i -> CTPS(zero(T), i), nv))
end

function polyseries_variables(x0::SVector{6,T}; order::Int=1) where T
    nv = 6
    set_descriptor!(nv, order)
    return SVector{nv,CTPS{T}}(ntuple(i -> CTPS(x0[i], i), nv))

end

function polyseries_one_turn_map(lat::Lattice, beam::Beam{T};
                        r0::SVector{6,T}=SVector{6,T}(zeros(T,6)), order::Int=1) where T
    r = polyseries_variables(r0; order=order)
    return linepass(lat, r, beam)
end

"""
    tpsa_map(lat::Lattice, beam::Beam;
                     order::Int=1, closed_orbit=nothing)

Compute the Taylor transfer map through `lat` about `closed_orbit`.
"""
function tpsa_map(lat::Lattice, beam::Beam;
                           order::Int=1,
                           closed_orbit::Union{Nothing,AbstractVector}=nothing)
    set_descriptor!(6, order)
    co = isnothing(closed_orbit) ? zeros(Float64, 6) : Float64.(closed_orbit)
    length(co) == 6 || throw(ArgumentError("closed_orbit must contain six coordinates"))
    r0 = SVector(
        CTPS(co[1], 1, 6, order),
        CTPS(co[2], 2, 6, order),
        CTPS(co[3], 3, 6, order),
        CTPS(co[4], 4, 6, order),
        CTPS(co[5], 5, 6, order),
        CTPS(co[6], 6, 6, order),
    )
    return linepass(lat, r0, beam)
end

