"""
    TrackPadPolySeriesExt

PolySeries extension for TrackPad.jl. Provides CTPS (truncated power series)
tracking through accelerator elements for computing high-order Taylor maps.

This module is loaded automatically when `using PolySeries` is called alongside TrackPad.
"""
module TrackPadPolySeriesExt

using TrackPad
using PolySeries
using StaticArrays

# Type alias for readability
# const CTPSVec{T} = SVector{6,CTPS{T}} where T

# ============================================================================
# CTPS-specific helpers (self-contained, no core modification)
# ============================================================================

# ── Misalignment ──

@inline function apply_misalignment_tpsa(r::SVector{6,CTPS{T}}, t::SVector{6,T}) where T
    return SVector{6,CTPS{T}}(r[1] + t[1], r[2] + t[2], r[3] + t[3],
                               r[4] + t[4], r[5] + t[5], r[6] + t[6])
end

@inline function apply_rotation_tpsa(r::SVector{6,CTPS{T}}, R::SMatrix{6,6,T,36}) where T
    return SVector{6,CTPS{T}}(
        R[1,1]*r[1] + R[1,2]*r[2] + R[1,3]*r[3] + R[1,4]*r[4] + R[1,5]*r[5] + R[1,6]*r[6],
        R[2,1]*r[1] + R[2,2]*r[2] + R[2,3]*r[3] + R[2,4]*r[4] + R[2,5]*r[5] + R[2,6]*r[6],
        R[3,1]*r[1] + R[3,2]*r[2] + R[3,3]*r[3] + R[3,4]*r[4] + R[3,5]*r[5] + R[3,6]*r[6],
        R[4,1]*r[1] + R[4,2]*r[2] + R[4,3]*r[3] + R[4,4]*r[4] + R[4,5]*r[5] + R[4,6]*r[6],
        R[5,1]*r[1] + R[5,2]*r[2] + R[5,3]*r[3] + R[5,4]*r[4] + R[5,5]*r[5] + R[5,6]*r[6],
        R[6,1]*r[1] + R[6,2]*r[2] + R[6,3]*r[3] + R[6,4]*r[4] + R[6,5]*r[5] + R[6,6]*r[6])
end

@inline function enter_misalignment(r::SVector{6,CTPS{T}}, t1::SVector{6,T}, r1::SMatrix{6,6,T,36}) where T
    r = apply_misalignment_tpsa(r, t1)
    if !iszero(r1)
        r = apply_rotation_tpsa(r, r1)
    end
    return r
end

@inline function exit_misalignment(r::SVector{6,CTPS{T}}, t2::SVector{6,T}, r2::SMatrix{6,6,T,36}) where T
    if !iszero(r2)
        r = apply_rotation_tpsa(r, r2)
    end
    r = apply_misalignment_tpsa(r, t2)
    return r
end

# ── Drift ──

@inline function drift6_tpsa(r::SVector{6,CTPS{T}}, L::T, beti::T) where T
    # Exact Hamiltonian: pz = sqrt(1 + 2δ/β + δ² - px² - py²)
    pz2 = CTPS(one(T)) + 2*r[6]*beti + r[6]^2 - r[2]^2 - r[4]^2
    NormL = L / sqrt(pz2)
    x_new = r[1] + NormL * r[2]
    y_new = r[3] + NormL * r[4]
    z_new = r[5] + NormL * (beti + r[6]) - L * beti
    return SVector{6,CTPS{T}}(x_new, r[2], y_new, r[4], z_new, r[6])
end

# # ── Linearized drift (kept for reference) ──
# @inline function drift6_tpsa_linearized(r::SVector{6,CTPS{T}}, L::T, beti::T) where T
#     NormL = L / (CTPS(one(T)) + r[6])
#     x_new = r[1] + NormL * r[2]
#     y_new = r[3] + NormL * r[4]
#     z_new = r[5] + NormL * (r[2]^2 + r[4]^2) / (2 * (CTPS(one(T)) + r[6]))
#     return SVector{6,CTPS{T}}(x_new, r[2], y_new, r[4], z_new, r[6])
# end

# ── Multipole kicks ──

@inline function strthinkick_tpsa(r::SVector{6,CTPS{T}},
                                   polynom_a::SVector{N,T},
                                   polynom_b::SVector{N,T},
                                   L::T, max_order::Int) where {T,N}
    ReSum = CTPS(polynom_b[max_order + 1])
    ImSum = CTPS(polynom_a[max_order + 1])
    @inbounds for i in max_order:-1:1
        ReSumTemp = ReSum * r[1] - ImSum * r[3] + polynom_b[i]
        ImSum = ImSum * r[1] + ReSum * r[3] + polynom_a[i]
        ReSum = ReSumTemp
    end
    px_new = r[2] - L * ReSum
    py_new = r[4] + L * ImSum
    return SVector{6,CTPS{T}}(r[1], px_new, r[3], py_new, r[5], r[6])
end

@inline function bndthinkick_tpsa(r::SVector{6,CTPS{T}},
                                   polynom_a::SVector{N,T},
                                   polynom_b::SVector{N,T},
                                   L::T, irho::T,
                                   max_order::Int, beti::T) where {T,N}
    ReSum = CTPS(polynom_b[max_order + 1])
    ImSum = CTPS(polynom_a[max_order + 1])
    @inbounds for i in max_order:-1:1
        ReSumTemp = ReSum * r[1] - ImSum * r[3] + polynom_b[i]
        ImSum = ImSum * r[1] + ReSum * r[3] + polynom_a[i]
        ReSum = ReSumTemp
    end
    px_new = r[2] - L * (ReSum - (r[6] - r[1] * irho) * irho)
    py_new = r[4] + L * ImSum
    z_new  = r[5] + L * irho * r[1] * beti
    return SVector{6,CTPS{T}}(r[1], px_new, r[3], py_new, z_new, r[6])
end

# ── Symplectic integrators ──

@inline function symplectic4_pass_tpsa(r::SVector{6,CTPS{T}}, L::T,
                                        polynom_a::SVector{N,T},
                                        polynom_b::SVector{N,T},
                                        max_order::Int, num_steps::Int,
                                        beti::T) where {T,N}
    SL = L / num_steps
    L1 = SL * T(TrackPad.DRIFT1)
    L2 = SL * T(TrackPad.DRIFT2)
    K1 = SL * T(TrackPad.KICK1)
    K2 = SL * T(TrackPad.KICK2)
    for _ in 1:num_steps
        r = drift6_tpsa(r, L1, beti)
        r = strthinkick_tpsa(r, polynom_a, polynom_b, K1, max_order)
        r = drift6_tpsa(r, L2, beti)
        r = strthinkick_tpsa(r, polynom_a, polynom_b, K2, max_order)
        r = drift6_tpsa(r, L2, beti)
        r = strthinkick_tpsa(r, polynom_a, polynom_b, K1, max_order)
        r = drift6_tpsa(r, L1, beti)
    end
    return r
end

@inline function symplectic4_bend_pass_tpsa(r::SVector{6,CTPS{T}}, L::T,
                                             polynom_a::SVector{N,T},
                                             polynom_b::SVector{N,T},
                                             irho::T,
                                             max_order::Int, num_steps::Int,
                                             beti::T) where {T,N}
    SL = L / num_steps
    L1 = SL * T(TrackPad.DRIFT1)
    L2 = SL * T(TrackPad.DRIFT2)
    K1 = SL * T(TrackPad.KICK1)
    K2 = SL * T(TrackPad.KICK2)
    for _ in 1:num_steps
        r = drift6_tpsa(r, L1, beti)
        r = bndthinkick_tpsa(r, polynom_a, polynom_b, K1, irho, max_order, beti)
        r = drift6_tpsa(r, L2, beti)
        r = bndthinkick_tpsa(r, polynom_a, polynom_b, K2, irho, max_order, beti)
        r = drift6_tpsa(r, L2, beti)
        r = bndthinkick_tpsa(r, polynom_a, polynom_b, K1, irho, max_order, beti)
        r = drift6_tpsa(r, L1, beti)
    end
    return r
end

# ── Edge focusing ──

@inline function edge_fringe_entrance_tpsa(r::SVector{6,CTPS{T}}, inv_rho::T, edge_angle::T,
                                            fint::T, gap::T, method::Int) where T
    if iszero(fint) || iszero(gap) || method == 0
        fringecorr = zero(T)
    else
        sedge = sin(edge_angle)
        cedge = cos(edge_angle)
        fringecorr = inv_rho * gap * fint * (one(T) + sedge^2) / cedge
    end

    fx = inv_rho * tan(edge_angle)

    if method == 1
        fy = inv_rho * tan(edge_angle - fringecorr / (one(T) + cst(r[6])))
    elseif method == 2
        fy = inv_rho * tan(edge_angle - fringecorr / (one(T) + cst(r[6]))) / (one(T) + cst(r[6]))
    elseif method == 3
        fy = inv_rho * tan(edge_angle - fringecorr + cst(r[2]) / (one(T) + cst(r[6])))
    else
        fy = inv_rho * tan(edge_angle - fringecorr / (one(T) + cst(r[6])))
    end

    px_new = r[2] + r[1] * fx
    py_new = r[4] - r[3] * fy
    return SVector{6,CTPS{T}}(r[1], px_new, r[3], py_new, r[5], r[6])
end

@inline function edge_fringe_exit_tpsa(r::SVector{6,CTPS{T}}, inv_rho::T, edge_angle::T,
                                        fint::T, gap::T, method::Int) where T
    if iszero(fint) || iszero(gap) || method == 0
        fringecorr = zero(T)
    else
        sedge = sin(edge_angle)
        cedge = cos(edge_angle)
        fringecorr = inv_rho * gap * fint * (one(T) + sedge^2) / cedge
    end

    fx = inv_rho * tan(edge_angle)

    if method == 1
        fy = inv_rho * tan(edge_angle - fringecorr / (one(T) + cst(r[6])))
    elseif method == 2
        fy = inv_rho * tan(edge_angle - fringecorr / (one(T) + cst(r[6]))) / (one(T) + cst(r[6]))
    elseif method == 3
        fy = inv_rho * tan(edge_angle - fringecorr - cst(r[2]) / (one(T) + cst(r[6])))
    else
        fy = inv_rho * tan(edge_angle - fringecorr / (one(T) + cst(r[6])))
    end

    px_new = r[2] + r[1] * fx
    py_new = r[4] - r[3] * fy
    return SVector{6,CTPS{T}}(r[1], px_new, r[3], py_new, r[5], r[6])
end

# ============================================================================
# pass! dispatches on CTPS coordinates
# ============================================================================

# ── Drift ──

function TrackPad.pass!(elem::Drift{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)
    r = drift6_tpsa(r, elem.L, beti)
    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── Marker ──

function TrackPad.pass!(elem::Marker, r::SVector{6,CTPS{T}}, beti::T) where T
    return r
end

# ── Quadrupole ──

function TrackPad.pass!(elem::Quadrupole{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)

    polynom_b = SVector{4,T}(elem.polynom_b[1], elem.k1, elem.polynom_b[3], elem.polynom_b[4])
    if elem.L > zero(T)
        kick_correction_b = SVector{4,T}(
            polynom_b[1] - sin(elem.kick_angle[1]) / elem.L,
            polynom_b[2], polynom_b[3], polynom_b[4])
        kick_correction_a = SVector{4,T}(
            elem.polynom_a[1] + sin(elem.kick_angle[2]) / elem.L,
            elem.polynom_a[2], elem.polynom_a[3], elem.polynom_a[4])
    else
        kick_correction_b = polynom_b
        kick_correction_a = elem.polynom_a
    end

    r = symplectic4_pass_tpsa(r, elem.L, kick_correction_a, kick_correction_b,
                               elem.max_order, elem.num_int_steps, beti)
    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── Sextupole ──

function TrackPad.pass!(elem::Sextupole{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)

    polynom_b = SVector{4,T}(elem.polynom_b[1], elem.polynom_b[2], elem.k2/2, elem.polynom_b[4])
    r = symplectic4_pass_tpsa(r, elem.L, elem.polynom_a, polynom_b,
                               elem.max_order, elem.num_int_steps, beti)

    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── Octupole ──

function TrackPad.pass!(elem::Octupole{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)

    polynom_b = SVector{4,T}(elem.polynom_b[1], elem.polynom_b[2], elem.polynom_b[3], elem.k3/6)
    r = symplectic4_pass_tpsa(r, elem.L, elem.polynom_a, polynom_b,
                               elem.max_order, elem.num_int_steps, beti)

    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── Thin Multipole ──

function TrackPad.pass!(elem::ThinMultipole{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)
    r = strthinkick_tpsa(r, elem.polynom_a, elem.polynom_b, one(T), elem.max_order)
    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── SBend (Sector Bend) ──

function TrackPad.pass!(elem::SBend{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)

    irho = elem.angle / elem.L
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

    if elem.fringe_bend_entrance != 0
        r = edge_fringe_entrance_tpsa(r, irho, elem.e1, elem.fint1, elem.gap,
                                       elem.fringe_bend_entrance)
    end

    r = symplectic4_bend_pass_tpsa(r, elem.L, polynom_a, polynom_b, irho,
                                    elem.max_order, elem.num_int_steps, beti)

    if elem.fringe_bend_exit != 0
        r = edge_fringe_exit_tpsa(r, irho, elem.e2, elem.fint2, elem.gap,
                                   elem.fringe_bend_exit)
    end

    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── RFCavity ──

function TrackPad.pass!(elem::RFCavity{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    beta = inv(beti)
    if elem.L > zero(T)
        r = drift6_tpsa(r, elem.L / 2, beti)
    end
    if elem.energy > zero(T)
        nv = elem.volt / elem.energy
        phase = T(2pi) * elem.freq * ((r[5] - elem.lag) / T(TrackPad.C_LIGHT)) - elem.philag
        delta_new = r[6] - nv * sin(phase) / (beta * beta)
        r = SVector{6,CTPS{T}}(r[1], r[2], r[3], r[4], r[5], delta_new)
    end
    if elem.L > zero(T)
        r = drift6_tpsa(r, elem.L / 2, beti)
    end
    return r
end

# ── Solenoid ──

function TrackPad.pass!(elem::Solenoid{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)

    ks = elem.ks
    L = elem.L

    if iszero(ks)
        r = drift6_tpsa(r, L, beti)
    else
        p_norm = inv(CTPS(one(T)) + r[6])
        x   = r[1]
        xpr = r[2] * p_norm
        y   = r[3]
        ypr = r[4] * p_norm
        H   = ks * p_norm / 2
        S   = sin(L * H)
        C   = cos(L * H)

        x_new  = x*C*C + xpr*C*S/H + y*C*S + ypr*S*S/H
        px_new = (-x*H*C*S + xpr*C*C - y*H*S*S + ypr*C*S) / p_norm
        y_new  = -x*C*S - xpr*S*S/H + y*C*C + ypr*C*S/H
        py_new = (x*H*S*S - xpr*C*S - y*C*S*H + ypr*C*C) / p_norm
        z_new  = r[5] + L*(H*H*(x*x + y*y) + 2*H*(xpr*y - ypr*x) + xpr*xpr + ypr*ypr) / 2

        r = SVector{6,CTPS{T}}(x_new, px_new, y_new, py_new, z_new, r[6])
    end

    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── Corrector ──

function TrackPad.pass!(elem::Corrector{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)

    p_norm = inv(CTPS(one(T)) + r[6])
    NormL = elem.L * p_norm
    z_new = r[5] + NormL * p_norm *
            (elem.xkick^2 / 3 + elem.ykick^2 / 3 +
             r[2]^2 + r[4]^2 + r[2] * elem.xkick + r[4] * elem.ykick) / 2
    x_new  = r[1] + NormL * (r[2] + elem.xkick / 2)
    px_new = r[2] + elem.xkick
    y_new  = r[3] + NormL * (r[4] + elem.ykick / 2)
    py_new = r[4] + elem.ykick
    r = SVector{6,CTPS{T}}(x_new, px_new, y_new, py_new, z_new, r[6])

    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── DriftSC ──

function TrackPad.pass!(elem::DriftSC{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)
    r = drift6_tpsa(r, elem.L, beti)
    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── QuadrupoleSC ──

function TrackPad.pass!(elem::QuadrupoleSC{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)

    polynom_b = SVector{4,T}(elem.k0, elem.k1, elem.k2 / 2, elem.k3 / 6)
    polynom_a = elem.polynom_a
    if elem.L > zero(T)
        polynom_b = SVector{4,T}(polynom_b[1] - sin(elem.kick_angle[1]) / elem.L,
                                  polynom_b[2], polynom_b[3], polynom_b[4])
        polynom_a = SVector{4,T}(polynom_a[1] + sin(elem.kick_angle[2]) / elem.L,
                                  polynom_a[2], polynom_a[3], polynom_a[4])
    end
    r = symplectic4_pass_tpsa(r, elem.L, polynom_a, polynom_b,
                               clamp(elem.max_order, 0, 3), elem.num_int_steps, beti)

    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── SextupoleSC ──

function TrackPad.pass!(elem::SextupoleSC{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)
    polynom_b = SVector{4,T}(elem.k0, elem.k1, elem.k2 / 2, elem.k3 / 6)
    r = symplectic4_pass_tpsa(r, elem.L, elem.polynom_a, polynom_b,
                               clamp(elem.max_order, 0, 3), elem.num_int_steps, beti)
    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── OctupoleSC ──

function TrackPad.pass!(elem::OctupoleSC{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)
    polynom_b = SVector{4,T}(elem.k0, elem.k1, elem.k2 / 2, elem.k3 / 6)
    r = symplectic4_pass_tpsa(r, elem.L, elem.polynom_a, polynom_b,
                               clamp(elem.max_order, 0, 3), elem.num_int_steps, beti)
    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── SBendSC ──

function TrackPad.pass!(elem::SBendSC{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)

    irho = elem.angle / elem.L
    polynom_b = elem.polynom_b
    polynom_a = elem.polynom_a
    if elem.L > zero(T)
        polynom_b = SVector{4,T}(polynom_b[1] - sin(elem.kick_angle[1]) / elem.L,
                                  polynom_b[2], polynom_b[3], polynom_b[4])
        polynom_a = SVector{4,T}(polynom_a[1] + sin(elem.kick_angle[2]) / elem.L,
                                  polynom_a[2], polynom_a[3], polynom_a[4])
    end

    if elem.fringe_bend_entrance != 0
        r = edge_fringe_entrance_tpsa(r, irho, elem.e1, elem.fint1, elem.gap,
                                       elem.fringe_bend_entrance)
    end
    r = symplectic4_bend_pass_tpsa(r, elem.L, polynom_a, polynom_b, irho,
                                    clamp(elem.max_order, 0, 3), elem.num_int_steps, beti)
    if elem.fringe_bend_exit != 0
        r = edge_fringe_exit_tpsa(r, irho, elem.e2, elem.fint2, elem.gap,
                                   elem.fringe_bend_exit)
    end

    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── SpaceCharge (no-op) ──

function TrackPad.pass!(elem::SpaceCharge{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    return r
end

# ── Translation ──

function TrackPad.pass!(elem::Translation{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    pz2 = CTPS(one(T)) + 2*r[6]*beti + r[6]^2 - r[2]^2 - r[4]^2
    pz = sqrt(pz2)
    x_new = r[1] - (elem.dx + elem.ds * r[2] / pz)
    y_new = r[3] - (elem.dy + elem.ds * r[4] / pz)
    z_new = r[5] + elem.ds * (beti + r[6]) / pz
    return SVector{6,CTPS{T}}(x_new, r[2], y_new, r[4], z_new, r[6])
end

# ── YRotation ──

function TrackPad.pass!(elem::YRotation{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    angle = -elem.angle
    if iszero(angle)
        return r
    end
    ca = cos(angle)
    sa = sin(angle)
    ta = tan(angle)

    pz2 = CTPS(one(T)) + 2*r[6]*beti + r[6]^2 - r[2]^2 - r[4]^2
    pz = sqrt(pz2)
    ptt = CTPS(one(T)) - ta * r[2] / pz

    x_new  = r[1] / (ca * ptt)
    px_new = ca * r[2] + sa * pz
    y_new  = r[3] + ta * r[1] * r[4] / (pz * ptt)
    z_new  = r[5] + ta * r[1] * (beti + r[6]) / (pz * ptt)
    return SVector{6,CTPS{T}}(x_new, px_new, y_new, r[4], z_new, r[6])
end

# ── CrabCavity ──

function TrackPad.pass!(elem::CrabCavity{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    volt = elem.volt * (one(T) + elem.errors[1])
    phi = elem.phi + elem.errors[2]
    E = max(abs(elem.energy), eps(T))
    ang = elem.k * r[5] + phi

    if elem.L > zero(T)
        r = drift6_tpsa(r, elem.L / 2, beti)
    end
    px_new    = r[2] + (volt / E) * sin(ang * beti)
    delta_new = r[6] - (elem.k * volt / E * beti) * r[1] * cos(ang * beti)
    r = SVector{6,CTPS{T}}(r[1], px_new, r[3], r[4], r[5], delta_new)
    if elem.L > zero(T)
        r = drift6_tpsa(r, elem.L / 2, beti)
    end
    return r
end

# ── AccelCavity ──

function TrackPad.pass!(elem::AccelCavity{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    beta = inv(beti)
    beta2 = beta * beta
    if beta2 <= eps(T)
        return r
    end
    E = max(abs(elem.energy), eps(T))
    sv = sin(elem.k * r[5] + elem.phis) - sin(elem.phis)
    delta_new = r[6] + (elem.volt / (beta2 * E)) * sv
    return SVector{6,CTPS{T}}(r[1], r[2], r[3], r[4], r[5], delta_new)
end

# ── LongitudinalRFMap ──

function TrackPad.pass!(elem::LongitudinalRFMap{T,E}, r::SVector{6,CTPS{T}}, beti::T) where {T,E}
    k = T(TrackPad._rf_k(elem.rf))
    if abs(k) <= eps(T)
        return r
    end
    h = T(TrackPad._rf_h(elem.rf))
    beta = inv(beti)
    eta = elem.alphac - (one(T) - beta * beta)
    z_new = r[5] - (T(2pi) * h * eta / k) * r[6]
    return SVector{6,CTPS{T}}(r[1], r[2], r[3], r[4], z_new, r[6])
end

# ── LorentzBoost ──

function TrackPad.pass!(elem::LorentzBoost{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    if elem.mode != 0
        return r
    end
    invcos = inv(elem.cosang)
    x_new     = r[1] + elem.tanang * r[5]
    delta_new = r[6] - elem.tanang * r[2]
    return SVector{6,CTPS{T}}(x_new, r[2]*invcos, r[3], r[4]*invcos, r[5]*invcos, delta_new)
end

# ── InvLorentzBoost ──

function TrackPad.pass!(elem::InvLorentzBoost{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    if elem.mode != 0
        return r
    end
    x_new     = r[1] - elem.sinang * r[5]
    delta_new = r[6] + elem.sinang * r[2]
    return SVector{6,CTPS{T}}(x_new, r[2]*elem.cosang, r[3], r[4]*elem.cosang,
                               r[5]*elem.cosang, delta_new)
end

# ── StrongThinGaussianBeam ──

function TrackPad.pass!(elem::StrongThinGaussianBeam{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    sx2 = max(elem.rmssizex^2, eps(T))
    sy2 = max(elem.rmssizey^2, eps(T))
    dx = r[1] - elem.xoffset
    dy = r[3] - elem.yoffset
    g = elem.amplitude * exp(CTPS(-one(T)/2) * (dx^2 / sx2 + dy^2 / sy2))
    px_new = r[2] - g * dx / sx2
    py_new = r[4] - g * dy / sy2
    return SVector{6,CTPS{T}}(r[1], px_new, r[3], py_new, r[5], r[6])
end

# ── StrongGaussianBeam ──

function TrackPad.pass!(elem::StrongGaussianBeam{T,N,V}, r::SVector{6,CTPS{T}}, beti::T) where {T,N,V}
    if elem.nzslice <= 0
        return r
    end
    sx2 = max(elem.beamsize[1]^2, eps(T))
    sy2 = max(elem.beamsize[2]^2, eps(T))
    factor = elem.charge / max(abs(elem.total_energy), eps(T))
    px = r[2]
    py = r[4]
    @inbounds for i in 1:elem.nzslice
        w    = i <= length(elem.zslice_npar) ? elem.zslice_npar[i] : one(T) / T(elem.nzslice)
        xoff = i <= length(elem.xoffsets)    ? elem.xoffsets[i]    : zero(T)
        yoff = i <= length(elem.yoffsets)    ? elem.yoffsets[i]    : zero(T)
        dx = r[1] - xoff
        dy = r[3] - yoff
        g = w * exp(CTPS(-one(T)/2) * (dx^2 / sx2 + dy^2 / sy2))
        px = px - factor * g * dx / sx2
        py = py - factor * g * dy / sy2
    end
    return SVector{6,CTPS{T}}(r[1], px, r[3], py, r[5], r[6])
end

# ── LongitudinalRLCWake ──

function TrackPad.pass!(elem::LongitudinalRLCWake{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    if iszero(elem.scale)
        return r
    end
    t = min(cst(r[5]) / T(TrackPad.C_LIGHT), zero(T))
    delta_new = r[6] - elem.scale * TrackPad.wakefieldfunc_RLCWake(elem, t)
    return SVector{6,CTPS{T}}(r[1], r[2], r[3], r[4], r[5], delta_new)
end

# ── LongitudinalWake ──

function TrackPad.pass!(elem::LongitudinalWake{T,N,V}, r::SVector{6,CTPS{T}}, beti::T) where {T,N,V}
    if iszero(elem.scale)
        return r
    end
    t = cst(r[5]) / T(TrackPad.C_LIGHT)
    delta_new = r[6] - elem.scale * TrackPad.wakefieldfunc(elem, t)
    return SVector{6,CTPS{T}}(r[1], r[2], r[3], r[4], r[5], delta_new)
end

# ============================================================================
# linepass / ringpass — extend originals, dispatch on CTPS automatically
# ============================================================================

function TrackPad.linepass(lat::Lattice, r::SVector{6,CTPS{T}}, beam::Beam{T};
                           time::Real=zero(T), turn::Integer=0) where T
    β_inv = TrackPad.beti(beam)
    ctx = TrackPad.TimeContext(T(time); turn=turn)
    for elem in lat.elements
        elem_now = TrackPad._resolve_for_time(elem, ctx)
        r = TrackPad.pass!(elem_now, r, β_inv)
    end
    return r
end

function TrackPad.linepass(lat::Lattice, r::SVector{6,CTPS{T}};
                           time::Real=zero(T), turn::Integer=0) where T
    beam = Beam(T(1.0e9))
    return TrackPad.linepass(lat, r, beam; time=time, turn=turn)
end

function TrackPad.ringpass(lat::Lattice, r::SVector{6,CTPS{T}}, beam::Beam{T}, nturns::Int;
                           time::Real=zero(T), dt_turn::Real=zero(T), turn::Integer=0) where T
    t = T(time)
    dt = T(dt_turn)
    trn = Int(turn)
    for _ in 1:nturns
        r = TrackPad.linepass(lat, r, beam; time=t, turn=trn)
        t += dt
        trn += 1
    end
    return r
end

# ============================================================================
# TPSA-specific utilities (no Float64 counterpart)
# ============================================================================

function TrackPad.polyseries_variables(::Type{T}; order::Int=1) where T
    nv = 6
    set_descriptor!(nv, order)
    return SVector{nv,CTPS{T}}(ntuple(i -> CTPS(zero(T), i), nv))
end

function  TrackPad.polyseries_variables(x0::SVector{6,T}; order::Int=1) where T
    nv = 6
    set_descriptor!(nv, order)
    return SVector{nv,CTPS{T}}(ntuple(i -> CTPS(x0[i], i), nv))
    
end

function TrackPad.polyseries_one_turn_map(lat::Lattice, beam::Beam{T};
                        r0::SVector{6,T}=SVector{6,T}(zeros(T,6)), order::Int=1) where T
    r = TrackPad.polyseries_variables(r0; order=order)
    return TrackPad.linepass(lat, r, beam)
end

end # module TrackPadPolySeriesExt