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
    z_new = r[5] - (NormL * (beti + r[6]) - L * beti)
    return SVector{6,CTPS{T}}(x_new, r[2], y_new, r[4], z_new, r[6])
end

# # ── Linearized drift (kept for reference) ──
# @inline function drift6_tpsa_linearized(r::SVector{6,CTPS{T}}, L::T, beti::T) where T
#     NormL = L / (CTPS(one(T)) + r[6])
#     x_new = r[1] + NormL * r[2]
#     y_new = r[3] + NormL * r[4]
#     z_new = r[5] - NormL * (r[2]^2 + r[4]^2) / (2 * (CTPS(one(T)) + r[6]))
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
    px_new = r[2] - L * (ReSum - (r[6] * beti - r[1] * irho) * irho)
    py_new = r[4] + L * ImSum
    z_new  = r[5] - L * irho * r[1] * beti
    return SVector{6,CTPS{T}}(r[1], px_new, r[3], py_new, z_new, r[6])
end

@inline function multipole_fringe_tpsa(r::SVector{6,CTPS{T}},
                                        polynom_a::SVector{N,T},
                                        polynom_b::SVector{N,T},
                                        max_order::Int, edge::T,
                                        skip_b0::Int, beti::T) where {T,N}
    FX = CTPS(zero(T)); FY = CTPS(zero(T))
    FX_X = CTPS(zero(T)); FX_Y = CTPS(zero(T))
    FY_X = CTPS(zero(T)); FY_Y = CTPS(zero(T))
    RX = CTPS(one(T)); IX = CTPS(zero(T))

    @inbounds for n in 0:max_order
        B = polynom_b[n + 1]
        A = polynom_a[n + 1]
        j = T(n + 1)
        DRX = RX
        DIX = IX
        RX = DRX * r[1] - DIX * r[3]
        IX = DRX * r[3] + DIX * r[1]

        U = CTPS(zero(T)); V = CTPS(zero(T))
        DU = CTPS(zero(T)); DV = CTPS(zero(T))
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
    den = MA * MD - MB * MC

    x_new = r[1] - FX * DEL
    y_new = r[3] - FY * DEL
    px_new = (MD * r[2] - MB * r[4]) / den
    py_new = (MA * r[4] - MC * r[2]) / den
    z_new = r[5] + (px_new * FX + py_new * FY) * DEL * DEL
    return SVector{6,CTPS{T}}(x_new, px_new, y_new, py_new, z_new, r[6])
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
#
# Dipole edge focusing is evaluated with TrackPad's generic kernels
# (`edge_fringe_entrance`/`edge_fringe_exit`), which are written for any
# coordinate type. An earlier TPSA-specific copy used `cst(r[6])`, i.e. only the
# constant part of the series, and therefore dropped the chromatic (∂/∂δ) and,
# for the THOMX model, the ∂/∂px terms of the edge focusing from the map.
const edge_fringe_entrance_tpsa = TrackPad.edge_fringe_entrance
const edge_fringe_exit_tpsa = TrackPad.edge_fringe_exit

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

    if !iszero(elem.fringe_entrance)
        r = multipole_fringe_tpsa(r, kick_correction_a, kick_correction_b,
                                  elem.max_order, one(T), 1, beti)
    end
    r = symplectic4_pass_tpsa(r, elem.L, kick_correction_a, kick_correction_b,
                               elem.max_order, elem.num_int_steps, beti)
    if !iszero(elem.fringe_exit)
        r = multipole_fringe_tpsa(r, kick_correction_a, kick_correction_b,
                                  elem.max_order, -one(T), 1, beti)
    end
    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── Sextupole ──

function TrackPad.pass!(elem::Sextupole{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)

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
        r = multipole_fringe_tpsa(r, polynom_a, polynom_b,
                                  elem.max_order, one(T), 1, beti)
    end
    r = symplectic4_pass_tpsa(r, elem.L, polynom_a, polynom_b,
                               elem.max_order, elem.num_int_steps, beti)
    if !iszero(elem.fringe_exit)
        r = multipole_fringe_tpsa(r, polynom_a, polynom_b,
                                  elem.max_order, -one(T), 1, beti)
    end

    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── Octupole ──

function TrackPad.pass!(elem::Octupole{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)

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
        r = multipole_fringe_tpsa(r, polynom_a, polynom_b,
                                  elem.max_order, one(T), 1, beti)
    end
    r = symplectic4_pass_tpsa(r, elem.L, polynom_a, polynom_b,
                               elem.max_order, elem.num_int_steps, beti)
    if !iszero(elem.fringe_exit)
        r = multipole_fringe_tpsa(r, polynom_a, polynom_b,
                                  elem.max_order, -one(T), 1, beti)
    end

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
    if !iszero(elem.fringe_quad_entrance)
        r = multipole_fringe_tpsa(r, polynom_a, polynom_b,
                                  elem.max_order, one(T), 1, beti)
    end

    r = symplectic4_bend_pass_tpsa(r, elem.L, polynom_a, polynom_b, irho,
                                    elem.max_order, elem.num_int_steps, beti)

    if !iszero(elem.fringe_quad_exit)
        r = multipole_fringe_tpsa(r, polynom_a, polynom_b,
                                  elem.max_order, -one(T), 1, beti)
    end
    if elem.fringe_bend_exit != 0
        r = edge_fringe_exit_tpsa(r, irho, elem.e2, elem.fint2, elem.gap,
                                   elem.fringe_bend_exit)
    end

    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── RFCavity ──

function TrackPad.pass!(elem::RFCavity{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    if elem.L > zero(T)
        r = drift6_tpsa(r, elem.L / 2, beti)
    end
    if elem.energy > zero(T)
        kick = elem.charge * elem.volt / TrackPad._reference_p0c(elem.energy, beti)
        phase = -T(2pi) * elem.freq * ((r[5] + elem.lag) / T(TrackPad.C_LIGHT)) - elem.philag
        delta_new = r[6] - kick * sin(phase)
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
        momentum = TrackPad._momentum_norm(r[6], beti)
        p_norm = inv(momentum)
        momentum_jacobian = (beti + r[6]) * p_norm
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
        z_new  = r[5] - momentum_jacobian * L *
                 (H*H*(x*x + y*y) + 2*H*(xpr*y - ypr*x) + xpr*xpr + ypr*ypr) / 2

        r = SVector{6,CTPS{T}}(x_new, px_new, y_new, py_new, z_new, r[6])
    end

    r = exit_misalignment(r, elem.t2, elem.r2)
    return r
end

# ── Corrector ──

function TrackPad.pass!(elem::Corrector{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    r = enter_misalignment(r, elem.t1, elem.r1)

    momentum = TrackPad._momentum_norm(r[6], beti)
    p_norm = inv(momentum)
    momentum_jacobian = (beti + r[6]) * p_norm
    NormL = elem.L * p_norm
    z_new = r[5] - momentum_jacobian * NormL * p_norm *
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

# ── CrabCavity ──

function TrackPad.pass!(elem::CrabCavity{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    volt = elem.volt * (one(T) + elem.errors[1])
    phi = elem.phi + elem.errors[2]
    p0c = max(TrackPad._reference_p0c(elem.energy, beti), eps(T))
    ang = -elem.k * r[5] + phi

    if elem.L > zero(T)
        r = drift6_tpsa(r, elem.L / 2, beti)
    end
    kick = elem.charge * volt / p0c
    px_new    = r[2] + kick * sin(ang)
    delta_new = r[6] - elem.k * kick * r[1] * cos(ang)
    r = SVector{6,CTPS{T}}(r[1], px_new, r[3], r[4], r[5], delta_new)
    if elem.L > zero(T)
        r = drift6_tpsa(r, elem.L / 2, beti)
    end
    return r
end

# ── AccelCavity ──

function TrackPad.pass!(elem::AccelCavity{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    p0c = max(TrackPad._reference_p0c(elem.energy, beti), eps(T))
    sv = sin(-elem.k * r[5] + elem.phis) - sin(elem.phis)
    delta_new = r[6] + (elem.charge * elem.volt / p0c) * sv
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
    z_new = r[5] - (T(2pi) * h * eta * beti / k) * r[6]
    return SVector{6,CTPS{T}}(r[1], r[2], r[3], r[4], z_new, r[6])
end

# ── LorentzBoost ──

function TrackPad.pass!(elem::LorentzBoost{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    if elem.mode != 0
        return r
    end
    x_new     = r[1] - elem.tanang * r[5]
    z_new     = r[5] / elem.cosang
    delta_new = elem.tanang * elem.cosang * r[2] + elem.cosang * r[6]
    return SVector{6,CTPS{T}}(x_new, r[2], r[3], r[4], z_new, delta_new)
end

# ── InvLorentzBoost ──

function TrackPad.pass!(elem::InvLorentzBoost{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    if elem.mode != 0
        return r
    end
    x_new     = r[1] + elem.sinang * r[5]
    delta_new = (r[6] - elem.sinang * r[2]) / elem.cosang
    return SVector{6,CTPS{T}}(x_new, r[2], r[3], r[4],
                              r[5]*elem.cosang, delta_new)
end

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
    return TrackPad._round_beam_field_series(x, y, (sx2 + sy2) / 2)
end

function TrackPad.pass!(elem::StrongThinGaussianBeam{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    dx = r[1] - elem.xoffset
    dy = r[3] - elem.yoffset
    Ex, Ey = _tpsa_beam_field(dx, dy, elem.rmssizex, elem.rmssizey, "StrongThinGaussianBeam")
    return SVector{6,CTPS{T}}(r[1], r[2] + elem.amplitude * Ex, r[3], r[4] + elem.amplitude * Ey, r[5], r[6])
end

function TrackPad.pass!(elem::StrongGaussianBeam{T,N,V}, r::SVector{6,CTPS{T}}, beti::T) where {T,N,V}
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
# (the base TrackPad.pass! methods throw ArgumentError).

# ── Elements served by TrackPad's generic kernels ──
# Marker, Patch, Translation, YRotation and SpaceCharge have no CTPS-specific
# method: the core `pass!` implementations are written for any coordinate type
# and their loss branches are guarded (`_check_pz2`, `_check_tiny`), so CTPS
# coordinates flow through them unchanged.

# ── ExactSBend ──
# The exact-bend body, wedge (`bend_edge`, uses `asin`) and rotation (`yrot`)
# are generic and work on CTPS. The hard-edge bend fringe (`bend_fringe`)
# requires `atan`, which PolySeries does not define, so it is rejected
# explicitly instead of failing deep inside the kernel.

function TrackPad.pass!(elem::ExactSBend{T,N}, r::SVector{6,CTPS{T}}, beti::T) where {T,N}
    if elem.fringe_bend_entrance != 0 || elem.fringe_bend_exit != 0
        throw(ArgumentError(
            "ExactSBend TPSA map: the hard-edge bend fringe (`fringe_bend_entrance`/" *
            "`fringe_bend_exit` != 0) needs `atan` on CTPS, which PolySeries does " *
            "not provide. Construct the element with fringe_bend_entrance=0, " *
            "fringe_bend_exit=0 for TPSA maps."))
    end
    # Call the generic core kernel (the signature below excludes this method).
    return invoke(TrackPad.pass!, Tuple{ExactSBend{T,N}, SVector{6}, T}, elem, r, beti)
end

# ── LBend ──
# The linear bend body branches on the sign of the momentum-dependent focusing
# strengths, which has no series analogue.

function TrackPad.pass!(::LBend{T,N}, ::SVector{6,CTPS{T}}, ::T) where {T,N}
    throw(ArgumentError(
        "LBend has no TPSA map: its body selects the trigonometric or hyperbolic " *
        "branch from the sign of the momentum-dependent focusing. Use SBend or " *
        "ExactSBend for Taylor maps."))
end

# ============================================================================
# linepass / ringpass — extend originals, dispatch on CTPS automatically
# ============================================================================

function TrackPad.linepass(lat::Lattice, r::SVector{6,CTPS{T}}, beam::Beam{T};
                           time::Real=zero(T), turn::Integer=0,
                           check_apertures::Bool=true) where T
    # `check_apertures` is accepted for signature parity with the Float64 path;
    # an aperture test needs ordered comparisons that CTPS does not define, so
    # a TPSA map is always computed as if the apertures were absent.
    β_inv = TrackPad.beti(beam)
    ctx = TrackPad.TimeContext(T(time); turn=turn)
    for elem in lat.elements
        elem_now = TrackPad._resolve_for_time(elem, ctx)
        r = TrackPad.pass!(elem_now, r, β_inv)
    end
    return r
end

function TrackPad.linepass(lat::Lattice, r::SVector{6,CTPS{T}};
                           time::Real=zero(T), turn::Integer=0,
                           check_apertures::Bool=true) where T
    beam = Beam(T(1.0e9))
    return TrackPad.linepass(lat, r, beam; time=time, turn=turn)
end

function TrackPad.ringpass(lat::Lattice, r::SVector{6,CTPS{T}}, beam::Beam{T}, nturns::Int;
                           time::Real=zero(T), dt_turn::Real=zero(T), turn::Integer=0,
                           check_apertures::Bool=true) where T
    TrackPad._require_periodic(lat, "ringpass")
    nturns >= 0 || throw(ArgumentError("nturns must be nonnegative"))
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

"""
    TrackPad.tpsa_map(lat::Lattice, beam::Beam;
                     order::Int=1, closed_orbit=nothing)

Compute the Taylor transfer map through `lat` about `closed_orbit`.
"""
function TrackPad.tpsa_map(lat::Lattice, beam::Beam;
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
    return TrackPad.linepass(lat, r0, beam)
end

end # module TrackPadPolySeriesExt
