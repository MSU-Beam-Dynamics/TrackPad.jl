"""
    tracking.jl

Core particle tracking functions for TrackPad.jl.
Implements allocation-free symplectic integrators using StaticArrays.

Coordinate convention (matching JuTrack.jl / Accelerator Toolbox):
- r[1]: x  - Horizontal position
- r[2]: px - Horizontal momentum (px/p0)
- r[3]: y  - Vertical position  
- r[4]: py - Vertical momentum (py/p0)
- r[5]: z  - Path length difference (Δs or -c*Δt)
- r[6]: δ  - Relative momentum deviation (Δp/p0)
"""

using LinearAlgebra
using StaticArrays

export pass!, linepass!
export drift6!, strthinkick!

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
# Global Settings (can be toggled)
# =============================================================================
# Use exact Hamiltonian pz = sqrt((1 + 2δ/β + δ² - px² - py²)) vs linearized pz = 1 + δ
USE_EXACT_HAMILTONIAN::Bool = true

# =============================================================================
# Helper Functions (allocation-free)
# =============================================================================

"""
    check_lost(r::AbstractVector) -> Bool

Check if particle is lost (coordinates exceed limits or are NaN).
"""
@inline function check_lost(r::AbstractVector{T}) where T
    return isnan(r[1]) || abs(r[1]) > COORD_LIMIT || abs(r[3]) > COORD_LIMIT ||
           abs(r[2]) > ANGLE_LIMIT || abs(r[4]) > ANGLE_LIMIT
end

"""
    apply_misalignment!(r, t, R)

Apply misalignment: translation t and rotation R to coordinates r (in-place for mutable).
Returns new coordinates for immutable StaticArrays.
"""
@inline function apply_misalignment(r::SVector{6,T}, t::SVector{6,T}, R::SMatrix{6,6,T,36}) where T
    r_new = r + t
    return R * r_new
end

@inline function apply_misalignment(r::SVector{6,T}, t::SVector{6,T}) where T
    return r + t
end

# =============================================================================
# Drift Tracking
# =============================================================================

"""
    drift6!(r, L, beti=1.0) -> SVector{6,T}

Track particle through a drift space of length L.
Uses exact Hamiltonian when USE_EXACT_HAMILTONIAN is true.

# Arguments
- `r::SVector{6,T}`: 6D phase space coordinates
- `L::T`: Drift length
- `beti::T`: 1/β (inverse relativistic velocity), default 1.0

# Returns
- Updated coordinates as SVector{6,T}
"""
@inline function drift6(r::SVector{6,T}, L::T, beti::T=one(T)) where T
    if USE_EXACT_HAMILTONIAN
        # Exact Hamiltonian: pz = sqrt(1 + 2δ/β + δ² - px² - py²)
        pz2 = one(T) + 2*r[6]*beti + r[6]^2 - r[2]^2 - r[4]^2
        if pz2 <= zero(T)
            # Particle is lost
            return SVector{6,T}(T(NaN), T(NaN), T(NaN), T(NaN), T(NaN), T(NaN))
        end
        NormL = L / sqrt(pz2)
        x_new = r[1] + NormL * r[2]
        y_new = r[3] + NormL * r[4]
        z_new = r[5] + NormL * (beti + r[6]) - L * beti
        return SVector{6,T}(x_new, r[2], y_new, r[4], z_new, r[6])
    else
        # Linearized approximation: pz ≈ 1 + δ
        NormL = L / (one(T) + r[6])
        x_new = r[1] + NormL * r[2]
        y_new = r[3] + NormL * r[4]
        z_new = r[5] + NormL * (r[2]^2 + r[4]^2) / (2*(one(T) + r[6]))
        return SVector{6,T}(x_new, r[2], y_new, r[4], z_new, r[6])
    end
end

"""
    pass!(elem::Drift, r::SVector{6,T}, beti=1.0) -> SVector{6,T}

Track particle through a Drift element.
"""
function pass!(elem::Drift{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
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
function pass!(elem::Marker, r::SVector{6,T}, beti::T=one(T)) where T
    return r
end

# =============================================================================
# Dipole Edge Focusing (Fringe Fields)
# =============================================================================

"""
    edge_fringe_entrance(r, inv_rho, edge_angle, fint, gap, method) -> SVector{6,T}

Apply dipole edge focusing at entrance.

# Arguments
- `r`: 6D coordinates
- `inv_rho`: Inverse bending radius (1/ρ)
- `edge_angle`: Pole face rotation angle (e1)
- `fint`: Fringe field integral
- `gap`: Magnet gap
- `method`: Fringe calculation method (0=none, 1=Brown, 2=SOLEIL, 3=THOMX)
"""
@inline function edge_fringe_entrance(r::SVector{6,T}, inv_rho::T, edge_angle::T, 
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
        fy = inv_rho * tan(edge_angle - fringecorr / (one(T) + r[6]))
    elseif method == 2
        fy = inv_rho * tan(edge_angle - fringecorr / (one(T) + r[6])) / (one(T) + r[6])
    elseif method == 3
        fy = inv_rho * tan(edge_angle - fringecorr + r[2] / (one(T) + r[6]))
    else  # Fallback to Brown (method 1)
        fy = inv_rho * tan(edge_angle - fringecorr / (one(T) + r[6]))
    end
    
    px_new = r[2] + r[1] * fx
    py_new = r[4] - r[3] * fy
    
    return SVector{6,T}(r[1], px_new, r[3], py_new, r[5], r[6])
end

"""
    edge_fringe_exit(r, inv_rho, edge_angle, fint, gap, method) -> SVector{6,T}

Apply dipole edge focusing at exit.
Same as entrance but with opposite sign for THOMX method px term.
"""
@inline function edge_fringe_exit(r::SVector{6,T}, inv_rho::T, edge_angle::T, 
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
        fy = inv_rho * tan(edge_angle - fringecorr / (one(T) + r[6]))
    elseif method == 2
        fy = inv_rho * tan(edge_angle - fringecorr / (one(T) + r[6])) / (one(T) + r[6])
    elseif method == 3
        # Note: exit uses -r[2] instead of +r[2]
        fy = inv_rho * tan(edge_angle - fringecorr - r[2] / (one(T) + r[6]))
    else  # Fallback to Brown (method 1)
        fy = inv_rho * tan(edge_angle - fringecorr / (one(T) + r[6]))
    end
    
    px_new = r[2] + r[1] * fx
    py_new = r[4] - r[3] * fy
    
    return SVector{6,T}(r[1], px_new, r[3], py_new, r[5], r[6])
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
@inline function bndthinkick(r::SVector{6,T}, 
                             polynom_a::SVector{N,T}, 
                             polynom_b::SVector{N,T}, 
                             L::T, 
                             irho::T,
                             max_order::Int,
                             beti::T) where {T,N}
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
    # px -= L * (ReSum - (δ - x*irho) * irho)
    # py += L * ImSum
    # z += L * irho * x * beti  (path length in bend)
    px_new = r[2] - L * (ReSum - (r[6] - r[1] * irho) * irho)
    py_new = r[4] + L * ImSum
    z_new = r[5] + L * irho * r[1] * beti
    
    return SVector{6,T}(r[1], px_new, r[3], py_new, z_new, r[6])
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
@inline function strthinkick(r::SVector{6,T}, 
                             polynom_a::SVector{N,T}, 
                             polynom_b::SVector{N,T}, 
                             L::T, 
                             max_order::Int) where {T,N}
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
    
    return SVector{6,T}(r[1], px_new, r[3], py_new, r[5], r[6])
end

# =============================================================================
# 4th-Order Symplectic Integrator
# =============================================================================

"""
    symplectic4_pass(r, L, polynom_a, polynom_b, max_order, num_steps, beti) -> SVector{6,T}

4th-order Yoshida symplectic integrator for thick multipole elements.
Drift-Kick-Drift-Kick-Drift-Kick-Drift pattern per step.
"""
@inline function symplectic4_pass(r::SVector{6,T},
                                   L::T,
                                   polynom_a::SVector{N,T},
                                   polynom_b::SVector{N,T},
                                   max_order::Int,
                                   num_steps::Int,
                                   beti::T) where {T,N}
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
@inline function symplectic4_bend_pass(r::SVector{6,T},
                                        L::T,
                                        polynom_a::SVector{N,T},
                                        polynom_b::SVector{N,T},
                                        irho::T,
                                        max_order::Int,
                                        num_steps::Int,
                                        beti::T) where {T,N}
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

# =============================================================================
# Quadrupole Tracking
# =============================================================================

"""
    pass!(elem::Quadrupole, r::SVector{6,T}, beti=1.0) -> SVector{6,T}

Track particle through a Quadrupole using 4th-order symplectic integrator.
"""
function pass!(elem::Quadrupole{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
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
    
    # Symplectic integration
    r = symplectic4_pass(r, elem.L, kick_correction_a, kick_correction_b, 
                         elem.max_order, elem.num_int_steps, beti)
    
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
function pass!(elem::Sextupole{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    # Apply entrance misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    # k2 goes to polynom_b[3] (with factor 1/2 for standard normalization)
    polynom_b = SVector{4,T}(elem.polynom_b[1], elem.polynom_b[2], elem.k2/2, elem.polynom_b[4])
    
    # Symplectic integration
    r = symplectic4_pass(r, elem.L, elem.polynom_a, polynom_b, 
                         elem.max_order, elem.num_int_steps, beti)
    
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
function pass!(elem::Octupole{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    # Apply entrance misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    # k3 goes to polynom_b[4] (with factor 1/6 for standard normalization)
    polynom_b = SVector{4,T}(elem.polynom_b[1], elem.polynom_b[2], elem.polynom_b[3], elem.k3/6)
    
    # Symplectic integration
    r = symplectic4_pass(r, elem.L, elem.polynom_a, polynom_b, 
                         elem.max_order, elem.num_int_steps, beti)
    
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
"""
function pass!(elem::RFCavity{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    # Match JuTrack RFCA map (with nturn=0 in single-pass context).
    beta = inv(beti)
    if elem.L > zero(T)
        r = drift6(r, elem.L / 2, beti)
    end
    if elem.energy > zero(T)
        nv = elem.volt / elem.energy
        phase = T(2pi) * elem.freq * ((r[5] - elem.lag) / T(C_LIGHT)) - elem.philag
        delta_new = r[6] - nv * sin(phase) / (beta * beta)
        r = SVector{6,T}(r[1], r[2], r[3], r[4], r[5], delta_new)
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

Track particle through a Solenoid using exact matrix transformation.
"""
function pass!(elem::Solenoid{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
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
        # Match JuTrack SOLENOID map (linearized p_norm convention).
        p_norm = inv(one(T) + r[6])
        x = r[1]
        xpr = r[2] * p_norm
        y = r[3]
        ypr = r[4] * p_norm
        H = ks * p_norm / 2
        S = sin(L * H)
        C = cos(L * H)

        x_new = x * C * C + xpr * C * S / H + y * C * S + ypr * S * S / H
        px_new = (-x * H * C * S + xpr * C * C - y * H * S * S + ypr * C * S) / p_norm
        y_new = -x * C * S - xpr * S * S / H + y * C * C + ypr * C * S / H
        py_new = (x * H * S * S - xpr * C * S - y * C * S * H + ypr * C * C) / p_norm
        z_new = r[5] + L * (H * H * (x * x + y * y) + 2 * H * (xpr * y - ypr * x) + xpr * xpr + ypr * ypr) / 2

        r = SVector{6,T}(x_new, px_new, y_new, py_new, z_new, r[6])
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
function pass!(elem::Corrector{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    # Apply entrance misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    # Match JuTrack CORRECTOR map.
    p_norm = inv(one(T) + r[6])
    NormL = elem.L * p_norm
    z_new = r[5] + NormL * p_norm *
            (elem.xkick^2 / 3 + elem.ykick^2 / 3 +
             r[2]^2 + r[4]^2 + r[2] * elem.xkick + r[4] * elem.ykick) / 2
    x_new = r[1] + NormL * (r[2] + elem.xkick / 2)
    px_new = r[2] + elem.xkick
    y_new = r[3] + NormL * (r[4] + elem.ykick / 2)
    py_new = r[4] + elem.ykick
    r = SVector{6,T}(x_new, px_new, y_new, py_new, z_new, r[6])
    
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
function pass!(elem::ThinMultipole{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    # Apply entrance misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    # JuTrack thinMULTIPOLE uses an integrated-strength kick with unit kick length.
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
function pass!(elem::SBend{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
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
    
    # Symplectic integration through body
    r = symplectic4_bend_pass(r, elem.L, polynom_a, polynom_b, irho, 
                               elem.max_order, elem.num_int_steps, beti)
    
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
    pxyz(dp1, px, py) -> T

Helper for exact bend Hamiltonian: pz = sqrt(dp1^2 - px^2 - py^2)
"""
@inline function pxyz(dp1::T, px::T, py::T) where T
    val = dp1^2 - px^2 - py^2
    return val > zero(T) ? sqrt(val) : zero(T)
end

"""
    yrot!(r, phi, beti) -> SVector{6,T}

Rotation in free space (Forest 10.26).
"""
@inline function yrot(r::SVector{6,T}, phi::T, beti::T) where T
    if phi == zero(T)
        return r
    end
    
    dp1 = beti + r[6]
    c = cos(phi)
    s = sin(phi)
    pz = pxyz(dp1, r[2], r[4])
    
    p = c * pz - s * r[2]
    px_new = s * pz + c * r[2]
    x_new = r[1] * pz / p
    
    y_new = r[3] + r[1] * r[4] * s / p
    z_new = r[5] + dp1 * r[1] * s / p
    
    return SVector{6,T}(x_new, px_new, y_new, r[4], z_new, r[6])
end

"""
    bend_fringe(r, irho, gK, beti) -> SVector{6,T}

Hard-edge bend fringe (Forest 13.13).
"""
@inline function bend_fringe(r::SVector{6,T}, irho::T, gK::T, beti::T) where T
    b0 = irho
    dp1 = beti + r[6]
    pz = pxyz(dp1, r[2], r[4])
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
    powsec = sec(arg_sec)^2
    
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
    
    dd_num = (b0 * dp1 * (px * pz4 * (py2 - pz2) + b0 * gK * 
              (-(pz4 * py2z2_sq) + px2^2 * (3*py2*pz2 + 2*pz4) + 
              px2 * (3*py6 + 8*py4*pz2 + 7*py2*pz4 + pz6)))) * powsec
              
    dd = dd_num / denom
    
    # symplectic correction
    yf = (2 * r[3]) / (1 + sqrt(1 - 2 * dpy * r[3]))
    dxf = 0.5 * dpx * yf^2
    dct = 0.5 * dd * yf^2
    dpyf = phi * yf
    
    return SVector{6,T}(r[1] + dxf, r[2], yf, r[4] - dpyf, r[5] - dct, r[6])
end

"""
    bend_edge(r, rhoinv, theta, beti) -> SVector{6,T}

Ideal wedge map (Forest 12.41).
"""
@inline function bend_edge(r::SVector{6,T}, rhoinv::T, theta::T, beti::T) where T
    if abs(rhoinv) < 1e-6
        return r
    end
    
    dp1 = beti + r[6]
    c = cos(theta)
    s = sin(theta)
    pz = pxyz(dp1, r[2], r[4])
    d2 = pxyz(dp1, 0.0, r[4])
    
    px_new = r[2] * c + (pz - rhoinv * r[1]) * s
    
    # dasin term
    val1 = r[2] / d2
    val2 = px_new / d2
    # clamp to [-1, 1] to avoid domain error
    val1 = clamp(val1, -one(T), one(T))
    val2 = clamp(val2, -one(T), one(T))
    dasin = asin(val1) - asin(val2)
    
    num = r[1] * (r[2] * sin(2*theta) + s^2 * (2*pz - rhoinv * r[1]))
    den = pxyz(dp1, px_new, r[4]) + pz * c - r[2] * s
    
    x_new = r[1] * c + num / den
    y_new = r[3] + r[4] * (theta / rhoinv + dasin / rhoinv)
    z_new = r[5] + dp1 / rhoinv * (theta + dasin)
    
    return SVector{6,T}(x_new, px_new, y_new, r[4], z_new, r[6])
end

"""
    exact_bend_body(r, irho, L, beti) -> SVector{6,T}

Exact bend body map (Forest 12.18).
"""
@inline function exact_bend_body(r::SVector{6,T}, irho::T, L::T, beti::T) where T
    dp1 = beti + r[6]
    pz = pxyz(dp1, r[2], r[4])
    
    if abs(irho) < 1e-6
        # Drift limit
        NormL = L / pz
        return SVector{6,T}(
            r[1] + r[2] * NormL,
            r[2],
            r[3] + r[4] * NormL,
            r[4],
            r[5] + NormL * dp1 - L * beti, # Path length diff
            r[6]
        )
    else
        pzmx = pz - (1 + r[1] * irho)
        cs = cos(irho * L)
        sn = sin(irho * L)
        
        px_new = r[2] * cs + pzmx * sn
        
        d2 = pxyz(dp1, 0.0, r[4])
        val1 = r[2] / d2
        val2 = px_new / d2
        val1 = clamp(val1, -one(T), one(T))
        val2 = clamp(val2, -one(T), one(T))
        
        dasin = L + (asin(val1) - asin(val2)) / irho
        
        x_new = (pxyz(dp1, px_new, r[4]) - pzmx * cs + r[2] * sn - 1) / irho
        y_new = r[3] + r[4] * dasin
        z_new = r[5] + dp1 * dasin - L * beti
        
        return SVector{6,T}(x_new, px_new, y_new, r[4], z_new, r[6])
    end
end

"""
    pass!(elem::ExactSBend, r, beti)

Track through Exact Sector Bend.
"""
function pass!(elem::ExactSBend{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    # Entrance Misalignment
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end
    
    irho = elem.angle / elem.L
    
    # Coordinate Rotation (Entrance)
    r = yrot(r, elem.e1, beti)
    
    # Entrance Fringe
    if elem.fringe_bend_entrance != 0
        r = bend_fringe(r, irho, elem.gk, beti)
    end
    
    # Entrance Edge
    r = bend_edge(r, irho, -elem.e1, beti)
    
    # Integrator (Body)
    # Drift-Kick-Drift using exact_bend_body
    if elem.num_int_steps == 0
        r = exact_bend_body(r, irho, elem.L, beti)
    else
        SL = elem.L / elem.num_int_steps
        L1 = SL * T(DRIFT1)
        L2 = SL * T(DRIFT2)
        K1 = SL * T(KICK1)
        K2 = SL * T(KICK2)
        
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

        for _ in 1:elem.num_int_steps
            r = exact_bend_body(r, irho, L1, beti)
            r = strthinkick(r, polynom_a, polynom_b, K1, elem.max_order)
            r = exact_bend_body(r, irho, L2, beti)
            r = strthinkick(r, polynom_a, polynom_b, K2, elem.max_order)
            r = exact_bend_body(r, irho, L2, beti)
            r = strthinkick(r, polynom_a, polynom_b, K1, elem.max_order)
            r = exact_bend_body(r, irho, L1, beti)
        end
    end
    
    # Exit Edge
    r = bend_edge(r, irho, -elem.e2, beti)
    
    # Exit Fringe
    if elem.fringe_bend_exit != 0
        r = bend_fringe(r, -irho, elem.gk, beti)
    end
    
    # Coordinate Rotation (Exit)
    r = yrot(r, elem.e2, beti)
    
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

@inline _nan6(::Type{T}) where T = SVector{6, T}(ntuple(_ -> T(NaN), 6))

function pass!(elem::DriftSC{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
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

function pass!(elem::QuadrupoleSC{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
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

function pass!(elem::SextupoleSC{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
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

function pass!(elem::OctupoleSC{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
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

function pass!(elem::SBendSC{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end

    if iszero(elem.L)
        return _nan6(T)
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

@inline function _lbend_body(r::SVector{6,T}, L::T, grd::T, b_angle::T, by_error::T) where T
    if iszero(L)
        return r
    end

    p_norm = inv(one(T) + r[6])
    Kx = b_angle / L
    G1 = (Kx * Kx + grd) * p_norm
    G2 = -grd * p_norm
    tol = sqrt(eps(T))

    MHD = one(T); M12 = L; M21 = zero(T)
    MVD = one(T); M34 = L; M43 = zero(T)
    arg1 = zero(T); arg2 = zero(T)
    sqrtG1 = zero(T); sqrtG2 = zero(T)

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
    delta = r[6]
    dterm = delta * p_norm - by_error

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

    z_new = r[5] + xpr * xpr * (L + MHD * M12) / 4
    if abs(G1) >= tol
        z_new += (L - MHD * M12) * (x * x * G1 + dterm * dterm * Kx * Kx / G1 - 2 * x * Kx * dterm) / 4
        z_new += M12 * M21 * (x * xpr - xpr * dterm * Kx / G1) / 2
        z_new += Kx * x * M12 + xpr * (one(T) - MHD) * Kx / G1 + dterm * (L - M12) * Kx * Kx / G1
    end
    z_new += ((L - MVD * M34) * y * y * G2 + ypr * ypr * (L + MVD * M34)) / 4
    z_new += M34 * M43 * y * ypr / 2

    return SVector{6, T}(x_new, px_new, y_new, py_new, z_new, r[6])
end

function pass!(elem::LBend{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end

    if iszero(elem.L)
        return _nan6(T)
    end
    irho = elem.angle / elem.L
    r = edge_fringe_entrance(r, irho, elem.e1, elem.fint1, elem.full_gap, 1)
    r = _lbend_body(r, elem.L, elem.K, elem.angle, elem.by_error)
    r = edge_fringe_exit(r, irho, elem.e2, elem.fint2, elem.full_gap, 1)

    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    return apply_misalignment(r, elem.t2)
end

# =============================================================================
# Auxiliary Canonical Elements
# =============================================================================

function pass!(elem::SpaceCharge{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    # TrackPad's single-particle API has no bunch moments/current.
    # Keep SPACECHARGE as a no-op here (JuTrack also gives zero kick at I=0).
    return r
end

function pass!(elem::Translation{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    pz2 = one(T) + 2 * r[6] * beti + r[6]^2 - r[2]^2 - r[4]^2
    if pz2 <= zero(T)
        return _nan6(T)
    end
    pz = sqrt(pz2)
    x_new = r[1] - (elem.dx + elem.ds * r[2] / pz)
    y_new = r[3] - (elem.dy + elem.ds * r[4] / pz)
    z_new = r[5] + elem.ds * (beti + r[6]) / pz
    return SVector{6, T}(x_new, r[2], y_new, r[4], z_new, r[6])
end

function pass!(elem::YRotation{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    angle = -elem.angle
    if iszero(angle)
        return r
    end
    ca = cos(angle)
    sa = sin(angle)
    ta = tan(angle)

    pz2 = one(T) + 2 * r[6] * beti + r[6]^2 - r[2]^2 - r[4]^2
    if pz2 <= zero(T)
        return _nan6(T)
    end
    pz = sqrt(pz2)
    ptt = one(T) - ta * r[2] / pz
    if abs(ptt) < sqrt(eps(T))
        return _nan6(T)
    end

    x_new = r[1] / (ca * ptt)
    px_new = ca * r[2] + sa * pz
    y_new = r[3] + ta * r[1] * r[4] / (pz * ptt)
    z_new = r[5] + ta * r[1] * (beti + r[6]) / (pz * ptt)
    return SVector{6, T}(x_new, px_new, y_new, r[4], z_new, r[6])
end

# =============================================================================
# Wiggler
# =============================================================================

@inline function _sinc_taylor(x::T) where T
    x2 = x * x
    return one(T) - x2 / 6 * (one(T) - x2 / 20 * (one(T) - x2 / 42 * (one(T) - x2 / 72)))
end

@inline function _wig_ax_axpy(elem::Wiggler{T,N,V}, r::SVector{6,T}, Zw::T, Aw::T, Po::T) where {T,N,V}
    x = r[1]
    y = r[3]
    kw = T(2pi) / elem.lw
    ax = zero(T)
    axpy = zero(T)

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

@inline function _wig_ay_aypx(elem::Wiggler{T,N,V}, r::SVector{6,T}, Zw::T, Aw::T, Po::T) where {T,N,V}
    x = r[1]
    y = r[3]
    kw = T(2pi) / elem.lw
    ay = zero(T)
    aypx = zero(T)

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

@inline function _wig_map_2nd(elem::Wiggler{T,N,V}, r::SVector{6,T}, dl::T, Zw::T, Aw::T, Po::T) where {T,N,V}
    delta = r[6]
    inv1pd = inv(one(T) + delta)
    dld = dl * inv1pd
    dl2 = dl / 2
    dl2d = dl2 * inv1pd

    Zw += dl2
    x = r[1]
    px = r[2]
    y = r[3]
    py = r[4]
    z = r[5]

    ay, aypx = _wig_ay_aypx(elem, SVector{6,T}(x, px, y, py, z, delta), Zw, Aw, Po)
    px -= aypx
    py -= ay
    y += dl2d * py
    z += (dl2d / 2) * py^2 * inv1pd

    ay, aypx = _wig_ay_aypx(elem, SVector{6,T}(x, px, y, py, z, delta), Zw, Aw, Po)
    px += aypx
    py += ay

    ax, axpy = _wig_ax_axpy(elem, SVector{6,T}(x, px, y, py, z, delta), Zw, Aw, Po)
    px -= ax
    py -= axpy
    x += dld * px
    z += (dld / 2) * px^2 * inv1pd

    ax, axpy = _wig_ax_axpy(elem, SVector{6,T}(x, px, y, py, z, delta), Zw, Aw, Po)
    px += ax
    py += axpy

    ay, aypx = _wig_ay_aypx(elem, SVector{6,T}(x, px, y, py, z, delta), Zw, Aw, Po)
    px -= aypx
    py -= ay
    y += dl2d * py
    z += (dl2d / 2) * py^2 * inv1pd

    ay, aypx = _wig_ay_aypx(elem, SVector{6,T}(x, px, y, py, z, delta), Zw, Aw, Po)
    px += aypx
    py += ay
    Zw += dl2

    return SVector{6,T}(x, px, y, py, z, delta), Zw
end

@inline function _wig_pass_4th(elem::Wiggler{T,N,V}, r::SVector{6,T}) where {T,N,V}
    PN = elem.Nsteps
    Nw = round(Int, elem.L / elem.lw)
    Nstep = PN * max(1, Nw)
    dl = elem.lw / T(PN)
    dl1 = dl * T(1.3512071919596573)
    dl0 = dl * T(-1.7024143839193146)

    Po = sqrt((elem.energy / T(M_ELECTRON))^2 - one(T))
    Aw = T(1e-9) * T(C_LIGHT) / (T(M_ELECTRON) * T(1e-9)) / T(2pi) * elem.lw * elem.Bmax
    Zw = zero(T)

    @inbounds for _ in 1:Nstep
        r, Zw = _wig_map_2nd(elem, r, dl1, Zw, Aw, Po)
        r, Zw = _wig_map_2nd(elem, r, dl0, Zw, Aw, Po)
        r, Zw = _wig_map_2nd(elem, r, dl1, Zw, Aw, Po)
    end
    return r
end

@inline function _wig_B(elem::Wiggler{T,N,V}, r::SVector{6,T}, Zw::T) where {T,N,V}
    x = r[1]
    y = r[3]
    kw = T(2pi) / elem.lw
    PB0 = elem.Bmax
    Bxv = zero(T)
    Byv = zero(T)
    Bzv = zero(T)

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

@inline function _wig_radiation_kicks(r::SVector{6,T}, Bxv::T, Byv::T, Po::T, srCoef::T, dl::T) where T
    B2 = Bxv^2 + Byv^2
    if iszero(B2)
        return r
    end
    H = Po * T(M_ELECTRON) / T(C_LIGHT)
    irho2 = B2 / (H * H)
    dFactor = (one(T) + r[6])^2
    dDelta = -srCoef * dFactor * irho2 * dl
    scale = one(T) + dDelta
    return SVector{6,T}(r[1], r[2] * scale, r[3], r[4] * scale, r[5], r[6] + dDelta)
end

@inline function _wig_pass_4th_rad(elem::Wiggler{T,N,V}, r::SVector{6,T}) where {T,N,V}
    PN = elem.Nsteps
    Nw = round(Int, elem.L / elem.lw)
    Nstep = PN * max(1, Nw)
    SL = elem.lw / T(PN)
    dl1 = SL * T(1.3512071919596573)
    dl0 = SL * T(-1.7024143839193146)

    gamma = elem.energy / T(M_ELECTRON)
    Po = sqrt(gamma^2 - one(T))
    Aw = T(1e-9) * T(C_LIGHT) / (T(M_ELECTRON) * T(1e-9)) / T(2pi) * elem.lw * elem.Bmax
    srCoef = T(2.0 / 3.0) * T(2.8179403205e-15) * gamma^3
    Zw = zero(T)

    ax, _ = _wig_ax_axpy(elem, r, Zw, Aw, Po)
    ay, _ = _wig_ay_aypx(elem, r, Zw, Aw, Po)
    r = SVector{6,T}(r[1], r[2] - ax, r[3], r[4] - ay, r[5], r[6])
    Bxv, Byv, _ = _wig_B(elem, r, Zw)
    r = _wig_radiation_kicks(r, Bxv, Byv, Po, srCoef, SL)
    r = SVector{6,T}(r[1], r[2] + ax, r[3], r[4] + ay, r[5], r[6])

    @inbounds for _ in 1:Nstep
        r, Zw = _wig_map_2nd(elem, r, dl1, Zw, Aw, Po)
        r, Zw = _wig_map_2nd(elem, r, dl0, Zw, Aw, Po)
        r, Zw = _wig_map_2nd(elem, r, dl1, Zw, Aw, Po)

        ax, _ = _wig_ax_axpy(elem, r, Zw, Aw, Po)
        ay, _ = _wig_ay_aypx(elem, r, Zw, Aw, Po)
        r = SVector{6,T}(r[1], r[2] - ax, r[3], r[4] - ay, r[5], r[6])
        Bxv, Byv, _ = _wig_B(elem, r, Zw)
        r = _wig_radiation_kicks(r, Bxv, Byv, Po, srCoef, SL)
        r = SVector{6,T}(r[1], r[2] + ax, r[3], r[4] + ay, r[5], r[6])
    end

    return r
end

function pass!(elem::Wiggler{T,N,V}, r::SVector{6,T}, beti::T=one(T)) where {T,N,V}
    r = apply_misalignment(r, elem.t1)
    if !iszero(elem.r1)
        r = elem.r1 * r
    end

    if elem.rad == 1
        r = _wig_pass_4th_rad(elem, r)
    else
        r = _wig_pass_4th(elem, r)
    end

    if !iszero(elem.r2)
        r = elem.r2 * r
    end
    return apply_misalignment(r, elem.t2)
end

# =============================================================================
# Crab / Accelerating Cavities
# =============================================================================

function pass!(elem::CrabCavity{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    volt = elem.volt * (one(T) + elem.errors[1])
    phi = elem.phi + elem.errors[2]
    E = max(abs(elem.energy), eps(T))
    ang = elem.k * r[5] + phi

    if elem.L > zero(T)
        r = drift6(r, elem.L / 2, beti)
    end
    px_new = r[2] + (volt / E) * sin(ang * beti)
    delta_new = r[6] - (elem.k * volt / E * beti) * r[1] * cos(ang * beti)
    r = SVector{6, T}(r[1], px_new, r[3], r[4], r[5], delta_new)
    if elem.L > zero(T)
        r = drift6(r, elem.L / 2, beti)
    end
    return r
end

function pass!(elem::AccelCavity{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    beta = inv(beti)
    beta2 = beta * beta
    if beta2 <= eps(T)
        return r
    end
    E = max(abs(elem.energy), eps(T))
    sv = sin(elem.k * r[5] + elem.phis) - sin(elem.phis)
    delta_new = r[6] + (elem.volt / (beta2 * E)) * sv
    return SVector{6, T}(r[1], r[2], r[3], r[4], r[5], delta_new)
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

function pass!(elem::LongitudinalRFMap{T,E}, r::SVector{6,T}, beti::T=one(T)) where {T,E}
    k = T(_rf_k(elem.rf))
    if abs(k) <= eps(T)
        return r
    end
    h = T(_rf_h(elem.rf))
    beta = inv(beti)
    eta = elem.alphac - (one(T) - beta * beta)
    z_new = r[5] - (T(2pi) * h * eta / k) * r[6]
    return SVector{6, T}(r[1], r[2], r[3], r[4], z_new, r[6])
end

function pass!(elem::LorentzBoost{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    if elem.mode != 0
        return r
    end
    invcos = inv(elem.cosang)
    x_new = r[1] + elem.tanang * r[5]
    delta_new = r[6] - elem.tanang * r[2]
    return SVector{6, T}(x_new, r[2] * invcos, r[3], r[4] * invcos, r[5] * invcos, delta_new)
end

function pass!(elem::InvLorentzBoost{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    if elem.mode != 0
        return r
    end
    x_new = r[1] - elem.sinang * r[5]
    delta_new = r[6] + elem.sinang * r[2]
    return SVector{6, T}(x_new, r[2] * elem.cosang, r[3], r[4] * elem.cosang, r[5] * elem.cosang, delta_new)
end

# =============================================================================
# Strong Beam-Beam (single-particle approximations)
# =============================================================================

function pass!(elem::StrongThinGaussianBeam{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    sx2 = max(elem.rmssizex^2, eps(T))
    sy2 = max(elem.rmssizey^2, eps(T))
    dx = r[1] - elem.xoffset
    dy = r[3] - elem.yoffset
    g = elem.amplitude * exp(-T(0.5) * (dx^2 / sx2 + dy^2 / sy2))
    px_new = r[2] - g * dx / sx2
    py_new = r[4] - g * dy / sy2
    return SVector{6, T}(r[1], px_new, r[3], py_new, r[5], r[6])
end

function pass!(elem::StrongGaussianBeam{T,N,V}, r::SVector{6,T}, beti::T=one(T)) where {T,N,V}
    if elem.nzslice <= 0
        return r
    end
    sx2 = max(elem.beamsize[1]^2, eps(T))
    sy2 = max(elem.beamsize[2]^2, eps(T))
    factor = elem.charge / max(abs(elem.total_energy), eps(T))
    px = r[2]
    py = r[4]
    @inbounds for i in 1:elem.nzslice
        w = i <= length(elem.zslice_npar) ? elem.zslice_npar[i] : one(T) / T(elem.nzslice)
        xoff = i <= length(elem.xoffsets) ? elem.xoffsets[i] : zero(T)
        yoff = i <= length(elem.yoffsets) ? elem.yoffsets[i] : zero(T)
        dx = r[1] - xoff
        dy = r[3] - yoff
        g = w * exp(-T(0.5) * (dx^2 / sx2 + dy^2 / sy2))
        px -= factor * g * dx / sx2
        py -= factor * g * dy / sy2
    end
    return SVector{6, T}(r[1], px, r[3], py, r[5], r[6])
end

# =============================================================================
# Longitudinal Wake models
# =============================================================================

function pass!(elem::LongitudinalRLCWake{T,N}, r::SVector{6,T}, beti::T=one(T)) where {T,N}
    if iszero(elem.scale)
        return r
    end
    t = min(r[5] / T(C_LIGHT), zero(T))
    delta_new = r[6] - elem.scale * wakefieldfunc_RLCWake(elem, t)
    return SVector{6, T}(r[1], r[2], r[3], r[4], r[5], delta_new)
end

function pass!(elem::LongitudinalWake{T,N,V}, r::SVector{6,T}, beti::T=one(T)) where {T,N,V}
    if iszero(elem.scale)
        return r
    end
    t = r[5] / T(C_LIGHT)
    delta_new = r[6] - elem.scale * wakefieldfunc(elem, t)
    return SVector{6, T}(r[1], r[2], r[3], r[4], r[5], delta_new)
end
