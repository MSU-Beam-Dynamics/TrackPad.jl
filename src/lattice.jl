"""
    lattice.jl

Lattice representation and high-level tracking functions for TrackPad.jl.
"""

using StaticArrays

export Lattice, Beam, isperiodic, linepass!, ringpass!

# =============================================================================
# Beam Definition
# =============================================================================

"""
    Beam{T}

Lightweight beam structure for TrackPad.jl.
Unlike JuTrack's heavy Beam struct, this focuses on essential tracking parameters.

# Fields
- `energy::T`: Reference kinetic energy in eV
- `mass::T`: Particle mass in eV (default: electron mass)
- `charge::T`: Particle charge (default: -1.0 for electrons)
- `gamma::T`: Relativistic gamma factor
- `beta::T`: Relativistic velocity (v/c)

# Example
```julia
beam = Beam(1.0e9)  # 1 GeV electron beam
```
"""
struct Beam{T}
    energy::T      # Reference energy [eV]
    mass::T        # Particle mass [eV]
    charge::T      # Particle charge
    gamma::T       # Lorentz factor
    beta::T        # v/c
end

"""Electron rest-mass energy in eV."""
const M_ELECTRON = 0.51099895069e6  # eV (electron rest mass energy)

"""Proton rest-mass energy in eV."""
const M_PROTON = 938.27208816e6     # eV (proton rest mass energy)

"""
    Beam(energy; mass=M_ELECTRON, charge=-1.0)

Construct a beam with reference kinetic energy `energy` in eV. `mass` is the
rest-mass energy in eV and `charge` is signed in units of elementary charge.
"""
function Beam(energy::T; mass::T=T(M_ELECTRON), charge::T=T(-1.0)) where T
    gamma = (energy + mass) / mass
    beta = sqrt(one(T) - one(T) / gamma^2)
    return Beam{T}(energy, mass, charge, gamma, beta)
end

"""
    beti(beam::Beam)

Return 1/β for the beam (used in tracking for relativistic correction).
"""
@inline beti(beam::Beam{T}) where T = one(T) / beam.beta

# =============================================================================
# Lattice Definition
# =============================================================================

"""
    Lattice{E}

A lattice is an ordered sequence of accelerator elements with one of two
boundary conditions: an open line (`periodic=false`) or a closed ring
(`periodic=true`).

# Fields
- `elements::Vector{E}`: Vector of elements (all subtypes of AbstractElement)
- `name::Symbol`: Optional lattice name
- `periodic::Bool`: Whether the end of the lattice closes at its start

# Example
```julia
# Create elements
d1 = Drift(1.0)
q1 = Quadrupole(0.5, 0.5)
q2 = Quadrupole(0.5, -0.5)

# Build FODO cell
fodo = Lattice([d1, q1, d1, q2]; periodic=true)
```
"""
struct Lattice{E<:AbstractElement}
    elements::Vector{E}
    name::Symbol
    periodic::Bool
end

Lattice(elements::Vector{E}; name::Symbol=:LATTICE, periodic::Bool=false) where {E<:AbstractElement} =
    Lattice{E}(elements, name, periodic)

# Allow construction from any iterable of elements
function Lattice(elements; name::Symbol=:LATTICE, periodic::Bool=false)
    elem_vec = collect(elements)
    return Lattice(elem_vec; name=name, periodic=periodic)
end

"""Return `true` when the lattice has periodic ring boundary conditions."""
isperiodic(lat::Lattice) = lat.periodic

@inline function _require_periodic(lat::Lattice, operation::AbstractString)
    lat.periodic || throw(ArgumentError(
        "$operation requires a periodic lattice; construct it with periodic=true",
    ))
    return nothing
end

Base.length(lat::Lattice) = length(lat.elements)
Base.getindex(lat::Lattice, i) = lat.elements[i]
Base.iterate(lat::Lattice) = iterate(lat.elements)
Base.iterate(lat::Lattice, state) = iterate(lat.elements, state)
Base.eachindex(lat::Lattice) = eachindex(lat.elements)

"""
    total_length(lat::Lattice; time=0.0, turn=0)

Return the total length of the lattice.
For time-varying elements, length is evaluated at `time`.
"""
function total_length(lat::Lattice; time::Real=0.0, turn::Integer=0)
    ctx = TimeContext(Float64(time); turn=turn)
    L = zero(Float64)
    for elem in lat.elements
        L += Float64(get_length(_resolve_for_time(elem, ctx)))
    end
    return L
end

"""
    materialize_lattice(lat; time=0.0, turn=0)

Resolve all time-varying elements at `(time, turn)` and return a static lattice.
Use this before GPU adaptation so kernels only see concrete element structs.
"""
function materialize_lattice(lat::Lattice; time::Real=0.0, turn::Integer=0)
    ctx = TimeContext(Float64(time); turn=turn)
    elems = [materialize(elem, ctx) for elem in lat.elements]
    return Lattice(elems; name=lat.name, periodic=lat.periodic)
end

"""
    get_length(elem::AbstractElement)

Get the length of an element.
"""
get_length(elem::Drift) = elem.L
get_length(elem::Quadrupole) = elem.L
get_length(elem::Sextupole) = elem.L
get_length(elem::Octupole) = elem.L
get_length(elem::SBend) = elem.L
get_length(elem::SBendSC) = elem.L
get_length(elem::LBend) = elem.L
get_length(elem::RFCavity) = elem.L
get_length(elem::ThinMultipole) = elem.L
get_length(elem::Solenoid) = elem.L
get_length(elem::Corrector) = elem.L
get_length(elem::ExactSBend) = elem.L
get_length(elem::Marker) = zero(Float64)
get_length(elem::Patch) = zero(Float64)
get_length(elem::DriftSC) = elem.L
get_length(elem::QuadrupoleSC) = elem.L
get_length(elem::SextupoleSC) = elem.L
get_length(elem::OctupoleSC) = elem.L
get_length(elem::SpaceCharge) = elem.L
get_length(elem::Translation) = elem.L
get_length(elem::YRotation) = elem.L
get_length(elem::Wiggler) = elem.L
get_length(elem::CrabCavity) = elem.L
get_length(elem::AccelCavity) = elem.L
get_length(elem::LongitudinalRFMap) = zero(Float64)
get_length(elem::LorentzBoost) = zero(Float64)
get_length(elem::InvLorentzBoost) = zero(Float64)
get_length(elem::StrongThinGaussianBeam) = zero(Float64)
get_length(elem::StrongGaussianBeam) = zero(Float64)
get_length(elem::LongitudinalRLCWake) = zero(Float64)
get_length(elem::LongitudinalWake) = zero(Float64)
get_length(elem::TimeVaryingElement) = get_length(elem.base)

# =============================================================================
# Single Particle Tracking
# =============================================================================

"""
    linepass(lat::Lattice, r::SVector{6,T}, beam::Beam) -> SVector{6,T}

Track a single particle through the lattice (immutable version).

# Arguments
- `lat`: Lattice to track through
- `r`: Initial 6D phase space coordinates
- `beam`: Beam parameters

# Returns
- Final 6D coordinates after tracking
"""
function linepass(lat::Lattice, r::SVector{6,S}, beam::Beam{T};
                  time::Real=zero(T), turn::Integer=0) where {T,S}
    β_inv = beti(beam)
    ctx = TimeContext(T(time); turn=turn)
    for elem in lat.elements
        elem_now = _resolve_for_time(elem, ctx)
        r = pass!(elem_now, r, β_inv)
        if check_lost(r)
            return r  # Return immediately if particle is lost
        end
    end
    return r
end

"""
    linepass(lat::Lattice, r::SVector{6,T}) -> SVector{6,T}

Track a single particle through the lattice using default beam (1 GeV electron).
"""
function linepass(lat::Lattice, r::SVector{6,S}; time::Real=zero(Float64), turn::Integer=0) where S
    beam = Beam(1.0e9)
    return linepass(lat, r, beam; time=time, turn=turn)
end

"""
    ringpass(lat::Lattice, r::SVector{6,T}, beam::Beam, nturns::Int) -> SVector{6,T}

Track a single particle for multiple turns through a ring lattice.

# Arguments
- `lat`: Ring lattice
- `r`: Initial coordinates
- `beam`: Beam parameters
- `nturns`: Number of turns

# Returns
- Final coordinates after all turns
"""
function ringpass(lat::Lattice, r::SVector{6,S}, beam::Beam{T}, nturns::Int;
                  time::Real=zero(T), dt_turn::Real=zero(T), turn::Integer=0) where {T,S}
    _require_periodic(lat, "ringpass")
    nturns >= 0 || throw(ArgumentError("nturns must be nonnegative"))
    t = T(time)
    dt = T(dt_turn)
    trn = Int(turn)
    for _ in 1:nturns
        r = linepass(lat, r, beam; time=t, turn=trn)
        if check_lost(r)
            return r
        end
        t += dt
        trn += 1
    end
    return r
end

# =============================================================================
# Multi-Particle Tracking (Matrix-based for performance)
# =============================================================================

"""
    linepass!(coords::Matrix{T}, lat::Lattice, beam::Beam, lost_flags::Vector{Int}) -> Nothing

Track multiple particles through a lattice (in-place, mutating version).

# Arguments
- `coords`: Nparticles × 6 matrix of coordinates (mutated in-place)
- `lat`: Lattice to track through
- `beam`: Beam parameters
- `lost_flags`: Vector of lost flags (0=alive, 1=lost)

# Note
This function provides JuTrack-compatible interface for multi-particle tracking.
"""
function linepass!(coords::Matrix{T}, lat::Lattice, beam::Beam{T},
                   lost_flags::Vector{Int}; time::Real=zero(T), turn::Integer=0) where T
    nparticles = size(coords, 1)
    β_inv = beti(beam)
    ctx = TimeContext(T(time); turn=turn)
    
    for elem in lat.elements
        elem_now = _resolve_for_time(elem, ctx)
        # Collective elements need the whole bunch, not one particle at a time.
        if elem_now isa Union{LongitudinalRLCWake, LongitudinalWake}
            _apply_longitudinal_wake!(coords, lost_flags, nparticles, elem_now)
            continue
        end
        rap, eap = _elem_apertures(elem_now)
        check_aperture = _apertures_active(rap, eap)
        for i in 1:nparticles
            if lost_flags[i] == 1
                continue
            end

            # Extract particle coordinates as SVector
            r = SVector{6,T}(coords[i, 1], coords[i, 2], coords[i, 3],
                             coords[i, 4], coords[i, 5], coords[i, 6])

            # Track through element
            r_new = pass!(elem_now, r, β_inv)

            # Check if lost
            if check_lost(r_new)
                lost_flags[i] = 1
                continue
            end

            # Store back
            coords[i, 1] = r_new[1]
            coords[i, 2] = r_new[2]
            coords[i, 3] = r_new[3]
            coords[i, 4] = r_new[4]
            coords[i, 5] = r_new[5]
            coords[i, 6] = r_new[6]

            # Aperture loss keeps the evolved coordinates (JuTrack semantics).
            check_aperture && outside_aperture(r_new, rap, eap) && (lost_flags[i] = 1)
        end
    end
    
    return nothing
end

"""
    ringpass!(coords::Matrix{T}, lat::Lattice, beam::Beam, 
              lost_flags::Vector{Int}, nturns::Int) -> Nothing

Track multiple particles for multiple turns through a ring (in-place).
"""
function ringpass!(coords::Matrix{T}, lat::Lattice, beam::Beam{T},
                   lost_flags::Vector{Int}, nturns::Int;
                   time::Real=zero(T), dt_turn::Real=zero(T), turn::Integer=0) where T
    _require_periodic(lat, "ringpass!")
    nturns >= 0 || throw(ArgumentError("nturns must be nonnegative"))
    t = T(time)
    dt = T(dt_turn)
    trn = Int(turn)
    for _ in 1:nturns
        linepass!(coords, lat, beam, lost_flags; time=t, turn=trn)
        t += dt
        trn += 1
    end
    return nothing
end

# =============================================================================
# Collective-element helpers
# =============================================================================

"""
    physical_wake_scale(beam::Beam, bunch_charge::Real, nmacro::Integer)

Recommended `scale` for `LongitudinalRLCWake`/`LongitudinalWake` when the
Green function carries physical units (V/C, e.g. an `Rshunt` in ohms) and a
bunch of total charge `bunch_charge` [C] is represented by `nmacro` equal
macroparticles:

    scale = qhat * Qmacro / (P0*c)
    Qmacro = bunch_charge / nmacro

Here `qhat = beam.charge` is the signed reference-particle charge in elementary
charge units, `Qmacro` is the signed source-macroparticle charge in C, and
`P0*c = beta0*(energy + mass)` is in eV. Since a V/C Green function multiplied
by `Qmacro` gives volts, multiplication by `qhat` gives the test particle's
energy change in eV. For a bunch made of the reference species,
`bunch_charge` and `beam.charge` have the same sign, giving a positive scale
and a decelerating kick through `delta -= scale * V(z)`.
"""
function physical_wake_scale(beam::Beam{T}, bunch_charge::Real,
                             nmacro::Integer) where T
    nmacro > 0 || throw(ArgumentError("nmacro must be positive"))
    isfinite(bunch_charge) || throw(ArgumentError("bunch_charge must be finite"))
    isfinite(beam.charge) || throw(ArgumentError("beam charge must be finite"))
    p0c = beam.beta * (beam.energy + beam.mass)
    isfinite(p0c) && p0c > zero(T) ||
        throw(ArgumentError("reference beam momentum P0*c must be finite and positive"))
    qmacro = T(bunch_charge) / T(nmacro)
    return beam.charge * qmacro / p0c
end

# =============================================================================
# Convenience Functions
# =============================================================================

"""
    spos(lat::Lattice; time=0.0) -> Vector{Float64}

Return the s-position (longitudinal position) of each element in the lattice.
For time-varying elements, positions are evaluated at `time`.
"""
function spos(lat::Lattice; time::Real=0.0, turn::Integer=0)
    ctx = TimeContext(Float64(time); turn=turn)
    s = zeros(Float64, length(lat) + 1)
    for (i, elem) in enumerate(lat.elements)
        s[i+1] = s[i] + Float64(get_length(_resolve_for_time(elem, ctx)))
    end
    return s
end

@inline function _element_name(elem)
    return hasproperty(elem, :name) ? getproperty(elem, :name) : nothing
end

@inline function _element_name(elem::TimeVaryingElement)
    return hasproperty(elem.base, :name) ? getproperty(elem.base, :name) : nothing
end

"""
    findelem(lat::Lattice, name::Symbol) -> Vector{Int}

Find indices of elements with matching name.
"""
function findelem(lat::Lattice, name::Symbol)
    indices = Int[]
    for (i, elem) in enumerate(lat.elements)
        elem_name = _element_name(elem)
        if elem_name == name
            push!(indices, i)
        end
    end
    return indices
end
