"""
    lattice.jl

Lattice representation and high-level tracking functions for TrackPad.jl.
"""

using StaticArrays

export Lattice, Beam, isperiodic, refine_lattice, linepass!, ringpass!

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
function Beam(energy::T; mass::Real=T(M_ELECTRON), charge::Real=T(-1.0)) where T
    mass = T(mass)
    charge = T(charge)
    gamma = (energy + mass) / mass
    beta = sqrt(one(T) - one(T) / gamma^2)
    return Beam{T}(energy, mass, charge, gamma, beta)
end

"""
    classical_radius(beam::Beam) -> r0 [m]

Classical radius of the beam's particle species.
"""
classical_radius(beam::Beam) = classical_radius(beam.mass, beam.charge)

"""
    beambeam_amplitude(weak::Beam, num_particle, strong_charge) -> amplitude

`amplitude = N r0_w q_w q_s / γ_w` for [`StrongThinGaussianBeam`](@ref): the kick
felt by the weak (tracked) beam `weak` from `num_particle` strong-beam
particles of charge `strong_charge` (units of `e`). Positive for like charges.
"""
function beambeam_amplitude(weak::Beam, num_particle::Real, strong_charge::Real)
    return num_particle * classical_radius(weak) * weak.charge * strong_charge / weak.gamma
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
Base.firstindex(lat::Lattice) = firstindex(lat.elements)
Base.lastindex(lat::Lattice) = lastindex(lat.elements)
Base.iterate(lat::Lattice) = iterate(lat.elements)
Base.iterate(lat::Lattice, state) = iterate(lat.elements, state)
Base.eachindex(lat::Lattice) = eachindex(lat.elements)

@inline function _replace_element_fields(elem::E, changes::NamedTuple) where E
    values = ntuple(fieldcount(E)) do i
        field = fieldname(E, i)
        hasproperty(changes, field) ? getproperty(changes, field) : getfield(elem, i)
    end
    return E(values...)
end

@inline function _slice_kick_angle(kick_angle::SVector{2,T}, count::Int) where T
    # Tracking uses sin(kick_angle) / L, so preserve that field after L is split.
    return SVector{2,T}(asin(sin(kick_angle[1]) / count),
                        asin(sin(kick_angle[2]) / count))
end

const _SLICED_DRIFT = Union{Drift,DriftSC}

function _slice_drift(elem::_SLICED_DRIFT, count::Int)
    return [_replace_element_fields(elem, (
        L = elem.L / count,
        t1 = i == 1 ? elem.t1 : zero(elem.t1),
        t2 = i == count ? elem.t2 : zero(elem.t2),
        r1 = i == 1 ? elem.r1 : zero(elem.r1),
        r2 = i == count ? elem.r2 : zero(elem.r2),
    )) for i in 1:count]
end

const _SLICED_MULTIPOLE = Union{
    Quadrupole,Sextupole,Octupole,
    QuadrupoleSC,SextupoleSC,OctupoleSC,
}

function _slice_multipole(elem::_SLICED_MULTIPOLE, count::Int)
    kick_angle = _slice_kick_angle(elem.kick_angle, count)
    return [_replace_element_fields(elem, (
        L = elem.L / count,
        num_int_steps = 1,
        fringe_entrance = i == 1 ? elem.fringe_entrance : 0,
        fringe_exit = i == count ? elem.fringe_exit : 0,
        t1 = i == 1 ? elem.t1 : zero(elem.t1),
        t2 = i == count ? elem.t2 : zero(elem.t2),
        r1 = i == 1 ? elem.r1 : zero(elem.r1),
        r2 = i == count ? elem.r2 : zero(elem.r2),
        kick_angle = kick_angle,
    )) for i in 1:count]
end

const _SLICED_BEND = Union{SBend,ExactSBend,SBendSC}

function _slice_bend(elem::_SLICED_BEND, count::Int)
    kick_angle = _slice_kick_angle(elem.kick_angle, count)
    return [_replace_element_fields(elem, (
        L = elem.L / count,
        angle = elem.angle / count,
        e1 = i == 1 ? elem.e1 : zero(elem.e1),
        e2 = i == count ? elem.e2 : zero(elem.e2),
        num_int_steps = elem.num_int_steps == 0 ? 0 : 1,
        fint1 = i == 1 ? elem.fint1 : zero(elem.fint1),
        fint2 = i == count ? elem.fint2 : zero(elem.fint2),
        fringe_bend_entrance = i == 1 ? elem.fringe_bend_entrance : 0,
        fringe_bend_exit = i == count ? elem.fringe_bend_exit : 0,
        fringe_quad_entrance = i == 1 ? elem.fringe_quad_entrance : 0,
        fringe_quad_exit = i == count ? elem.fringe_quad_exit : 0,
        t1 = i == 1 ? elem.t1 : zero(elem.t1),
        t2 = i == count ? elem.t2 : zero(elem.t2),
        r1 = i == 1 ? elem.r1 : zero(elem.r1),
        r2 = i == count ? elem.r2 : zero(elem.r2),
        kick_angle = kick_angle,
    )) for i in 1:count]
end

function _slice_lbend(elem::LBend, count::Int)
    return [_replace_element_fields(elem, (
        L = elem.L / count,
        angle = elem.angle / count,
        e1 = i == 1 ? elem.e1 : zero(elem.e1),
        e2 = i == count ? elem.e2 : zero(elem.e2),
        fint1 = i == 1 ? elem.fint1 : zero(elem.fint1),
        fint2 = i == count ? elem.fint2 : zero(elem.fint2),
        t1 = i == 1 ? elem.t1 : zero(elem.t1),
        t2 = i == count ? elem.t2 : zero(elem.t2),
        r1 = i == 1 ? elem.r1 : zero(elem.r1),
        r2 = i == count ? elem.r2 : zero(elem.r2),
    )) for i in 1:count]
end

@inline _has_tracking_steps(::AbstractElement) = false
@inline _has_tracking_steps(::_SLICED_MULTIPOLE) = true
@inline _has_tracking_steps(::_SLICED_BEND) = true
@inline _is_length_refinable(::AbstractElement) = false
@inline _is_length_refinable(::Union{_SLICED_DRIFT,_SLICED_BEND,LBend}) = true

function _optics_slice_count(elem::AbstractElement, sample_integrator_steps::Bool,
                             max_step)
    count = 1
    if sample_integrator_steps && _has_tracking_steps(elem)
        count = max(count, getfield(elem, :num_int_steps))
    end
    if max_step !== nothing && _is_length_refinable(elem) && !iszero(get_length(elem))
        count = max(count, ceil(Int, abs(get_length(elem)) / max_step))
        # Do not replace a configured bend integrator by fewer, larger steps.
        if elem isa _SLICED_BEND && elem.num_int_steps > 0
            count = max(count, elem.num_int_steps)
        end
    end
    return count
end

function _slice_for_optics(elem::AbstractElement, count::Int)
    count == 1 && return AbstractElement[elem]
    elem isa _SLICED_DRIFT && return _slice_drift(elem, count)
    elem isa _SLICED_MULTIPOLE && return _slice_multipole(elem, count)
    elem isa _SLICED_BEND && return _slice_bend(elem, count)
    elem isa LBend && return _slice_lbend(elem, count)
    throw(ArgumentError("$(typeof(elem)) cannot be split for optics sampling"))
end

"""
    refine_lattice(lat; sample_integrator_steps=true, max_step=nothing)

Return a lattice refined for optics sampling. Thick multipoles and bends are
split at their configured integration steps when `sample_integrator_steps` is
true. `max_step` additionally limits the length of drift and bend pieces.

Entrance offsets, rotations, pole-face maps, and fringes are retained only on
the first piece; their exit counterparts are retained only on the last piece.
If `max_step` requests bend pieces shorter than the configured integration
step, the bend integration is refined to match the requested sampling.
"""
function refine_lattice(lat::Lattice;
                        sample_integrator_steps::Bool=true,
                        max_step::Union{Nothing,Real}=nothing)
    if max_step !== nothing
        isfinite(max_step) && max_step > 0 ||
            throw(ArgumentError("max_step must be finite and positive"))
    end

    elements = AbstractElement[]
    for elem in lat.elements
        count = _optics_slice_count(elem, sample_integrator_steps, max_step)
        append!(elements, _slice_for_optics(elem, count))
    end
    return Lattice(elements; name=lat.name, periodic=lat.periodic)
end

"""
    _resolved_elements(lat, ctx)

Element vector with every time-varying element materialized at `ctx`. Returns
`lat.elements` itself when the lattice is static, so the common case allocates
nothing.
"""
function _resolved_elements(lat::Lattice, ctx::TimeContext)
    any(e -> e isa TimeVaryingElement, lat.elements) || return lat.elements
    return [_resolve_for_time(elem, ctx) for elem in lat.elements]
end

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
- Final 6D coordinates after tracking, or `NaN` coordinates when the particle
  is lost on an element aperture (set `check_apertures=false` to track through
  apertures as if they were absent).
"""
function linepass(lat::Lattice, r::SVector{6,S}, beam::Beam{T};
                  time::Real=zero(T), turn::Integer=0,
                  check_apertures::Bool=true) where {T,S}
    β_inv = beti(beam)
    ctx = TimeContext(T(time); turn=turn)
    for elem in lat.elements
        elem_now = _resolve_for_time(elem, ctx)
        r = pass!(elem_now, r, β_inv)
        if check_lost(r)
            return r  # Return immediately if particle is lost
        end
        if check_apertures
            rap, eap = _elem_apertures(elem_now)
            if aperture_lost(r, rap, eap)
                # Single-particle tracking carries no lost flag, so an aperture
                # loss is reported as NaN coordinates: without this the particle
                # kept being tracked through the rest of the lattice and the
                # caller saw a finite, plausible-looking result. `linepass!`
                # keeps the evolved coordinates because it has a flag to set.
                return _lost_coords(r, eltype(rap))
            end
        end
    end
    return r
end

"""
    linepass(lat::Lattice, r::SVector{6,T}) -> SVector{6,T}

Track a single particle through the lattice using default beam (1 GeV electron).
"""
function linepass(lat::Lattice, r::SVector{6,S}; time::Real=zero(Float64), turn::Integer=0,
                  check_apertures::Bool=true) where S
    beam = Beam(1.0e9)
    return linepass(lat, r, beam; time=time, turn=turn, check_apertures=check_apertures)
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
                  time::Real=zero(T), dt_turn::Real=zero(T), turn::Integer=0,
                  check_apertures::Bool=true) where {T,S}
    _require_periodic(lat, "ringpass")
    nturns >= 0 || throw(ArgumentError("nturns must be nonnegative"))
    t = T(time)
    dt = T(dt_turn)
    trn = Int(turn)
    for _ in 1:nturns
        r = linepass(lat, r, beam; time=t, turn=trn, check_apertures=check_apertures)
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
