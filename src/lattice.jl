"""
    lattice.jl

Lattice representation and high-level tracking functions for TrackPad.jl.
"""

using StaticArrays

export Lattice, Beam, isperiodic, refine_lattice, track, track!, linepass!

# =============================================================================
# Beam Definition
# =============================================================================

"""
    Beam{T}

The reference particle: the five numbers every kernel needs and nothing else.
Macroparticle coordinates are kept separately as `N×6` matrices.

# Fields
- `energy::T`: reference **total** energy `E0` in eV
- `mass::T`: rest-mass energy `m0c²` in eV
- `charge::T`: signed charge in units of `e`
- `gamma::T`: `E0/(m0c²)`
- `beta::T`: `P0c/E0`

See the constructor [`Beam`](@ref) for the kinetic-energy and momentum forms.
"""
struct Beam{T}
    energy::T      # Reference total energy E0 [eV]
    mass::T        # Rest-mass energy m0*c^2 [eV]
    charge::T      # Signed charge [e]
    gamma::T       # E0 / (m0*c^2)
    beta::T        # P0*c / E0
end

"""Electron rest-mass energy in eV."""
const M_ELECTRON = 0.51099895069e6  # eV (electron rest mass energy)

"""Proton rest-mass energy in eV."""
const M_PROTON = 938.27208816e6     # eV (proton rest mass energy)

"""
    Beam(energy; mass=M_ELECTRON, charge=-1.0)
    Beam(; kinetic, mass=M_ELECTRON, charge=-1.0)
    Beam(; pc, mass=M_ELECTRON, charge=-1.0)

The reference particle. The positional `energy` is the **total** energy `E0`
in eV — what MAD-X `ENERGY`, PALS `E_tot_ref` and most lattice files mean.
For low-energy machines, where the kinetic energy or the momentum is the
natural number, use the keyword forms: `kinetic` is `K0 = E0 − m0c²` and `pc`
is `P0c`, both in eV. `mass` is the rest-mass energy `m0c²` in eV; `charge` is
signed in units of `e`.

Derived fields are `gamma = E0/(m0c²)` and `beta = P0c/E0`; [`kinetic_energy`](@ref)
and [`p0c`](@ref) return the other two energy-like quantities.

`Beam(3.0e9)` is a 3 GeV total-energy electron. A total energy below the
rest mass is rejected — the usual sign that a kinetic energy was passed
positionally; write `Beam(kinetic=K, mass=m)` for that.

!!! note "Changed in 0.2"
    Before 0.2 the positional argument was the kinetic energy (JuTrack's
    convention). `Beam(kinetic=3.0e9)` reproduces the old `Beam(3.0e9)`
    exactly.
"""
function Beam(energy::T; mass::Real=T(M_ELECTRON), charge::Real=T(-1.0)) where T
    mass = T(mass)
    charge = T(charge)
    energy >= mass || throw(ArgumentError(
        "total energy $(energy) eV is below the rest-mass energy $(mass) eV. " *
        "For a kinetic energy write Beam(kinetic=K, mass=m); for a momentum " *
        "Beam(pc=P0c, mass=m)."))
    gamma = energy / mass
    beta = sqrt(one(T) - one(T) / gamma^2)
    return Beam{T}(energy, mass, charge, gamma, beta)
end

function Beam(; kinetic=nothing, pc=nothing, mass::Real=M_ELECTRON, charge::Real=-1.0)
    (kinetic === nothing) == (pc === nothing) && throw(ArgumentError(
        "Beam(; ...) takes exactly one of `kinetic` or `pc`"))
    if kinetic !== nothing
        T = promote_type(typeof(kinetic), typeof(mass))
        kinetic >= 0 || throw(ArgumentError("kinetic energy must be nonnegative"))
        # Same operation order as the pre-0.2 constructor, so `Beam(kinetic=K)`
        # reproduces `Beam(K)` of earlier versions (and JuTrack) bit for bit.
        return Beam(T(kinetic) + T(mass); mass=T(mass), charge=T(charge))
    else
        T = promote_type(typeof(pc), typeof(mass))
        pc >= 0 || throw(ArgumentError("pc must be nonnegative"))
        return Beam(hypot(T(pc), T(mass)); mass=T(mass), charge=T(charge))
    end
end

"""
    kinetic_energy(beam::Beam) -> K0 [eV]

Reference kinetic energy `E0 − m0c²`.
"""
kinetic_energy(beam::Beam) = beam.energy - beam.mass

"""
    p0c(beam::Beam) -> P0*c [eV]

Reference momentum times `c`, `β0 E0`. This is the quantity that normalises the
canonical momenta and every RF and wake kick.
"""
p0c(beam::Beam) = beam.beta * beam.energy

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
@inline _is_length_refinable(::Union{_SLICED_DRIFT,_SLICED_MULTIPOLE,_SLICED_BEND,LBend}) = true

function _optics_slice_count(elem::AbstractElement, sample_integrator_steps::Bool,
                             max_step, slices::Int=1)
    count = 1
    if sample_integrator_steps && _has_tracking_steps(elem)
        count = max(count, getfield(elem, :num_int_steps))
    end
    refinable = _is_length_refinable(elem) && !iszero(get_length(elem))
    if refinable && max_step !== nothing
        count = max(count, ceil(Int, abs(get_length(elem)) / max_step))
    end
    if refinable && slices > 1
        count = max(count, slices)
    end
    # Splitting gives every piece one integration step: never replace a
    # configured integrator by fewer, larger steps.
    if count > 1 && _has_tracking_steps(elem) && getfield(elem, :num_int_steps) > 0
        count = max(count, getfield(elem, :num_int_steps))
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
    refine_lattice(lat; sample_integrator_steps=true, max_step=nothing, slices=1)

Return a lattice refined for optics sampling: the lattice [`twiss`](@ref)
evaluates when given the same keywords. Elements of finite length (drifts,
thick multipoles, bends) are split into pieces — one per configured
integration step when `sample_integrator_steps` is true, no longer than
`max_step`, and at least `slices` of them; the largest count wins. Splitting
gives every piece one integration step, so a thick element is never
integrated in fewer steps than configured; asking for more pieces than steps
refines its integration.

Entrance offsets, rotations, pole-face maps, and fringes are retained only on
the first piece; their exit counterparts are retained only on the last piece.
"""
function refine_lattice(lat::Lattice;
                        sample_integrator_steps::Bool=true,
                        max_step::Union{Nothing,Real}=nothing,
                        slices::Int=1)
    if max_step !== nothing
        isfinite(max_step) && max_step > 0 ||
            throw(ArgumentError("max_step must be finite and positive"))
    end
    slices >= 1 || throw(ArgumentError("slices must be at least 1"))

    elements = AbstractElement[]
    for elem in lat.elements
        count = _optics_slice_count(elem, sample_integrator_steps, max_step, slices)
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
    track(lat, r::SVector{6}, beam = Beam(1e9); nturns = 1, kwargs...) -> SVector{6}
    track(lat, coords::Matrix, beam = Beam(1e9); nturns = 1, kwargs...) -> Matrix

Track through `lat` for `nturns` passes and return the final coordinates.

One initial condition is an `SVector{6}` and comes back as one; an `N x 6`
matrix is copied and tracked as a bunch (see [`track!`](@ref) to track one in
place, and for the `lost` and `threaded` keywords).

`nturns = 1` (the default) is a single pass through a line or a ring; more
turns require a periodic lattice. A particle lost on an element aperture or on
the global coordinate limits comes back as `NaN` coordinates
(`check_apertures = false` tracks through apertures as if they were absent).

Time-varying elements are resolved once per turn at
`(time + (n-1)*dt_turn, turn + n - 1)`.

`linepass(lat, r, beam)` is the single-pass JuTrack-compatible spelling.
"""
function track(lat::Lattice, r::SVector{6,S}, beam::Beam{T}=Beam(1.0e9);
               nturns::Integer=1, time::Real=zero(T), dt_turn::Real=zero(T),
               turn::Integer=0, check_apertures::Bool=true) where {T,S}
    _check_nturns(lat, nturns, "track")
    t = T(time)
    dt = T(dt_turn)
    trn = Int(turn)
    for _ in 1:nturns
        r = linepass(lat, r, beam; time=t, turn=trn, check_apertures=check_apertures)
        if check_lost(r)
            # As in `track!`, a lost particle leaves the tracker as NaN rather
            # than as the finite, plausible-looking coordinates it happened to
            # hold when it exceeded the limits. (`linepass` keeps those, for
            # JuTrack compatibility.)
            return _lost_coords(r, S)
        end
        t += dt
        trn += 1
    end
    return r
end

track(lat::Lattice, coords::AbstractMatrix, beam::Beam=Beam(1.0e9); kwargs...) =
    track!(copy(coords), lat, beam; kwargs...)

@inline function _check_nturns(lat::Lattice, nturns::Integer, what::AbstractString)
    nturns >= 0 || throw(ArgumentError("nturns must be nonnegative"))
    nturns <= 1 || lat.periodic || throw(ArgumentError(
        "$what: tracking $(nturns) turns requires a periodic lattice; this one is " *
        "an open line (construct it with `Lattice(...; periodic=true)` for a ring)"))
    return nothing
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
function linepass!(coords::Matrix{T}, lat::Lattice, beam::Beam,
                   lost_flags::AbstractVector; time::Real=zero(T), turn::Integer=0) where T
    return _linepass_core!(coords, lat, T(beti(beam)), lost_flags; time=time, turn=turn)
end

# One pass of the whole bunch, shared by `linepass!` and the serial path of
# `track!`. Takes the inverse velocity rather than the beam so that a Float32
# coordinate matrix can be tracked with a Float64 `Beam`.
function _linepass_core!(coords::Matrix{T}, lat::Lattice, β_inv::T,
                         lost_flags::AbstractVector;
                         time::Real=zero(T), turn::Integer=0) where T
    nparticles = size(coords, 1)
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
        # Function barrier: `elem_now` comes from a `Vector{AbstractElement}`,
        # so calling `pass!` on it directly inside the particle loop costs one
        # dynamic dispatch *per particle* and boxes every returned SVector on
        # the heap. Handing the element to a function that specializes on its
        # concrete type moves that to one dispatch per element and makes the
        # particle loop monomorphic and allocation-free.
        _track_element!(coords, lost_flags, nparticles, elem_now, β_inv,
                        rap, eap, check_aperture)
    end

    return nothing
end

# Specializes on the concrete element type `E`; see the call site above.
function _track_element!(coords::Matrix{T}, lost_flags::AbstractVector,
                         nparticles::Int, elem::E, β_inv::T,
                         rap, eap, check_aperture::Bool) where {T,E<:AbstractElement}
    @inbounds for i in 1:nparticles
        if !iszero(lost_flags[i])
            continue
        end

        r = SVector{6,T}(coords[i, 1], coords[i, 2], coords[i, 3],
                         coords[i, 4], coords[i, 5], coords[i, 6])

        r_new = pass!(elem, r, β_inv)

        if check_lost(r_new)
            lost_flags[i] = true
            continue
        end

        coords[i, 1] = r_new[1]
        coords[i, 2] = r_new[2]
        coords[i, 3] = r_new[3]
        coords[i, 4] = r_new[4]
        coords[i, 5] = r_new[5]
        coords[i, 6] = r_new[6]

        # Aperture loss keeps the evolved coordinates (JuTrack semantics).
        check_aperture && outside_aperture(r_new, rap, eap) && (lost_flags[i] = true)
    end
    return nothing
end

"""
    track!(coords::Matrix, lat, beam = Beam(1e9); nturns = 1, lost = nothing,
           threaded = false, time = 0, dt_turn = 0, turn = 0) -> coords

Track the `N x 6` matrix `coords` through `lat` for `nturns` passes, in place.

`nturns = 1` (the default) is a single pass through a line or a ring; more
turns require a periodic lattice. Particles lost on an element aperture or on
the global coordinate limits are written back as `NaN` coordinates and skipped
by the remaining elements and turns; pass `lost`, a vector with one entry per
particle, to also receive the flags (anything nonzero marks a lost particle).

`threaded = true` splits the bunch into one contiguous chunk per Julia thread.
The result is identical to the serial path but the element order within a turn
is the only synchronisation point, so collective elements (longitudinal wakes)
are not supported there.

Time-varying elements are resolved once per turn at
`(time + (n-1)*dt_turn, turn + n - 1)`.

`track!(coords, gl)` with a packed [`GPULattice`](@ref) is the CPU/GPU kernel
form of the same call. `linepass!` is the single-pass JuTrack-compatible
spelling, which reports losses through its `lost_flags` argument and leaves the
coordinates as they were.
"""
function track!(coords::Matrix{T}, lat::Lattice, beam::Beam=Beam(1.0e9);
                nturns::Integer=1,
                lost::Union{Nothing,AbstractVector}=nothing,
                threaded::Bool=false,
                time::Real=zero(T), dt_turn::Real=zero(T),
                turn::Integer=0) where T
    _check_nturns(lat, nturns, "track!")
    n = size(coords, 1)
    size(coords, 2) == 6 ||
        throw(ArgumentError("coords must be an N x 6 matrix, got $(size(coords))"))
    lost === nothing || length(lost) == n ||
        throw(ArgumentError("lost must have one entry per particle (got $(length(lost)) for $n)"))
    # `Vector{Bool}` is byte-addressed: a `BitVector` would let two threads
    # write adjacent bits of the same word, so it is copied in and out.
    bitflags = threaded && lost isa BitVector
    flags = lost === nothing || bitflags ? fill(false, n) : lost
    bitflags && copyto!(flags, lost)
    β_inv = T(beti(beam))
    t = T(time)
    dt = T(dt_turn)
    trn = Int(turn)

    if threaded
        any(e -> _resolve_for_time(e, TimeContext(t; turn=trn)) isa
                 Union{LongitudinalRLCWake, LongitudinalWake}, lat.elements) &&
            throw(ArgumentError(
                "track!: collective elements (longitudinal wakes) need the whole " *
                "bunch at once and are not supported with threaded=true"))
        # Implemented next to the chunking helpers in src/gpu.jl.
        _threaded_track!(coords, lat, β_inv, flags, nturns, t, dt, trn)
    else
        for _ in 1:nturns
            _linepass_core!(coords, lat, β_inv, flags; time=t, turn=trn)
            t += dt
            trn += 1
        end
    end

    _write_lost!(coords, flags)
    bitflags && copyto!(lost, flags)
    return coords
end

# Lost particles leave the tracker as NaN, whatever they were when they died.
function _write_lost!(coords::Matrix{T}, flags::AbstractVector) where T
    @inbounds for i in axes(coords, 1)
        iszero(flags[i]) && continue
        for j in 1:6
            coords[i, j] = T(NaN)
        end
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
    pc0 = p0c(beam)
    isfinite(pc0) && pc0 > zero(T) ||
        throw(ArgumentError("reference beam momentum P0*c must be finite and positive"))
    qmacro = T(bunch_charge) / T(nmacro)
    return beam.charge * qmacro / pc0
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
