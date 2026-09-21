"""
    AbstractElement

Root of the element type hierarchy. Every beamline element is a subtype and
provides a `pass!(elem, r::SVector{6}, beti)` method mapping six canonical
coordinates `(x, px, y, py, z, δE)` through the element. Kernels are generic in
the coordinate type, so the same `pass!` serves `Float64`, `Float32`, dual
numbers and PolySeries `CTPS` coordinates. See the elements guide for the
catalogue and [`Lattice`](@ref) for how elements are assembled.
"""
abstract type AbstractElement end

"""
    AbstractMagnet <: AbstractElement

Thick or thin magnetic elements integrated with the symplectic drift–kick
scheme: `Quadrupole`, `Sextupole`, `Octupole`, `SBend`, `ExactSBend`, `LBend`,
`Solenoid`, `Corrector` and their space-charge variants. Most carry
`num_int_steps`, `polynom_a`/`polynom_b`, misalignments and apertures; see
[Common Optional Fields](@ref common_fields).
"""
abstract type AbstractMagnet <: AbstractElement end

"""
    AbstractDrift <: AbstractElement

Field-free regions (`Drift`, `DriftSC`). Tracked with the exact longitudinal
Hamiltonian by default (`USE_EXACT_HAMILTONIAN`).
"""
abstract type AbstractDrift <: AbstractElement end

"""
    AbstractCavity <: AbstractElement

RF structures that change the energy coordinate: `RFCavity`, `CrabCavity`,
`AccelCavity`. Their kicks scale with `charge/energy` of the reference beam and
follow the phase convention in the conventions guide.
"""
abstract type AbstractCavity <: AbstractElement end

"""
    AbstractTransferMap <: AbstractElement

Elements defined directly by a map rather than by a field model. Parent of
[`AbstractTransverseMap`](@ref) and [`AbstractLongitudinalRFMap`](@ref).
"""
abstract type AbstractTransferMap <: AbstractElement end

"""
    AbstractTransverseMap <: AbstractTransferMap

Reserved for user-supplied transverse maps. No concrete subtype ships with
TrackPad; define one and a `pass!` method to plug a custom linear or nonlinear
map into a `Lattice`.
"""
abstract type AbstractTransverseMap <: AbstractTransferMap end

"""
    AbstractLongitudinalRFMap <: AbstractTransferMap

One-turn longitudinal maps that stand in for distributed RF, such as
`LongitudinalRFMap` (momentum compaction plus a cavity's phase slip).
"""
abstract type AbstractLongitudinalRFMap <: AbstractTransferMap end

function _promote_element_type(L, args...)
    T = typeof(L)
    for arg in args
        if !isnothing(arg)
            T = promote_type(T, eltype(arg))
        end
    end
    return T
end

_default_vec(val, ::Type{T}, ::Val{N}) where {T, N} = isnothing(val) ? zero(SVector{N, T}) : val
_default_mat(val, ::Type{T}, ::Val{N}) where {T, N} = isnothing(val) ? zero(SMatrix{N, N, T, N * N}) : val
