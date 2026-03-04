"""
    Marker{N}

A marker element with no length or physical effect.
"""
struct Marker{N} <: AbstractElement
    name::N
end
Marker(; name::Union{Symbol, String} = :MARKER) = Marker(Symbol(name))
Adapt.adapt_structure(to, x::Marker) = Marker(nothing)
