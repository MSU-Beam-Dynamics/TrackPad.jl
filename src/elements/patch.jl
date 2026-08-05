"""
    Patch{T, N}

Zero-length PALS reference-frame change. Offsets are expressed in metres,
pitch/tilt angles in radians, and `t_offset` in seconds.
"""
struct Patch{T, N} <: AbstractElement
    name::N
    x_offset::T
    y_offset::T
    z_offset::T
    x_pitch::T
    y_pitch::T
    tilt::T
    t_offset::T
end

function Patch(;
               name::Union{Symbol, String} = :PATCH,
               x_offset = 0.0,
               y_offset = 0.0,
               z_offset = 0.0,
               x_pitch = 0.0,
               y_pitch = 0.0,
               tilt = 0.0,
               t_offset = 0.0)
    T = promote_type(
        typeof(x_offset),
        typeof(y_offset),
        typeof(z_offset),
        typeof(x_pitch),
        typeof(y_pitch),
        typeof(tilt),
        typeof(t_offset),
    )
    return Patch{T, Symbol}(
        Symbol(name),
        T(x_offset),
        T(y_offset),
        T(z_offset),
        T(x_pitch),
        T(y_pitch),
        T(tilt),
        T(t_offset),
    )
end

function Adapt.adapt_structure(to, patch::Patch)
    return Patch(
        name=patch.name,
        x_offset=adapt(to, patch.x_offset),
        y_offset=adapt(to, patch.y_offset),
        z_offset=adapt(to, patch.z_offset),
        x_pitch=adapt(to, patch.x_pitch),
        y_pitch=adapt(to, patch.y_pitch),
        tilt=adapt(to, patch.tilt),
        t_offset=adapt(to, patch.t_offset),
    )
end
