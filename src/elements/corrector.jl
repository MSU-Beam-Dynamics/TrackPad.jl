"""
    Corrector{T, N}

A corrector element.
"""
struct Corrector{T, N} <: AbstractMagnet
    name::N
    L::T
    xkick::T
    ykick::T
    t1::SVector{6, T}
    t2::SVector{6, T}
    r1::SMatrix{6, 6, T, 36}
    r2::SMatrix{6, 6, T, 36}
end

function Corrector(L, hk, vk;
                   name::Union{Symbol, String} = :CORRECTOR,
                   t1 = nothing, t2 = nothing,
                   r1 = nothing, r2 = nothing)
    T = _promote_element_type(L, hk, vk, t1, t2, r1, r2)
    Corrector{T, Symbol}(Symbol(name), T(L), T(hk), T(vk),
        SVector{6, T}(_default_vec(t1, T, Val(6))),
        SVector{6, T}(_default_vec(t2, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r1, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r2, T, Val(6))))
end

function Adapt.adapt_structure(to, x::Corrector)
    Corrector(nothing, adapt(to, x.L), adapt(to, x.xkick), adapt(to, x.ykick),
              adapt(to, x.t1), adapt(to, x.t2), adapt(to, x.r1), adapt(to, x.r2))
end

"""
    HKicker(; kwargs...)

Horizontal-kicker convenience wrapper around `Corrector`.
"""
function HKicker(; name::Union{Symbol, String} = :HKICKER, L = 0.0, xkick = 0.0)
    return Corrector(L, xkick, zero(xkick); name=name)
end

"""
    VKicker(; kwargs...)

Vertical-kicker convenience wrapper around `Corrector`.
"""
function VKicker(; name::Union{Symbol, String} = :VKICKER, L = 0.0, ykick = 0.0)
    return Corrector(L, zero(ykick), ykick; name=name)
end
