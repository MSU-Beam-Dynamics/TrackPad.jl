"""
    LongitudinalRLCWake{T, N}

RLC resonator wake model metadata.
"""
struct LongitudinalRLCWake{T, N} <: AbstractElement
    name::N
    freq::T
    Rshunt::T
    Q0::T
    scale::T
end

function LongitudinalRLCWake(; name::Union{Symbol, String} = :LONGITUDINALRLCWAKE,
                              freq = 1.0e9, Rshunt = 1.0e6, Q0 = 1.0, scale = 0.0)
    T = _promote_element_type(freq, Rshunt, Q0, scale)
    LongitudinalRLCWake{T, Symbol}(Symbol(name), T(freq), T(Rshunt), T(Q0), T(scale))
end

function Adapt.adapt_structure(to, x::LongitudinalRLCWake)
    LongitudinalRLCWake(
        name = x.name, freq = adapt(to, x.freq), Rshunt = adapt(to, x.Rshunt),
        Q0 = adapt(to, x.Q0), scale = adapt(to, x.scale),
    )
end

@inline function wakefieldfunc_RLCWake(rlcwake::LongitudinalRLCWake{T}, t::T) where T
    Q0p = sqrt(rlcwake.Q0^2 - T(0.25))
    w0 = T(2pi) * rlcwake.freq
    w0p = w0 / rlcwake.Q0 * Q0p
    if t > zero(T)
        return zero(T)
    end
    return rlcwake.Rshunt * w0 / rlcwake.Q0 *
           (cos(w0p * t) + sin(w0p * t) / (2 * Q0p)) *
           exp(w0 * t / (2 * rlcwake.Q0))
end

"""
    LongitudinalWake{T, N}

Tabulated longitudinal wake model metadata.
"""
struct LongitudinalWake{T, N, V<:AbstractVector{T}} <: AbstractElement
    name::N
    times::V
    wakefields::V
    fliphalf::T
    scale::T
end

function LongitudinalWake(times::AbstractVector, wakefields::AbstractVector;
                          name::Union{Symbol, String} = :LONGITUDINALWAKE,
                          fliphalf = -1.0, scale = 0.0)
    length(times) == length(wakefields) || throw(ArgumentError("times and wakefields must have the same length"))
    length(times) >= 2 || throw(ArgumentError("LongitudinalWake requires at least two samples"))
    T = _promote_element_type(fliphalf, scale, times, wakefields)
    LongitudinalWake{T, Symbol, Vector{T}}(Symbol(name), T.(collect(times)), T.(collect(wakefields)), T(fliphalf), T(scale))
end

function Adapt.adapt_structure(to, x::LongitudinalWake)
    LongitudinalWake(
        adapt(to, x.times), adapt(to, x.wakefields);
        name = x.name, fliphalf = adapt(to, x.fliphalf), scale = adapt(to, x.scale),
    )
end

@inline function _linear_interpolate(x::T, x_points::AbstractVector{T}, y_points::AbstractVector{T}) where T
    if x <= x_points[1]
        slope = (y_points[2] - y_points[1]) / (x_points[2] - x_points[1])
        return y_points[1] + slope * (x - x_points[1])
    elseif x >= x_points[end]
        n = length(x_points)
        slope = (y_points[n] - y_points[n - 1]) / (x_points[n] - x_points[n - 1])
        return y_points[n] + slope * (x - x_points[n])
    end
    @inbounds for i in 2:length(x_points)
        if x < x_points[i]
            slope = (y_points[i] - y_points[i - 1]) / (x_points[i] - x_points[i - 1])
            return y_points[i - 1] + slope * (x - x_points[i - 1])
        end
    end
    return y_points[end]
end

@inline function wakefieldfunc(elem::LongitudinalWake{T}, t::T) where T
    if t > elem.times[1] * elem.fliphalf
        return zero(T)
    end
    return _linear_interpolate(t * elem.fliphalf, elem.times, elem.wakefields)
end
