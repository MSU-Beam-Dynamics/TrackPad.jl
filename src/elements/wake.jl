"""
    LongitudinalRLCWake{T, N}

RLC resonator longitudinal wake Green function metadata.

The implemented real-valued underdamped resonator formula requires
`freq > 0`, `Rshunt >= 0`, and `Q0 > 1/2`.

`wakefieldfunc_RLCWake` evaluates the Green function ``W(t)`` for a delay
``t = (z_\\mathrm{test} - z_\\mathrm{source})/c``; it vanishes for ``t > 0``
(a particle only feels sources ahead of it). Tracking applies the wake as a
collective kick: the potential is the discrete convolution of ``W`` with a
cloud-in-cell histogram of `r[5]` over `nbins` uniform bins, and each
macroparticle receives the bin-edge-interpolated potential at its own
coordinate. See `physical_wake_scale` for physical normalization.
"""
struct LongitudinalRLCWake{T, N} <: AbstractElement
    name::N
    freq::T
    Rshunt::T
    Q0::T
    scale::T
    nbins::Int
end

function LongitudinalRLCWake(; name::Union{Symbol, String} = :LONGITUDINALRLCWAKE,
                              freq = 1.0e9, Rshunt = 1.0e6, Q0 = 1.0, scale = 0.0,
                              nbins::Integer = 128)
    nbins >= 1 || throw(ArgumentError("nbins must be positive"))
    T = _promote_element_type(freq, Rshunt, Q0, scale)
    f = T(freq)
    r = T(Rshunt)
    q = T(Q0)
    s = T(scale)
    isfinite(f) && f > zero(T) || throw(ArgumentError("freq must be finite and positive"))
    isfinite(r) && r >= zero(T) || throw(ArgumentError("Rshunt must be finite and nonnegative"))
    isfinite(q) && q > T(0.5) || throw(ArgumentError("Q0 must be finite and greater than 1/2"))
    isfinite(s) || throw(ArgumentError("scale must be finite"))
    LongitudinalRLCWake{T, Symbol}(Symbol(name), f, r, q, s, Int(nbins))
end

function Adapt.adapt_structure(to, x::LongitudinalRLCWake)
    LongitudinalRLCWake(
        name = x.name, freq = adapt(to, x.freq), Rshunt = adapt(to, x.Rshunt),
        Q0 = adapt(to, x.Q0), scale = adapt(to, x.scale), nbins = x.nbins,
    )
end

"""
    wakefieldfunc_RLCWake(rlcwake::LongitudinalRLCWake, t) -> Green value

RLC resonator longitudinal wake **Green function** ``W(t)`` for the delay
argument ``t = (z_\\mathrm{test} - z_\\mathrm{source})/c``. Returns zero for
``t > 0`` (causality: only sources ahead of a particle contribute). The wake
potential itself is the convolution of this Green function with the bunch
histogram of `r[5]`; see `docs/src/conventions.md`.
"""
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

Tabulated longitudinal wake Green function metadata.

`wakefieldfunc` evaluates the tabulated Green function for a delay
``t = (z_\\mathrm{test} - z_\\mathrm{source})/c`` (with `fliphalf = -1`, the
table is sampled at ``|t|`` and linearly extrapolated outside its range).
Tracking applies the wake as a collective kick: the potential is the discrete
convolution of the tabulated Green function with a cloud-in-cell histogram of
`r[5]` over `nbins` uniform bins, and each macroparticle receives the
bin-edge-interpolated potential at its own coordinate.
"""
struct LongitudinalWake{T, N, V<:AbstractVector{T}} <: AbstractElement
    name::N
    times::V
    wakefields::V
    fliphalf::T
    scale::T
    nbins::Int
end

function LongitudinalWake(times::AbstractVector, wakefields::AbstractVector;
                          name::Union{Symbol, String} = :LONGITUDINALWAKE,
                          fliphalf = -1.0, scale = 0.0,
                          nbins::Integer = 128)
    length(times) == length(wakefields) || throw(ArgumentError("times and wakefields must have the same length"))
    length(times) >= 2 || throw(ArgumentError("LongitudinalWake requires at least two samples"))
    nbins >= 1 || throw(ArgumentError("nbins must be positive"))
    T = _promote_element_type(fliphalf, scale, times, wakefields)
    ts = T.(collect(times))
    ws = T.(collect(wakefields))
    fh = T(fliphalf)
    s = T(scale)
    all(isfinite, ts) || throw(ArgumentError("times must contain only finite values"))
    all(isfinite, ws) || throw(ArgumentError("wakefields must contain only finite values"))
    iszero(ts[1]) || throw(ArgumentError("times must start at zero delay"))
    all(i -> ts[i - 1] < ts[i], 2:length(ts)) ||
        throw(ArgumentError("times must be strictly increasing"))
    isfinite(fh) && fh < zero(T) ||
        throw(ArgumentError("fliphalf must be finite and negative for the canonical delay convention"))
    isfinite(s) || throw(ArgumentError("scale must be finite"))
    LongitudinalWake{T, Symbol, Vector{T}}(Symbol(name), ts, ws, fh, s, Int(nbins))
end

function Adapt.adapt_structure(to, x::LongitudinalWake)
    LongitudinalWake(
        adapt(to, x.times), adapt(to, x.wakefields);
        name = x.name, fliphalf = adapt(to, x.fliphalf), scale = adapt(to, x.scale),
        nbins = x.nbins,
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

"""
    wakefieldfunc(elem::LongitudinalWake, t) -> Green value

Tabulated longitudinal wake **Green function** for the delay argument
``t = (z_\\mathrm{test} - z_\\mathrm{source})/c``. With the default
`fliphalf = -1` the table (ascending positive `times`) is sampled at
``|t|`` and linearly extrapolated outside its range; values vanish where the
element's zero-delay guard applies. The wake potential itself is the
convolution of this Green function with the bunch histogram of `r[5]`;
see `docs/src/conventions.md`.
"""
@inline function wakefieldfunc(elem::LongitudinalWake{T}, t::T) where T
    if t > zero(T)
        return zero(T)
    end
    return _linear_interpolate(t * elem.fliphalf, elem.times, elem.wakefields)
end
