using StaticArrays

"""
    TimeContext(real_time, turn)

Evaluation context for time-dependent parameters:
- `real_time`: physical time-like scalar
- `turn`: integer turn index for ring tracking
"""
struct TimeContext{T<:Real}
    real_time::T
    turn::Int
end

TimeContext(real_time::T; turn::Integer=0) where {T<:Real} = TimeContext{T}(real_time, Int(turn))

@inline _to_time_context(ctx::TimeContext) = ctx
@inline _to_time_context(real_time::Real, turn::Integer=0) = TimeContext(Float64(real_time), Int(turn))

"""
    TimeFunction(f)

Lightweight callable wrapper for `f(t)`.
"""
struct TimeFunction{F<:Function}
    f::F
end

(tf::TimeFunction)(t) = tf.f(t)

"""
    TimeDependentParam

Parameter represented as a function of time.
"""
struct TimeDependentParam{F<:Function}
    f::TimeFunction{F}
end

TimeDependentParam(f::Function) = TimeDependentParam(TimeFunction(f))
TimeDependentParam(a) = TimeDependentParam((_t) -> a)
TimeDependentParam(a::TimeDependentParam) = a

Time() = TimeDependentParam((ctx) -> (ctx isa TimeContext ? ctx.real_time : ctx))
RealTime() = Time()
Turn() = TimeDependentParam((ctx) -> (ctx isa TimeContext ? ctx.turn : 0))

(d::TimeDependentParam)(ctx::TimeContext) = d.f(ctx)
(d::TimeDependentParam)(real_time::Real) = d.f(_to_time_context(real_time, 0))

Base.convert(::Type{TimeDependentParam}, a::Number) = TimeDependentParam(a)
Base.convert(::Type{TimeDependentParam}, a::TimeDependentParam) = a

Base.zero(::TimeDependentParam) = TimeDependentParam((_t) -> 0)
Base.one(::TimeDependentParam) = TimeDependentParam((_t) -> 1)

for op in (:+, :-, :*, :/, :^)
    @eval begin
        Base.$op(da::TimeDependentParam, b::Number) = begin
            fa = da.f
            TimeDependentParam((t) -> Base.$op(fa(t), b))
        end

        Base.$op(a::Number, db::TimeDependentParam) = begin
            fb = db.f
            TimeDependentParam((t) -> Base.$op(a, fb(t)))
        end

        Base.$op(da::TimeDependentParam, db::TimeDependentParam) = begin
            fa = da.f
            fb = db.f
            TimeDependentParam((t) -> Base.$op(fa(t), fb(t)))
        end
    end
end

function Base.literal_pow(::typeof(^), da::TimeDependentParam, ::Val{N}) where {N}
    fa = da.f
    return TimeDependentParam((t) -> Base.literal_pow(^, fa(t), Val(N)))
end

for op in (:+, :-, :sqrt, :exp, :log, :sin, :cos, :tan, :sinh, :cosh, :tanh,
           :asin, :acos, :atan, :asinh, :acosh, :atanh, :sinc, :conj, :log10,
           :isnan, :sign, :abs)
    @eval begin
        Base.$op(d::TimeDependentParam) = begin
            f = d.f
            TimeDependentParam((t) -> Base.$op(f(t)))
        end
    end
end

atan2(d1::TimeDependentParam, d2::TimeDependentParam) = begin
    f1 = d1.f
    f2 = d2.f
    TimeDependentParam((t) -> atan(f1(t), f2(t)))
end

Base.promote_rule(::Type{TimeDependentParam}, ::Type{U}) where {U<:Number} = TimeDependentParam
Base.broadcastable(o::TimeDependentParam) = Ref(o)

Base.isapprox(::TimeDependentParam, ::Number; kwargs...) = false
Base.isapprox(::Number, ::TimeDependentParam; kwargs...) = false
Base.:(==)(::TimeDependentParam, ::Number) = false
Base.:(==)(::Number, ::TimeDependentParam) = false
Base.isinf(::TimeDependentParam) = false

@inline teval(f::TimeFunction, ctx::TimeContext) = f(ctx)
@inline teval(f::TimeFunction, real_time::Real) = f(_to_time_context(real_time, 0))
@inline teval(f::TimeDependentParam, ctx::TimeContext) = f(ctx)
@inline teval(f::TimeDependentParam, real_time::Real) = f(real_time)
@inline teval(f, _ctx) = f
@inline teval(f::Tuple, ctx) = map(fi -> teval(fi, ctx), f)
@inline teval(f::StaticArray, ctx) = map(fi -> teval(fi, ctx), f)

time_lower(tp::TimeDependentParam) = tp.f
time_lower(tp) = tp
time_lower(tp::Tuple) = map(ti -> time_lower(ti), tp)
time_lower(tp::StaticArray) = static_timecheck(tp) ? TimeFunction(t -> teval(tp, t)) : tp

static_timecheck(_tp) = false
static_timecheck(::TimeDependentParam) = true
static_timecheck(::TimeFunction) = true
static_timecheck(t::Tuple) = any(static_timecheck, t)
static_timecheck(t::StaticArray) = any(static_timecheck, t)

"""
    TimeVaryingElement

Wrapper around a concrete element and a set of time-dependent field updates.
"""
struct TimeVaryingElement{E<:AbstractElement,U<:NamedTuple} <: AbstractElement
    base::E
    updates::U
end

function Adapt.adapt_structure(_to, x::TimeVaryingElement)
    throw(ArgumentError("TimeVaryingElement stores host closures. Call `materialize_lattice(...)` before GPU adaptation."))
end

"""
    timed(elem; kwargs...)

Create a lazily-evaluated element where any provided field can depend on time.
Each keyword can be a `Number`, `Function`, or `TimeDependentParam`.
"""
function timed(elem::E; kwargs...) where {E<:AbstractElement}
    updates_raw = NamedTuple(kwargs)
    for key in keys(updates_raw)
        hasfield(E, key) || throw(ArgumentError("$(E) has no field `$(key)`"))
    end
    updates_vals = map(TimeDependentParam, values(updates_raw))
    updates = NamedTuple{keys(updates_raw)}(updates_vals)
    return TimeVaryingElement{E,typeof(updates)}(elem, updates)
end

"""
    materialize(elem, t)

Evaluate time-dependent fields of `elem` at time `t`, returning a concrete
static element suitable for tracking kernels.
"""
function materialize(elem::TimeVaryingElement{E}, t) where {E<:AbstractElement}
    ctx = _to_time_context(t)
    base = elem.base
    updates = elem.updates
    vals = ntuple(i -> begin
        name = fieldname(E, i)
        raw = hasproperty(updates, name) ? teval(getproperty(updates, name), ctx) : getfield(base, i)
        convert(fieldtype(E, i), raw)
    end, fieldcount(E))
    return E(vals...)
end

materialize(elem::AbstractElement, _t) = elem

@inline _resolve_for_time(elem::AbstractElement, _time) = elem
@inline _resolve_for_time(elem::TimeVaryingElement, time) = materialize(elem, time)
