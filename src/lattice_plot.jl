"""
    LatticeGlyph

Backend-independent description of one element in a lattice illustration.
`s_start` and `s_end` are the physical element boundaries. `plot_start` and
`plot_end` expand zero-length elements to a visible width. A signed `height`
places focusing and defocusing multipoles on opposite sides of the baseline;
`centered=true` makes the glyph straddle the baseline instead.
"""
struct LatticeGlyph{T}
    index::Int
    name::Symbol
    kind::Symbol
    s_start::T
    s_end::T
    plot_start::T
    plot_end::T
    height::T
    centered::Bool
end

@inline _lattice_plot_kind(::AbstractElement) = :element
@inline _lattice_plot_kind(::AbstractDrift) = :drift
@inline _lattice_plot_kind(::Union{SBend,ExactSBend,SBendSC,LBend}) = :bend
@inline _lattice_plot_kind(::Union{Quadrupole,QuadrupoleSC}) = :quadrupole
@inline _lattice_plot_kind(::Union{Sextupole,SextupoleSC}) = :sextupole
@inline _lattice_plot_kind(::Union{Octupole,OctupoleSC}) = :octupole
@inline _lattice_plot_kind(::ThinMultipole) = :multipole
@inline _lattice_plot_kind(::RFCavity) = :rf_cavity
@inline _lattice_plot_kind(::AccelCavity) = :rf_cavity
@inline _lattice_plot_kind(::CrabCavity) = :crab_cavity
@inline _lattice_plot_kind(::Corrector) = :kicker
@inline _lattice_plot_kind(::Solenoid) = :solenoid
@inline _lattice_plot_kind(::Wiggler) = :wiggler
@inline _lattice_plot_kind(::Union{LongitudinalRLCWake,LongitudinalWake}) = :wake
@inline _lattice_plot_kind(::Union{StrongThinGaussianBeam,StrongGaussianBeam}) = :beambeam
@inline _lattice_plot_kind(::SpaceCharge) = :space_charge
@inline _lattice_plot_kind(::Union{Patch,Translation,YRotation}) = :patch

function _lattice_plot_kind(element::Marker)
    label = uppercase(String(element.name))
    return startswith(label, "BPM") || occursin("MON", label) ? :bpm : :marker
end

@inline _nonzero_sign(value) = iszero(value) ? one(value) : sign(value)
@inline _lattice_plot_height(::AbstractElement) = 0.45
@inline _lattice_plot_height(element::Quadrupole) = _nonzero_sign(element.k1)
@inline _lattice_plot_height(element::QuadrupoleSC) = _nonzero_sign(element.k1)
@inline _lattice_plot_height(element::Sextupole) = 0.75 * _nonzero_sign(element.k2)
@inline _lattice_plot_height(element::SextupoleSC) = 0.75 * _nonzero_sign(element.k2)
@inline _lattice_plot_height(element::Octupole) = 0.60 * _nonzero_sign(element.k3)
@inline _lattice_plot_height(element::OctupoleSC) = 0.60 * _nonzero_sign(element.k3)
@inline _lattice_plot_height(::Union{SBend,ExactSBend,SBendSC,LBend}) = 0.80
@inline _lattice_plot_height(::Union{RFCavity,AccelCavity}) = 0.85
@inline _lattice_plot_height(::CrabCavity) = 0.85
@inline _lattice_plot_height(::Marker) = 0.35
@inline _lattice_plot_height(element::Corrector) =
    0.50 * _nonzero_sign(iszero(element.xkick) ? element.ykick : element.xkick)
@inline _lattice_plot_height(::Solenoid) = 0.65
@inline _lattice_plot_height(::Wiggler) = 0.65
@inline _lattice_plot_height(::Union{LongitudinalRLCWake,LongitudinalWake}) = -0.45
@inline _lattice_plot_height(::Union{StrongThinGaussianBeam,StrongGaussianBeam}) = -0.55
@inline _lattice_plot_height(::SpaceCharge) = -0.55
@inline _lattice_plot_height(::Union{Patch,Translation,YRotation}) = -0.30

@inline _lattice_plot_centered(::AbstractElement) = false
@inline _lattice_plot_centered(::Union{SBend,ExactSBend,SBendSC,LBend}) = true
@inline _lattice_plot_centered(element::Union{Quadrupole,QuadrupoleSC}) =
    iszero(element.k1)
@inline _lattice_plot_centered(element::Union{Sextupole,SextupoleSC}) =
    iszero(element.k2)
@inline _lattice_plot_centered(element::Union{Octupole,OctupoleSC}) =
    iszero(element.k3)

"""
    lattice_plot_data(lat; thin_width=nothing, include_drifts=false,
                      time=0.0, turn=0)

Return `LatticeGlyph` objects for drawing a lattice strip on a longitudinal
axis. Thick elements retain their physical boundaries. Zero-length elements
are centered at their physical position and expanded to `thin_width`, which
defaults to 0.2% of the lattice length.

Drifts are omitted by default because the baseline represents them. Set
`include_drifts=true` to include explicit drift glyphs. The result has no
plotting-package dependency and can be consumed by any graphics backend.
"""
function lattice_plot_data(lat::Lattice;
                           thin_width::Union{Nothing,Real}=nothing,
                           include_drifts::Bool=false,
                           time::Real=0.0,
                           turn::Integer=0)
    positions = spos(lat; time, turn)
    circumference = positions[end]
    width = isnothing(thin_width) ? max(0.002 * circumference, 1.0e-6) : Float64(thin_width)
    isfinite(width) && width > 0 ||
        throw(ArgumentError("thin_width must be finite and positive"))
    context = TimeContext(Float64(time); turn)
    glyphs = LatticeGlyph{Float64}[]

    for (index, wrapped) in enumerate(lat)
        element = _resolve_for_time(wrapped, context)
        kind = _lattice_plot_kind(element)
        kind === :drift && !include_drifts && continue
        s_start = positions[index]
        s_end = positions[index + 1]
        plot_start, plot_end = if s_end > s_start
            (s_start, s_end)
        else
            (s_start - width / 2, s_start + width / 2)
        end
        raw_name = _element_name(element)
        name = isnothing(raw_name) ? Symbol(nameof(typeof(element))) : Symbol(raw_name)
        push!(glyphs, LatticeGlyph(
            index, name, kind, s_start, s_end, plot_start, plot_end,
            Float64(_lattice_plot_height(element)),
            _lattice_plot_centered(element),
        ))
    end
    return glyphs
end

"""
    plot_lattice!(axis, lat; kwargs...)

Draw a lattice illustration on a Makie axis. This method becomes available
after loading a Makie backend such as CairoMakie, GLMakie, or WGLMakie. Use
`linkxaxes!` to align the returned strip with Twiss, orbit, or other `s` plots.
"""
function plot_lattice! end
