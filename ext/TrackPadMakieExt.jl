module TrackPadMakieExt

using TrackPad
using Makie

const DEFAULT_COLORS = Dict{Symbol,Any}(
    :bend => "#4C78A8",
    :quadrupole => "#E45756",
    :sextupole => "#54A24B",
    :octupole => "#B279A2",
    :multipole => "#9D755D",
    :rf_cavity => "#F2CF5B",
    :crab_cavity => "#FF9DA6",
    :bpm => "#222222",
    :marker => "#777777",
    :kicker => "#F58518",
    :solenoid => "#72B7B2",
    :wiggler => "#1F77B4",
    :wake => "#BAB0AC",
    :beambeam => "#AF7AA1",
    :space_charge => "#8C564B",
    :patch => "#999999",
    :element => "#666666",
    :drift => "#DDDDDD",
)

"""
    TrackPad.plot_lattice!(axis, lat; height=1, baseline=0,
                           colors=DEFAULT_COLORS, kwargs...)

Draw element glyphs on an existing Makie axis. Additional keywords are passed
to [`TrackPad.lattice_plot_data`](@ref).
"""
function TrackPad.plot_lattice!(axis, lat::TrackPad.Lattice;
                                height::Real=1.0,
                                baseline::Real=0.0,
                                colors=DEFAULT_COLORS,
                                kwargs...)
    isfinite(height) && height > 0 ||
        throw(ArgumentError("height must be finite and positive"))
    isfinite(baseline) || throw(ArgumentError("baseline must be finite"))
    glyphs = TrackPad.lattice_plot_data(lat; kwargs...)
    plots = Any[]
    circumference = TrackPad.total_length(lat)
    push!(plots, lines!(axis, [0.0, circumference], [baseline, baseline]; color=:black))

    for glyph in glyphs
        extent = height * glyph.height
        bottom, top = if glyph.centered
            half_height = abs(extent) / 2
            (baseline - half_height, baseline + half_height)
        else
            (baseline + min(0.0, extent), baseline + max(0.0, extent))
        end
        points = Makie.Point2d[
            (glyph.plot_start, bottom), (glyph.plot_end, bottom),
            (glyph.plot_end, top), (glyph.plot_start, top),
        ]
        color = get(colors, glyph.kind, get(colors, :element, "#666666"))
        push!(plots, poly!(axis, points; color, strokecolor=:black, strokewidth=0.5))
    end

    xlims!(axis, 0, circumference)
    ylims!(axis, baseline - 1.25height, baseline + 1.25height)
    return (; glyphs, plots)
end

end
