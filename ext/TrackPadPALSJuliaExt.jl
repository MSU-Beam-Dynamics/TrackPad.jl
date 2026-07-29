module TrackPadPALSJuliaExt

using TrackPad
import PALSJulia

function _scalar_value(node)
    value = String(node)
    value == "true" && return true
    value == "false" && return false
    integer = tryparse(Int, value)
    integer !== nothing && return integer
    number = tryparse(Float64, value)
    number !== nothing && return number
    return value
end

function _node_value(node)
    if PALSJulia.is_scalar(node)
        return _scalar_value(node)
    elseif PALSJulia.is_sequence(node)
        return Any[_node_value(child) for child in node]
    elseif PALSJulia.is_map(node)
        return Dict{String,Any}(
            string(key) => _node_value(child) for (key, child) in node
        )
    end
    throw(ArgumentError("Unsupported PALSJulia YAML node"))
end

function TrackPad.resolve_pals_full(filename::AbstractString;
                                    lattice::Union{String,Symbol,Nothing}=nothing,
                                    branch::Union{String,Symbol,Nothing}=nothing,
                                    strict::Bool=true)
    root_lattice = lattice === nothing ? "" : string(lattice)
    expanded = PALSJulia.parse_and_expand_pals(
        string(filename), root_lattice; problems=:none,
    )
    if strict && !isempty(expanded.problems)
        throw(ArgumentError(
            "PALS expansion failed:\n" * join(expanded.problems, "\n"),
        ))
    end
    data = _node_value(expanded.full_expanded)
    return TrackPad.resolve_pals(
        data; lattice=lattice, branch=branch, strict=strict,
    )
end

end
