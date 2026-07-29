"""
    io.jl

Lattice I/O utilities for TrackPad.jl:

- [`read_pals`](@ref)  — read a PALS-standard YAML lattice file
- [`read_madx`](@ref)  — read a MAD-X sequence / input file
- [`write_pals`](@ref) — write a TrackPad `Lattice` to a PALS YAML file
"""

import YAML

export PALSResolvedElement, PALSResolvedBranch
export CompiledBranch, resolve_pals, resolve_pals_full, compile_branch
export read_pals, read_madx, write_pals

const _C_IO = 2.99792458e8  # m/s

# ============================================================
# ==================  PALS YAML Reader  ======================
# ============================================================

"""
One occurrence of an element in a resolved PALS branch.

`name` is the occurrence name, `source_name` is the inherited prototype name,
and `occurrence` disambiguates repeated uses without changing PALS names.
"""
struct PALSResolvedElement
    name::String
    source_name::String
    occurrence::Int
    definition::Dict{String,Any}
end

"""
A single PALS branch resolved into an ordered, occurrence-specific element list.

This is the interchange boundary used by EICViBE-style machine management:
topology remains outside TrackPad, while one selected tracking path is compiled
into a flat executable lattice.
"""
struct PALSResolvedBranch
    lattice_name::String
    name::String
    periodic::Bool
    elements::Vector{PALSResolvedElement}
    reference::Dict{String,Any}
end

"""
An executable TrackPad branch plus its source-occurrence index.

`source_index[(name, occurrence)]` maps an EICViBE/PALS occurrence to its
one-based TrackPad lattice index. Metadata-only `BeginningEle` and
`Placeholder` occurrences are intentionally absent from the index.
"""
struct CompiledBranch
    lattice::Lattice
    beam::Beam
    branch_name::String
    periodic::Bool
    source_index::Dict{Tuple{String,Int},Int}
end

mutable struct _PALSDocument
    elements::Dict{String,Dict{String,Any}}
    beamlines::Dict{String,Dict{String,Any}}
    lattices::Dict{String,Dict{String,Any}}
    uses::Vector{String}
end

_PALSDocument() = _PALSDocument(
    Dict{String,Dict{String,Any}}(),
    Dict{String,Dict{String,Any}}(),
    Dict{String,Dict{String,Any}}(),
    String[],
)

_pals_dict(x::AbstractDict) = Dict{String,Any}(string(k) => _pals_value(v) for (k, v) in x)
_pals_value(x::AbstractDict) = _pals_dict(x)
_pals_value(x::AbstractVector) = Any[_pals_value(v) for v in x]
_pals_value(x) = x

function _pals_facility(data::AbstractDict)
    root = _pals_dict(data)
    pals = get(root, "PALS", root)
    pals isa AbstractDict || throw(ArgumentError("PALS root must be a mapping"))
    return get(pals, "facility", pals)
end

function _pals_entries(facility)
    if facility isa AbstractVector
        return facility
    elseif facility isa AbstractDict
        return Any[Dict(string(k) => v) for (k, v) in facility]
    end
    throw(ArgumentError("PALS facility must be a list or mapping"))
end

function _pals_document(data::AbstractDict)
    doc = _PALSDocument()
    for item in _pals_entries(_pals_facility(data))
        item isa AbstractDict || throw(ArgumentError("Each PALS facility item must be a mapping"))
        length(item) == 1 || throw(ArgumentError("Each PALS facility item must contain one name"))
        name, raw = first(item)
        sname = string(name)
        if lowercase(sname) == "use"
            push!(doc.uses, string(raw))
            continue
        end
        raw isa AbstractDict ||
            throw(ArgumentError("PALS definition '$sname' must be a mapping"))
        definition = _pals_dict(raw)
        kind = uppercase(string(get(definition, "kind", "")))
        if kind == "BEAMLINE"
            doc.beamlines[sname] = definition
        elseif kind == "LATTICE"
            doc.lattices[sname] = definition
        else
            doc.elements[sname] = definition
        end
    end
    return doc
end

function _pals_deepmerge(base::Dict{String,Any}, override::Dict{String,Any})
    merged = deepcopy(base)
    for (key, value) in override
        if value isa AbstractDict && get(merged, key, nothing) isa AbstractDict
            merged[key] = _pals_deepmerge(
                _pals_dict(merged[key]),
                _pals_dict(value),
            )
        else
            merged[key] = deepcopy(value)
        end
    end
    return merged
end

function _pals_resolve_element_definition(name, doc::_PALSDocument;
                                           override=Dict{String,Any}(),
                                           stack=String[])
    name in stack && throw(ArgumentError("Circular PALS inheritance: $(join([stack; name], " -> "))"))
    local_definition = if haskey(doc.elements, name)
        deepcopy(doc.elements[name])
    else
        Dict{String,Any}()
    end
    merged_override = _pals_dict(override)
    inherit = get(merged_override, "inherit", get(local_definition, "inherit", nothing))
    if inherit !== nothing
        parent = string(inherit)
        haskey(doc.elements, parent) ||
            throw(ArgumentError("PALS element '$name' inherits unknown element '$parent'"))
        base, _ = _pals_resolve_element_definition(parent, doc; stack=[stack; name])
        delete!(local_definition, "inherit")
        delete!(merged_override, "inherit")
        return _pals_deepmerge(_pals_deepmerge(base, local_definition), merged_override), parent
    end
    isempty(local_definition) && !haskey(merged_override, "kind") &&
        throw(ArgumentError("Unknown PALS element '$name'"))
    return _pals_deepmerge(local_definition, merged_override), name
end

function _pals_branch_specs(lattice_name::String, lattice_def::Dict{String,Any})
    raw = get(lattice_def, "branches", Any[])
    raw isa AbstractVector ||
        throw(ArgumentError("PALS lattice '$lattice_name' branches must be a list"))
    specs = Tuple{String,Dict{String,Any}}[]
    for item in raw
        if item isa AbstractString
            push!(specs, (string(item), Dict{String,Any}()))
        elseif item isa AbstractDict && length(item) == 1
            name, definition = first(item)
            branch_def = definition === nothing ? Dict{String,Any}() :
                         definition isa AbstractDict ? _pals_dict(definition) :
                         throw(ArgumentError("PALS branch '$name' must be a mapping"))
            push!(specs, (string(name), branch_def))
        else
            throw(ArgumentError("Invalid branch entry in PALS lattice '$lattice_name'"))
        end
    end
    isempty(specs) && throw(ArgumentError("PALS lattice '$lattice_name' has no branches"))
    return specs
end

function _pals_choose_root(doc::_PALSDocument, lattice, branch, sequence)
    requested_lattice = lattice === nothing ? nothing : string(lattice)
    requested_branch = branch === nothing ? nothing : string(branch)
    if sequence !== nothing
        selected = string(sequence)
        if haskey(doc.lattices, selected)
            requested_lattice = selected
        elseif haskey(doc.beamlines, selected)
            requested_branch = selected
        else
            throw(ArgumentError("PALS lattice or BeamLine '$selected' was not found"))
        end
    end

    lattice_name = if requested_lattice !== nothing
        requested_lattice
    elseif !isempty(doc.uses)
        doc.uses[end]
    elseif !isempty(doc.lattices)
        last(collect(keys(doc.lattices)))
    else
        ""
    end

    if isempty(lattice_name)
        isempty(doc.beamlines) && throw(ArgumentError("No PALS Lattice or BeamLine found"))
        branch_name = requested_branch === nothing ? first(keys(doc.beamlines)) : requested_branch
        haskey(doc.beamlines, branch_name) ||
            throw(ArgumentError("PALS BeamLine '$branch_name' was not found"))
        return "", branch_name, deepcopy(doc.beamlines[branch_name])
    end

    haskey(doc.lattices, lattice_name) ||
        throw(ArgumentError("PALS lattice '$lattice_name' was not found"))
    specs = _pals_branch_specs(lattice_name, doc.lattices[lattice_name])
    index = requested_branch === nothing ? 1 : findfirst(spec -> spec[1] == requested_branch, specs)
    index === nothing &&
        throw(ArgumentError("PALS branch '$requested_branch' was not found in lattice '$lattice_name'"))
    branch_name, branch_override = specs[index]
    root_name = string(get(branch_override, "inherit", branch_name))
    branch_def = if haskey(branch_override, "kind") || haskey(branch_override, "line")
        _pals_deepmerge(get(doc.beamlines, root_name, Dict{String,Any}()), branch_override)
    else
        haskey(doc.beamlines, root_name) ||
            throw(ArgumentError("PALS branch '$branch_name' references unknown BeamLine '$root_name'"))
        _pals_deepmerge(doc.beamlines[root_name], branch_override)
    end
    delete!(branch_def, "inherit")
    return lattice_name, branch_name, branch_def
end

function _pals_expand_item!(result, item, doc, counts, strict, stack)
    name, override = if item isa AbstractString
        string(item), Dict{String,Any}()
    elseif item isa AbstractDict && length(item) == 1
        raw_name, raw_override = first(item)
        raw_override === nothing && (raw_override = Dict{String,Any}())
        raw_override isa AbstractDict ||
            throw(ArgumentError("PALS line item '$raw_name' must be a mapping"))
        string(raw_name), _pals_dict(raw_override)
    else
        throw(ArgumentError("Invalid PALS BeamLine item '$item'"))
    end

    repeat_count = Int(get(override, "repeat", 1))
    delete!(override, "repeat")
    iszero(repeat_count) && return result
    direction = Int(get(override, "direction", 1))
    delete!(override, "direction")
    direction in (-1, 1) || throw(ArgumentError("PALS direction must be -1 or 1"))
    direction == -1 && strict &&
        throw(ArgumentError("PALS true direction reversal is not supported by TrackPad"))

    is_line = haskey(doc.beamlines, name) ||
              uppercase(string(get(override, "kind", ""))) == "BEAMLINE"
    expanded = PALSResolvedElement[]
    if is_line
        name in stack && throw(ArgumentError("Circular PALS BeamLine reference: $(join([stack; name], " -> "))"))
        definition = _pals_deepmerge(get(doc.beamlines, name, Dict{String,Any}()), override)
        line = get(definition, "line", Any[])
        line isa AbstractVector ||
            throw(ArgumentError("PALS BeamLine '$name' line must be a list"))
        for child in line
            _pals_expand_item!(expanded, child, doc, counts, strict, [stack; name])
        end
        direction == -1 && reverse!(expanded)
    else
        definition, source_name = _pals_resolve_element_definition(name, doc; override=override)
        push!(expanded, PALSResolvedElement(name, source_name, 0, definition))
    end

    copies = abs(repeat_count)
    repeat_count < 0 && reverse!(expanded)
    for copy_index in 1:copies
        for element in expanded
            if is_line && copy_index == 1
                push!(result, element)
                continue
            end
            occurrence = get(counts, element.name, 0) + 1
            counts[element.name] = occurrence
            push!(result, PALSResolvedElement(
                element.name, element.source_name, occurrence, deepcopy(element.definition),
            ))
        end
    end
    return result
end

function _pals_reference(elements)
    for element in elements
        reference = get(element.definition, "ReferenceP", nothing)
        reference isa AbstractDict && return _pals_dict(reference)
    end
    return Dict{String,Any}()
end

"""
    resolve_pals(filename; lattice=nothing, branch=nothing, sequence=nothing,
                 strict=true) -> PALSResolvedBranch

Resolve one PALS lattice branch into an ordered list of occurrence-specific
definitions. This built-in resolver supports nested BeamLines, `repeat`, inline
definitions, inheritance, `use`, and branch selection. Full PALS expansion
(controllers, expressions, forks, and reference/floor bookkeeping) should be
performed by PALSJulia before calling TrackPad.
"""
function resolve_pals(data::AbstractDict;
                      lattice::Union{String,Symbol,Nothing}=nothing,
                      branch::Union{String,Symbol,Nothing}=nothing,
                      sequence::Union{String,Symbol,Nothing}=nothing,
                      strict::Bool=true)
    doc = _pals_document(data)
    lattice_name, branch_name, branch_def =
        _pals_choose_root(doc, lattice, branch, sequence)
    line = get(branch_def, "line", Any[])
    line isa AbstractVector ||
        throw(ArgumentError("PALS branch '$branch_name' line must be a list"))
    elements = PALSResolvedElement[]
    counts = Dict{String,Int}()
    for item in line
        _pals_expand_item!(elements, item, doc, counts, strict, String[])
    end
    periodic = Bool(get(branch_def, "periodic", false))
    return PALSResolvedBranch(
        lattice_name, branch_name, periodic, elements, _pals_reference(elements),
    )
end

function resolve_pals(filename::AbstractString; kwargs...)
    data = YAML.load_file(filename; dicttype=Dict{String,Any})
    data isa AbstractDict || throw(ArgumentError("PALS document root must be a mapping"))
    return resolve_pals(data; kwargs...)
end

"""
    resolve_pals_full(filename; kwargs...) -> PALSResolvedBranch

Parse and fully expand a PALS file with PALSJulia. This method is supplied by
the optional `TrackPadPALSJuliaExt` extension when PALSJulia is loaded.
"""
function resolve_pals_full end

function _pals_species_defaults(species)
    key = lowercase(replace(string(species), "_" => "", "-" => ""))
    key in ("electron", "e") && return Float64(M_ELECTRON), -1.0
    key in ("positron", "e+") && return Float64(M_ELECTRON), 1.0
    key in ("proton", "p") && return Float64(M_PROTON), 1.0
    key in ("antiproton", "pbar") && return Float64(M_PROTON), -1.0
    return Float64(M_ELECTRON), -1.0
end

function _pals_beam(reference;
                    beam_energy::Union{Real,Nothing}=nothing,
                    mass::Union{Real,Nothing}=nothing,
                    charge::Union{Real,Nothing}=nothing)
    inferred_mass, inferred_charge =
        _pals_species_defaults(get(reference, "species_ref", "electron"))
    inferred_mass = Float64(get(reference, "mass_ref", inferred_mass))
    inferred_charge = Float64(get(reference, "charge_ref", inferred_charge))
    beam_mass = mass === nothing ? inferred_mass : Float64(mass)
    beam_charge = charge === nothing ? inferred_charge : Float64(charge)
    kinetic = if beam_energy !== nothing
        Float64(beam_energy)
    elseif haskey(reference, "pc_ref")
        pc = Float64(reference["pc_ref"])
        sqrt(pc^2 + beam_mass^2) - beam_mass
    elseif haskey(reference, "E_tot_ref")
        Float64(reference["E_tot_ref"]) - beam_mass
    else
        1.0e9
    end
    kinetic >= 0 || throw(ArgumentError("PALS reference energy is below rest mass"))
    return Beam(kinetic; mass=beam_mass, charge=beam_charge)
end

"""
    read_pals(filename; lattice=nothing, branch=nothing, sequence=nothing,
              beam_energy=nothing, mass=nothing, charge=nothing, strict=true)

Resolve and compile one PALS branch into a TrackPad `(Lattice, Beam)` pair.
Explicit beam keywords override `BeginningEle.ReferenceP`; otherwise the
reference species and momentum/total energy are used.
"""
function read_pals(filename::AbstractString;
                   lattice::Union{String,Symbol,Nothing}=nothing,
                   branch::Union{String,Symbol,Nothing}=nothing,
                   sequence::Union{String,Symbol,Nothing}=nothing,
                   beam_energy::Union{Real,Nothing}=nothing,
                   mass::Union{Real,Nothing}=nothing,
                   charge::Union{Real,Nothing}=nothing,
                   strict::Bool=true)
    resolved = resolve_pals(
        filename; lattice=lattice, branch=branch, sequence=sequence, strict=strict,
    )
    compiled = compile_branch(
        resolved;
        beam_energy=beam_energy, mass=mass, charge=charge, strict=strict,
    )
    return compiled.lattice, compiled.beam
end

"""
    compile_branch(resolved::PALSResolvedBranch; kwargs...) -> CompiledBranch

Compile a resolved PALS/EICViBE branch into TrackPad's flat executable lattice
while retaining an occurrence-to-index map for diagnostics and live updates.
"""
function compile_branch(resolved::PALSResolvedBranch;
                        beam_energy::Union{Real,Nothing}=nothing,
                        mass::Union{Real,Nothing}=nothing,
                        charge::Union{Real,Nothing}=nothing,
                        strict::Bool=true)
    beam = _pals_beam(
        resolved.reference; beam_energy=beam_energy, mass=mass, charge=charge,
    )
    elements = AbstractElement[]
    source_index = Dict{Tuple{String,Int},Int}()
    for occurrence in resolved.elements
        kind = uppercase(string(get(occurrence.definition, "kind", "DRIFT")))
        kind in ("BEGINNINGELE", "PLACEHOLDER") && continue
        push!(elements, _pals_build_element(
            occurrence.name, occurrence.definition;
            strict=strict, beam_energy=beam.energy,
        ))
        source_index[(occurrence.name, occurrence.occurrence)] = length(elements)
    end
    name = isempty(resolved.lattice_name) ? resolved.name : resolved.lattice_name
    lattice = Lattice(elements; name=Symbol(name))
    return CompiledBranch(
        lattice, beam, resolved.name, resolved.periodic, source_index,
    )
end

"""
    compile_branch(elements; machine_name="machine", branch_name="main",
                   periodic=false, reference=Dict(), strict=true)

Compile an in-memory EICViBE-style element list. Each entry must be a mapping
with a `name` field and the same canonical `kind`, `length`, and parameter-group
fields used by PALS. This is the intended JuliaCall boundary for EICViBE.
"""
function compile_branch(elements::AbstractVector;
                        machine_name::Union{String,Symbol}="machine",
                        branch_name::Union{String,Symbol}="main",
                        periodic::Bool=false,
                        reference::AbstractDict=Dict{String,Any}(),
                        beam_energy::Union{Real,Nothing}=nothing,
                        mass::Union{Real,Nothing}=nothing,
                        charge::Union{Real,Nothing}=nothing,
                        strict::Bool=true)
    counts = Dict{String,Int}()
    occurrences = PALSResolvedElement[]
    for raw in elements
        raw isa AbstractDict ||
            throw(ArgumentError("Interchange elements must be mappings"))
        definition = _pals_dict(raw)
        haskey(definition, "name") ||
            throw(ArgumentError("Interchange element is missing its name"))
        name = string(pop!(definition, "name"))
        source_name = string(pop!(definition, "source_name", name))
        occurrence = get(counts, name, 0) + 1
        counts[name] = occurrence
        push!(occurrences, PALSResolvedElement(
            name, source_name, occurrence, definition,
        ))
    end
    resolved = PALSResolvedBranch(
        string(machine_name), string(branch_name), periodic,
        occurrences, _pals_dict(reference),
    )
    return compile_branch(
        resolved;
        beam_energy=beam_energy, mass=mass, charge=charge, strict=strict,
    )
end

_pals_get(d, names...; default=0.0) =
    something((get(d, name, nothing) for name in names)..., default)

function _pals_strength(group, order::Int, len::Float64; skew::Bool=false)
    prefix = skew ? "Ks" : "Kn"
    normalized = get(group, "$prefix$order", nothing)
    normalized !== nothing && return Float64(normalized)
    integrated = get(group, "$(prefix)$(order)L", nothing)
    integrated === nothing && return 0.0
    iszero(len) &&
        throw(ArgumentError("Integrated $prefix$order strength requires a nonzero length"))
    return Float64(integrated) / len
end

function _pals_bend_length(d, bp)
    haskey(d, "length") && return Float64(d["length"])
    angle = Float64(get(bp, "angle_ref", 0.0))
    haskey(bp, "g_ref") && !iszero(bp["g_ref"]) &&
        return abs(angle / Float64(bp["g_ref"]))
    haskey(bp, "radius_ref") &&
        return abs(angle * Float64(bp["radius_ref"]))
    if haskey(bp, "L_chord")
        chord = Float64(bp["L_chord"])
        return iszero(angle) ? chord : chord * abs(angle) / (2sin(abs(angle) / 2))
    end
    if haskey(bp, "L_rectangle")
        chord = Float64(bp["L_rectangle"])
        return iszero(angle) ? chord : chord * abs(angle) / (2sin(abs(angle) / 2))
    end
    !iszero(angle) && throw(ArgumentError(
        "PALS Bend needs length, g_ref, radius_ref, L_chord, or L_rectangle",
    ))
    return 0.0
end

function _pals_build_element(name::String, d::Dict;
                             strict::Bool=true, beam_energy::Real=1.0e9)
    kind = string(get(d, "kind", "Drift"))
    bp   = get(d, "BendP", Dict{String,Any}())
    len  = uppercase(kind) in ("BEND", "SBEND", "RBEND") ?
           _pals_bend_length(d, bp) : Float64(get(d, "length", 0.0))
    mmp  = get(d, "MagneticMultipoleP", Dict{String,Any}())
    rfp  = get(d, "RFP",               Dict{String,Any}())
    solp = get(d, "SolenoidP",         Dict{String,Any}())
    kp   = get(d, "KickerP",           Dict{String,Any}())
    bbp  = get(d, "BeamBeamP",         Dict{String,Any}())
    k = uppercase(kind)

    if k == "DRIFT"
        return Drift(len; name=Symbol(name))
    elseif k in ("MARKER", "INSTRUMENT", "MONITOR")
        return Marker(; name=Symbol(name))
    elseif k == "QUADRUPOLE"
        k1  = _pals_strength(mmp, 1, len)
        k1s = _pals_strength(mmp, 1, len; skew=true)
        pa  = k1s != 0 ? [0.0, k1s, 0.0, 0.0] : nothing
        return Quadrupole(len, k1; name=Symbol(name), polynom_a=pa)
    elseif k == "SEXTUPOLE"
        k2  = _pals_strength(mmp, 2, len)
        k2s = _pals_strength(mmp, 2, len; skew=true)
        pa  = k2s != 0 ? [0.0, 0.0, k2s, 0.0] : nothing
        return Sextupole(len, k2; name=Symbol(name), polynom_a=pa)
    elseif k == "OCTUPOLE"
        k3 = _pals_strength(mmp, 3, len)
        k3s = _pals_strength(mmp, 3, len; skew=true)
        pa = k3s != 0 ? [0.0, 0.0, 0.0, k3s] : nothing
        return Octupole(len, k3; name=Symbol(name), polynom_a=pa)
    elseif k in ("BEND", "SBEND")
        angle = Float64(get(bp, "angle_ref", 0.0))
        e1    = Float64(get(bp, "e1",        0.0))
        e2    = Float64(get(bp, "e2",        0.0))
        fint1 = Float64(get(bp, "edge1_int", 0.0))
        fint2 = Float64(get(bp, "edge2_int", fint1))
        hgap  = Float64(get(bp, "hgap",      0.0))
        k1    = _pals_strength(mmp, 1, len)
        k1s   = _pals_strength(mmp, 1, len; skew=true)
        pb    = k1 != 0 ? [0.0, k1, 0.0, 0.0] : nothing
        pa    = k1s != 0 ? [0.0, k1s, 0.0, 0.0] : nothing
        return SBend(len, angle, e1, e2; name=Symbol(name),
                     fint1=fint1, fint2=fint2, gap=2*hgap,
                     polynom_a=pa, polynom_b=pb)
    elseif k == "RBEND"
        angle = Float64(get(bp, "angle_ref", 0.0))
        e1    = Float64(get(bp, "e1",        0.0))
        e2    = Float64(get(bp, "e2",        0.0))
        fint1 = Float64(get(bp, "edge1_int", 0.0))
        fint2 = Float64(get(bp, "edge2_int", fint1))
        hgap  = Float64(get(bp, "hgap",      0.0))
        return RBend(len, angle; name=Symbol(name),
                     fint1=fint1, fint2=fint2, gap=2*hgap)
    elseif k == "RFCAVITY"
        volt  = Float64(get(rfp, "voltage",   0.0))
        freq  = Float64(get(rfp, "frequency", 0.0))
        h     = Float64(get(rfp, "harmon",    1.0))
        phase = Float64(get(rfp, "phase",     0.0))
        lag   = freq > 0 ? phase * _C_IO / freq : 0.0
        return RFCavity(
            len, volt, freq, lag;
            name=Symbol(name), h=h, energy=Float64(beam_energy),
        )
    elseif k == "CRABCAVITY"
        volt  = Float64(get(rfp, "voltage",   0.0))
        freq  = Float64(get(rfp, "frequency", 0.0))
        phase = Float64(get(rfp, "phase",     0.0))
        return CrabCavity(
            len; name=Symbol(name), volt=volt, freq=freq,
            phi=phase*2π, energy=Float64(beam_energy),
        )
    elseif k == "SOLENOID"
        ks = Float64(get(solp, "Ksol", 0.0))
        return Solenoid(len, ks; name=Symbol(name))
    elseif k == "KICKER"
        hkick = Float64(_pals_get(kp, "Kn0", "hkick"; default=0.0))
        vkick = Float64(_pals_get(kp, "Ks0", "vkick"; default=0.0))
        return Corrector(len, hkick, vkick; name=Symbol(name))
    elseif k == "HKICKER"
        xkick = Float64(_pals_get(kp, "Kn0", "hkick"; default=0.0))
        return HKicker(; name=Symbol(name), L=len, xkick=xkick)
    elseif k == "VKICKER"
        ykick = Float64(_pals_get(kp, "Ks0", "vkick"; default=0.0))
        return VKicker(; name=Symbol(name), L=len, ykick=ykick)
    elseif k in ("MULTIPOLE", "THINMULTIPOLE")
        pa = Float64[get(mmp, "Ks$(i-1)", 0.0) for i in 1:4]
        pb = Float64[get(mmp, "Kn$(i-1)", 0.0) for i in 1:4]
        return ThinMultipole(len, pa, pb; name=Symbol(name))
    elseif k == "BEAMBEAM"
        sx = Float64(get(bbp, "sigma_x",    1e-3))
        sy = Float64(get(bbp, "sigma_y",    1e-3))
        A  = Float64(get(bbp, "N_particle", 0.0))
        return StrongThinGaussianBeam(A, sx, sy; name=Symbol(name))
    elseif k == "WIGGLER"
        return Wiggler(len; name=Symbol(name))
    else
        strict && throw(ArgumentError(
            "PALS element '$name' has unsupported kind '$kind'",
        ))
        @warn "PALS: unsupported element kind '$kind' for '$name'; using Drift"
        return Drift(len; name=Symbol(name))
    end
end


# ============================================================
# ==================  MAD-X Reader  ==========================
# ============================================================

"""
    read_madx(filename; sequence=nothing, beam_energy=nothing,
              mass=nothing, charge=nothing, num_int_steps=10,
              strict=true) -> (Lattice, Beam)

Read a MAD-X lattice file (`.seq`, `.mad`, `.madx`) and return a `(Lattice, Beam)` pair.

Supports both `SEQUENCE ... ENDSEQUENCE` blocks (with AT positions) and the
`NAME: LINE = (elem1, n*elem2, ...)` syntax with nested expansion.

Supported constructs: variable assignments (`:=` deferred, `=` immediate),
element definitions, `LINE`/`SEQUENCE` definitions, `BEAM` commands, `USE,PERIOD=`,
`//` and `! ...` comments, `/* ... */` block comments, and arithmetic expressions.

# Unit conversions (MAD-X → TrackPad)
| MAD-X | Unit       | TrackPad | Unit          |
|-------|-----------|----------|---------------|
| VOLT  | MV        | volt     | V             |
| FREQ  | MHz       | freq     | Hz            |
| LAG   | cycles    | lag      | m (×C/freq)   |
| HGAP  | m (half)  | gap      | m (full, ×2)  |
"""
function read_madx(filename::AbstractString;
                   sequence::Union{String,Symbol,Nothing} = nothing,
                   beam_energy::Union{Real,Nothing} = nothing,
                   mass::Union{Real,Nothing} = nothing,
                   charge::Union{Real,Nothing} = nothing,
                   num_int_steps::Int = 10,
                   strict::Bool = true)
    num_int_steps > 0 ||
        throw(ArgumentError("num_int_steps must be positive"))
    text = read(filename, String)
    tokens = _madx_tokenize(text)

    # Variables table (pre-populated with constants)
    vars = Dict{String,Float64}(
        "pi"    => π,
        "twopi" => 2π,
        "e"     => exp(1.0),
    )

    # Two-pass variable pre-scan: handles variables defined after their use
    _madx_prescan_vars!(tokens, vars)
    _madx_prescan_vars!(tokens, vars)   # second pass resolves forward refs

    type_defs  = Dict{String,Dict{String,Any}}()    # element name → params
    line_defs  = Dict{String,Vector{Tuple{Int,String}}}()  # LINE name → [(n,elem)]
    sequences  = Dict{String,Vector{_MADXPlacement}}()     # SEQUENCE name → placements
    use_period = Ref{String}("")
    beam_info = Dict{String,Any}()

    _madx_parse!(
        tokens, vars, type_defs, line_defs, sequences, use_period, beam_info,
    )

    # Choose root sequence / line
    seq_name = if sequence !== nothing
        lowercase(string(sequence))
    elseif !isempty(use_period[])
        use_period[]
    elseif !isempty(sequences)
        first(keys(sequences))
    elseif !isempty(line_defs)
        first(keys(line_defs))
    else
        error("No sequence or LINE definition found in MAD-X file")
    end

    # Expand to ordered flat list of element names
    elem_names = if haskey(sequences, seq_name)
        _madx_order_sequence(seq_name, sequences)
    elseif haskey(line_defs, seq_name)
        _madx_expand_line(seq_name, line_defs)
    else
        error("Sequence/LINE '$seq_name' not found in MAD-X file")
    end

    beam = _madx_beam(
        beam_info; beam_energy=beam_energy, mass=mass, charge=charge,
    )

    # Build TrackPad elements (skip unknowns with a warning)
    elements = AbstractElement[]
    unknown_types = Set{String}()
    for n in elem_names
        if !haskey(type_defs, n)
            n in unknown_types || @warn "MAD-X: element '$n' has no definition — skipped"
            push!(unknown_types, n)
            continue
        end
        push!(elements, _madx_build_element(
            n, type_defs[n], type_defs;
            strict=strict, beam_energy=beam.energy,
            num_int_steps=num_int_steps,
        ))
    end

    lat  = Lattice(elements; name = Symbol(seq_name))
    return lat, beam
end

function _madx_beam(beam_info;
                    beam_energy::Union{Real,Nothing}=nothing,
                    mass::Union{Real,Nothing}=nothing,
                    charge::Union{Real,Nothing}=nothing)
    inferred_mass, inferred_charge =
        _pals_species_defaults(get(beam_info, "particle", "electron"))
    beam_mass = mass === nothing ? inferred_mass : Float64(mass)
    beam_charge = charge === nothing ? inferred_charge : Float64(charge)
    kinetic = if beam_energy !== nothing
        Float64(beam_energy)
    elseif haskey(beam_info, "pc")
        pc = Float64(beam_info["pc"]) * 1.0e9
        sqrt(pc^2 + beam_mass^2) - beam_mass
    elseif haskey(beam_info, "energy")
        Float64(beam_info["energy"]) * 1.0e9 - beam_mass
    else
        1.0e9
    end
    kinetic >= 0 || throw(ArgumentError("MAD-X beam energy is below rest mass"))
    return Beam(kinetic; mass=beam_mass, charge=beam_charge)
end

# ---- MAD-X placement record ----
struct _MADXPlacement
    name::String
    at::Float64
end

# ---- Tokenizer ----
function _madx_tokenize(text::AbstractString)
    # Remove comments: // ... EOL,  ! ... EOL,  /* ... */
    text = replace(text, r"//.*"    => " ")
    text = replace(text, r"!.*"     => " ")
    text = replace(text, r"/\*.*?\*/"s => " ")
    text = lowercase(text)

    tokens = String[]
    i = firstindex(text)
    while i <= lastindex(text)
        c = text[i]
        if isspace(c)
            i = nextind(text, i); continue
        elseif c == '"' || c == '\''
            j = nextind(text, i)
            while j <= lastindex(text) && text[j] != c
                j = nextind(text, j)
            end
            push!(tokens, text[i:j])
            i = j <= lastindex(text) ? nextind(text, j) : lastindex(text)+1
        elseif isdigit(c) || (c == '.' && i < lastindex(text) &&
                               isdigit(text[nextind(text,i)]))
            j = i
            while j <= lastindex(text)
                ch = text[j]
                nj = j < lastindex(text) ? nextind(text, j) : j
                if ch in "eE" && j > i
                    j = nj; continue
                end
                if ch in "+-" && j > i
                    prev = text[prevind(text, j)]
                    prev in "eE" || break
                    j = nj; continue
                end
                isdigit(ch) || ch in ".eE" || break
                j = nj
            end
            push!(tokens, text[i:prevind(text,j)])
            i = j
        elseif isletter(c) || c == '_'
            j = i
            while j <= lastindex(text)
                ch = text[j]
                (isletter(ch) || isdigit(ch) || ch in ('_', '.')) || break
                j = nextind(text, j)
            end
            push!(tokens, text[i:prevind(text,j)])
            i = j
        elseif c == ':' && i < lastindex(text) &&
               text[nextind(text,i)] == '='
            push!(tokens, ":=")
            i = nextind(text, nextind(text, i))
        else
            push!(tokens, string(c))
            i = nextind(text, i)
        end
    end
    return tokens
end

# ---- Variable pre-scanner ----
# Walk the token stream and evaluate all  name = expr  and  name := expr  assignments.
# This handles the common MAD-X pattern of variables defined *after* they are
# referenced in element defs via  :=  (deferred evaluation).
function _madx_prescan_vars!(tokens, vars)
    n = length(tokens)
    i = 1
    while i <= n
        statement_start = i == 1 || tokens[i-1] == ";"
        if statement_start && i+1 <= n && tokens[i+1] in ("=", ":=")
            name = tokens[i]
            if !isempty(name) && (isletter(name[1]) || name[1] == '_')
                s = _MADXState(tokens, i+2)
                try
                    val = _eval_expr!(s, vars)
                    isa(val, Number) && (vars[name] = Float64(val))
                catch
                end
                i = i + 2
                continue
            end
        end
        i += 1
    end
end

# ---- Parser state ----
mutable struct _MADXState
    tokens::Vector{String}
    pos::Int
end

_peek(s::_MADXState)  = s.pos <= length(s.tokens) ? s.tokens[s.pos] : ""
_next!(s::_MADXState) = (t = _peek(s); s.pos += 1; t)
_done(s::_MADXState)  = s.pos > length(s.tokens)

function _madx_consume_until_semi!(s)
    while !_done(s) && _peek(s) != ";"; _next!(s); end
    !_done(s) && _next!(s)
end

function _madx_parse!(tokens, vars, type_defs, line_defs, sequences, use_period,
                      beam_info)
    s = _MADXState(tokens, 1)
    while !_done(s)
        tok = _peek(s)
        (tok == ";" || tok == "") && (_next!(s); continue)

        # NAME := expr
        if s.pos+1 <= length(s.tokens) && s.tokens[s.pos+1] == ":="
            name = _next!(s); _next!(s)
            vars[name] = _eval_expr!(s, vars)
            _madx_consume_until_semi!(s); continue
        end
        # NAME = expr
        if s.pos+1 <= length(s.tokens) && s.tokens[s.pos+1] == "="
            name = _next!(s); _next!(s)
            vars[name] = _eval_expr!(s, vars)
            _madx_consume_until_semi!(s); continue
        end
        # NAME : definition
        if s.pos+1 <= length(s.tokens) && s.tokens[s.pos+1] == ":"
            name = _next!(s); _next!(s)
            _madx_parse_def!(
                s, name, vars, type_defs, line_defs, sequences, beam_info,
            )
            continue
        end
        # BEAM command
        if tok == "beam"
            _next!(s); _madx_parse_beam!(s, vars, beam_info); continue
        end
        # USE command — extract period/sequence name
        if tok == "use"
            _next!(s)
            _madx_parse_use!(s, use_period)
            continue
        end
        # Unknown statement — skip until semicolon
        _next!(s); _madx_consume_until_semi!(s)
    end
end

function _madx_parse_def!(s, name, vars, type_defs, line_defs, sequences,
                          beam_info)
    kw = uppercase(_next!(s))

    if kw == "BEAM"
        _madx_parse_beam!(s, vars, beam_info)
        return
    end

    if kw == "LINE"
        # NAME: LINE = (elem1, n*elem2, ...)
        _peek(s) == "=" && _next!(s)
        items = _madx_parse_line_list!(s)
        line_defs[name] = items
        _madx_consume_until_semi!(s)
        return
    end

    if kw == "SEQUENCE"
        _madx_parse_params!(s, vars)
        _madx_consume_until_semi!(s)
        placements = _MADXPlacement[]
        while !_done(s) && _peek(s) != "endsequence"
            _madx_parse_seq_elem!(s, placements, type_defs, vars)
        end
        !_done(s) && _next!(s)
        _madx_consume_until_semi!(s)
        sequences[name] = placements
        return
    end

    # Regular element definition
    params = _madx_parse_params!(s, vars)
    params["_type"] = kw
    if haskey(type_defs, name)
        merge!(type_defs[name], params)
    else
        type_defs[name] = params
    end
    _madx_consume_until_semi!(s)
end

# Parse a LINE item list: (elem1, 3*elem2, -elem3, ...)
# Returns Vector{Tuple{Int,String}} where Int is repeat count.
function _madx_parse_line_list!(s)
    items = Tuple{Int,String}[]
    _peek(s) == "(" && _next!(s)
    while !_done(s) && _peek(s) != ")" && _peek(s) != ";"
        tok = _peek(s)
        tok == "," && (_next!(s); continue)
        _next!(s)  # consume first token
        # Repetition:  n*elemname
        if _peek(s) == "*"
            n = tryparse(Int, tok)
            n === nothing && (n = 1)
            _next!(s)   # consume *
            ename = _next!(s)
            push!(items, (n, ename))
        # Negation:  -elemname  → still traverse, just reversed order
        elseif tok == "-"
            ename = _peek(s)
            if !isempty(ename) && ename != "," && ename != ")"
                _next!(s)
                push!(items, (1, ename))
            end
        else
            push!(items, (1, tok))
        end
    end
    _peek(s) == ")" && _next!(s)
    return items
end

function _madx_parse_params!(s, vars)
    params = Dict{String,Any}()
    _peek(s) == "," && _next!(s)
    while !_done(s) && _peek(s) != ";" && _peek(s) != "endsequence"
        key = _peek(s)
        (key == "," || key == "") && (_next!(s); continue)
        key == ";" && break
        _next!(s)
        if _peek(s) in ("=", ":=")
            _next!(s)
            params[key] = _eval_expr!(s, vars)
        else
            params[key] = true
        end
        _peek(s) == "," && _next!(s)
    end
    return params
end

function _madx_parse_seq_elem!(s, placements, type_defs, vars)
    tok = _peek(s)
    (tok == ";" || tok == "") && (_next!(s); return)
    name = _next!(s)
    if _peek(s) == ":"
        _next!(s)
        kw = uppercase(_next!(s))
        params = _madx_parse_params!(s, vars)
        params["_type"] = kw
        type_defs[name] = params
    else
        _peek(s) == "," && _next!(s)
        extra = _madx_parse_params!(s, vars)
        if haskey(type_defs, name)
            merge!(type_defs[name], extra)
        else
            type_defs[name] = extra
        end
    end
    at = haskey(type_defs, name) ? Float64(get(type_defs[name], "at", 0.0)) : 0.0
    push!(placements, _MADXPlacement(name, at))
    _madx_consume_until_semi!(s)
end

function _madx_parse_beam!(s, vars, beam_info)
    _peek(s) == "," && _next!(s)
    while !_done(s) && _peek(s) != ";"
        _peek(s) == "," && (_next!(s); continue)
        key = _next!(s)
        if _peek(s) in ("=", ":=")
            _next!(s)
            if key == "particle"
                beam_info[key] = strip(_next!(s), ['"', '\''])
            else
                beam_info[key] = _eval_expr!(s, vars)
            end
        else
            beam_info[key] = true
        end
        _peek(s) == "," && _next!(s)
    end
    _madx_consume_until_semi!(s)
end

function _madx_parse_use!(s, use_period)
    _peek(s) == "," && _next!(s)
    while !_done(s) && _peek(s) != ";"
        key = _peek(s)
        key == "," && (_next!(s); continue)
        _next!(s)
        if _peek(s) in ("=", ":=")
            _next!(s)
            val = _peek(s)
            if key in ("period", "sequence") && !isempty(val) &&
               (isletter(val[1]) || val[1] == '_')
                isempty(use_period[]) && (use_period[] = val)
                _next!(s)
            else
                _next!(s)  # skip value
            end
        end
    end
    _madx_consume_until_semi!(s)
end

# ---- LINE recursive expansion ----
function _madx_expand_line(line_name, line_defs; _depth=0, _visited=Set{String}())
    _depth > 200 && error("LINE expansion depth exceeded (circular reference?)")
    line_name in _visited && error("Circular LINE reference: '$line_name'")
    result = String[]
    push!(_visited, line_name)
    for (count, ename) in line_defs[line_name]
        if haskey(line_defs, ename)
            sub = _madx_expand_line(ename, line_defs;
                                    _depth=_depth+1,
                                    _visited=copy(_visited))
            for _ in 1:count; append!(result, sub); end
        else
            for _ in 1:count; push!(result, ename); end
        end
    end
    return result
end

# ---- Expression evaluator ----
function _eval_expr!(s::_MADXState, vars::Dict)
    left = _eval_term!(s, vars)
    while !_done(s) && _peek(s) in ("+", "-")
        op = _next!(s)
        right = _eval_term!(s, vars)
        left = op == "+" ? left + right : left - right
    end
    return left
end

function _eval_term!(s, vars)
    left = _eval_factor!(s, vars)
    while !_done(s) && _peek(s) in ("*", "/")
        op = _next!(s)
        right = _eval_factor!(s, vars)
        left = op == "*" ? left * right : left / right
    end
    return left
end

function _eval_factor!(s, vars)
    b = _eval_base!(s, vars)
    if !_done(s) && _peek(s) == "^"
        _next!(s)
        return b ^ _eval_factor!(s, vars)
    end
    return b
end

function _eval_base!(s, vars)
    tok = _peek(s)
    if tok == "-"; _next!(s); return -_eval_base!(s, vars); end
    if tok == "+"; _next!(s); return  _eval_base!(s, vars); end
    if tok == "("
        _next!(s)
        v = _eval_expr!(s, vars)
        _peek(s) == ")" && _next!(s)
        return v
    end
    if !isempty(tok) && (isdigit(tok[1]) || tok[1] == '.')
        _next!(s); return parse(Float64, tok)
    end
    name = _next!(s)
    if _peek(s) == "("
        _next!(s)
        arg1 = _eval_expr!(s, vars)
        arg2 = nothing
        if _peek(s) == ","
            _next!(s); arg2 = _eval_expr!(s, vars)
        end
        _peek(s) == ")" && _next!(s)
        return _call_builtin(name, arg1, arg2)
    end
    return Float64(get(vars, name, 0.0))
end

function _call_builtin(f, a, b)
    f == "sqrt"  && return sqrt(max(a, 0.0))
    f == "sin"   && return sin(a)
    f == "cos"   && return cos(a)
    f == "tan"   && return tan(a)
    f == "asin"  && return asin(clamp(a, -1.0, 1.0))
    f == "acos"  && return acos(clamp(a, -1.0, 1.0))
    f == "atan"  && return b === nothing ? atan(a) : atan(a, b)
    f == "exp"   && return exp(a)
    f == "log"   && return log(abs(a))
    f == "log10" && return log10(abs(a))
    f == "abs"   && return abs(a)
    f == "sign"  && return sign(a)
    f == "floor" && return floor(a)
    f == "ceil"  && return ceil(a)
    f == "round" && return round(a)
    f == "table" && return 0.0
    @warn "Unknown MAD-X function '$f' — returning 0"
    return 0.0
end

# ---- Sequence ordering ----
function _madx_order_sequence(seq_name, sequences)
    placements = sequences[seq_name]
    return [p.name for p in sort(placements; by = p -> p.at)]
end

# ---- Standard MAD-X base types ----
const _MADX_BASE_TYPES = Set(("DRIFT","MARKER","QUADRUPOLE","SEXTUPOLE","OCTUPOLE",
    "SBEND","RBEND","RFCAVITY","CRABCAVITY","SOLENOID","KICKER","HKICKER","VKICKER",
    "MULTIPOLE","DIPEDGE","MONITOR","INSTRUMENT","HMONITOR","VMONITOR","IP",
    "SEQUENCE","LINE","BEAM"))

function _madx_resolve_base(type_name, type_defs)
    uppercase(type_name) in _MADX_BASE_TYPES && return Dict{String,Any}()
    lname = lowercase(type_name)
    haskey(type_defs, lname) && return type_defs[lname]
    return Dict{String,Any}()
end

# ---- MAD-X element builder ----
function _madx_build_element(name::String, d::Dict, type_defs::Dict;
                             strict::Bool=true, beam_energy::Real=1.0e9,
                             num_int_steps::Int=10)
    type_kw = uppercase(get(d, "_type", "DRIFT"))
    base    = _madx_resolve_base(type_kw, type_defs)
    p       = merge(base, d)
    mtype   = uppercase(get(base, "_type", type_kw))
    mtype in _MADX_BASE_TYPES || (mtype = type_kw)

    L      = Float64(get(p, "l",      0.0))
    k1     = Float64(get(p, "k1",     0.0))
    k1s    = Float64(get(p, "k1s",    0.0))
    k2     = Float64(get(p, "k2",     0.0))
    k2s    = Float64(get(p, "k2s",    0.0))
    k3     = Float64(get(p, "k3",     0.0))
    k3s    = Float64(get(p, "k3s",    0.0))
    angle  = Float64(get(p, "angle",  0.0))
    e1     = Float64(get(p, "e1",     0.0))
    e2     = Float64(get(p, "e2",     0.0))
    fint   = Float64(get(p, "fint",   0.0))
    fintx  = Float64(get(p, "fintx", -1.0))
    fint2  = fintx < 0 ? fint : fintx
    hgap   = Float64(get(p, "hgap",   0.0))
    gap    = 2.0 * hgap
    volt   = Float64(get(p, "volt",   0.0)) * 1e6
    freq   = Float64(get(p, "freq",   0.0)) * 1e6
    lag_c  = Float64(get(p, "lag",    0.0))
    lag_m  = freq > 0 ? lag_c * _C_IO / freq : 0.0
    h_     = Float64(get(p, "harmon", 1.0))
    ks     = Float64(get(p, "ks",     0.0))
    hkick  = Float64(get(p, "hkick",  0.0))
    vkick  = Float64(get(p, "vkick",  0.0))
    kick   = Float64(get(p, "kick",   0.0))
    sname  = Symbol(name)

    if mtype in ("DRIFT", "INSTRUMENT", "MONITOR", "HMONITOR", "VMONITOR")
        return L > 0 ? Drift(L; name=sname) : Marker(; name=sname)
    elseif mtype in ("MARKER", "IP")
        return Marker(; name=sname)
    elseif mtype == "QUADRUPOLE"
        pa = k1s != 0 ? [k1s, 0.0, 0.0, 0.0] : nothing
        return Quadrupole(
            L, k1;
            name=sname, polynom_a=pa, num_int_steps=num_int_steps,
        )
    elseif mtype == "SEXTUPOLE"
        pa = k2s != 0 ? [0.0, k2s, 0.0, 0.0] : nothing
        return Sextupole(
            L, k2;
            name=sname, polynom_a=pa, num_int_steps=num_int_steps,
        )
    elseif mtype == "OCTUPOLE"
        pa = k3s != 0 ? [0.0, 0.0, k3s, 0.0] : nothing
        return Octupole(
            L, k3;
            name=sname, polynom_a=pa, num_int_steps=num_int_steps,
        )
    elseif mtype == "SBEND"
        pb = k1 != 0 ? [0.0, k1, 0.0, 0.0] : nothing
        pa = k1s != 0 ? [k1s, 0.0, 0.0, 0.0] : nothing
        return SBend(L, angle, e1, e2; name=sname,
                     fint1=fint, fint2=fint2, gap=gap,
                     polynom_a=pa, polynom_b=pb,
                     num_int_steps=num_int_steps)
    elseif mtype == "RBEND"
        arc_length = iszero(angle) ? L : L * angle / (2sin(angle / 2))
        return RBend(
            arc_length, angle;
            name=sname, fint1=fint, fint2=fint2, gap=gap,
            num_int_steps=num_int_steps,
        )
    elseif mtype == "RFCAVITY"
        return RFCavity(
            L, volt, freq, lag_m;
            name=sname, h=h_, energy=Float64(beam_energy),
        )
    elseif mtype == "CRABCAVITY"
        phi = lag_c * 2π
        return CrabCavity(
            L; name=sname, volt=volt, freq=freq,
            phi=phi, energy=Float64(beam_energy),
        )
    elseif mtype == "SOLENOID"
        return Solenoid(L, ks; name=sname)
    elseif mtype == "KICKER"
        return Corrector(L, hkick, vkick; name=sname)
    elseif mtype == "HKICKER"
        xk = kick != 0 ? kick : hkick
        return HKicker(; name=sname, L=L, xkick=xk)
    elseif mtype == "VKICKER"
        yk = kick != 0 ? kick : vkick
        return VKicker(; name=sname, L=L, ykick=yk)
    elseif mtype == "MULTIPOLE"
        pb = Float64[Float64(get(p, "knl[$(i-1)]", 0.0)) for i in 1:4]
        pa = Float64[Float64(get(p, "ksl[$(i-1)]", 0.0)) for i in 1:4]
        return ThinMultipole(
            L, pa, pb;
            name=sname, num_int_steps=num_int_steps,
        )
    elseif mtype == "DIPEDGE"
        return Marker(; name=sname)
    else
        strict && throw(ArgumentError(
            "MAD-X element '$name' has unsupported type '$mtype'",
        ))
        @warn "MAD-X: unknown type '$mtype' for '$name'; using Drift/Marker"
        return L > 0 ? Drift(L; name=sname) : Marker(; name=sname)
    end
end


# ============================================================
# ==================  PALS YAML Writer  ======================
# ============================================================

"""
    write_pals(filename, lat; beam=nothing, lattice_name=string(lat.name),
               branch_name="main", periodic=false)

Write a canonical PALS document containing a `BeginningEle`, one `BeamLine`,
one `Lattice`, and a final `use` statement. Repeated TrackPad element names are
given occurrence-specific PALS names so no parameters are lost.
"""
function write_pals(filename::AbstractString, lat::Lattice;
                    beam::Union{Beam,Nothing} = nothing,
                    lattice_name::String = string(lat.name),
                    branch_name::String = "main",
                    sequence_name::Union{String,Nothing} = nothing,
                    periodic::Bool = false)
    sequence_name !== nothing && (branch_name = sequence_name)
    used_names = Dict{String,Int}()
    element_names = String[]
    facility = Any[]

    reference_beam = beam === nothing ? Beam(1.0e9) : beam
    push!(facility, Dict(
        "beginning" => Dict(
            "kind" => "BeginningEle",
            "ReferenceP" => _pals_reference_definition(reference_beam),
        ),
    ))
    push!(element_names, "beginning")

    for elem in lat.elements
        base_name = string(_elem_name(elem))
        occurrence = get(used_names, base_name, 0) + 1
        used_names[base_name] = occurrence
        pals_name = occurrence == 1 ? base_name : "$(base_name)__$(occurrence)"
        push!(facility, Dict(pals_name => _pals_element_definition(elem)))
        push!(element_names, pals_name)
    end

    push!(facility, Dict(
        branch_name => Dict(
            "kind" => "BeamLine",
            "periodic" => periodic,
            "line" => element_names,
        ),
    ))
    push!(facility, Dict(
        lattice_name => Dict(
            "kind" => "Lattice",
            "branches" => Any[branch_name],
        ),
    ))
    push!(facility, Dict("use" => lattice_name))
    YAML.write_file(filename, Dict("PALS" => Dict("facility" => facility)),
                    "# PALS lattice file written by TrackPad.jl\n")
    return filename
end

_elem_name(elem) = hasproperty(elem, :name) ? getproperty(elem, :name) : :UNKNOWN

function _pals_reference_definition(beam::Beam)
    total_energy = Float64(beam.energy + beam.mass)
    pc_ref = sqrt(max(total_energy^2 - Float64(beam.mass)^2, 0.0))
    species = if isapprox(abs(beam.mass), M_ELECTRON; rtol=0, atol=1e-6)
        beam.charge < 0 ? "electron" : "positron"
    elseif isapprox(abs(beam.mass), M_PROTON; rtol=0, atol=1e-3)
        beam.charge < 0 ? "anti-proton" : "proton"
    else
        "custom"
    end
    reference = Dict{String,Any}(
        "species_ref" => species,
        "pc_ref" => pc_ref,
        "E_tot_ref" => total_energy,
    )
    if species == "custom"
        reference["mass_ref"] = Float64(beam.mass)
        reference["charge_ref"] = Float64(beam.charge)
    end
    return reference
end

function _pals_element_definition(elem::Drift)
    return Dict{String,Any}("kind" => "Drift", "length" => Float64(elem.L))
end

_pals_element_definition(::Marker) = Dict{String,Any}("kind" => "Marker")

function _pals_multipole_group(normal, skew, order)
    group = Dict{String,Any}("Kn$order" => Float64(normal))
    !iszero(skew) && (group["Ks$order"] = Float64(skew))
    return group
end

function _pals_element_definition(elem::Quadrupole)
    return Dict{String,Any}(
        "kind" => "Quadrupole",
        "length" => Float64(elem.L),
        "MagneticMultipoleP" =>
            _pals_multipole_group(elem.k1, elem.polynom_a[2], 1),
    )
end

function _pals_element_definition(elem::Sextupole)
    return Dict{String,Any}(
        "kind" => "Sextupole",
        "length" => Float64(elem.L),
        "MagneticMultipoleP" =>
            _pals_multipole_group(elem.k2, elem.polynom_a[3], 2),
    )
end

function _pals_element_definition(elem::Octupole)
    return Dict{String,Any}(
        "kind" => "Octupole",
        "length" => Float64(elem.L),
        "MagneticMultipoleP" =>
            _pals_multipole_group(elem.k3, elem.polynom_a[4], 3),
    )
end

function _pals_element_definition(elem::SBend)
    bend = Dict{String,Any}("angle_ref" => Float64(elem.angle))
    !iszero(elem.e1) && (bend["e1"] = Float64(elem.e1))
    !iszero(elem.e2) && (bend["e2"] = Float64(elem.e2))
    !iszero(elem.fint1) && (bend["edge1_int"] = Float64(elem.fint1))
    !iszero(elem.fint2) && (bend["edge2_int"] = Float64(elem.fint2))
    !iszero(elem.gap) && (bend["hgap"] = Float64(elem.gap / 2))
    definition = Dict{String,Any}(
        "kind" => "Bend",
        "length" => Float64(elem.L),
        "BendP" => bend,
    )
    if !iszero(elem.polynom_b[2]) || !iszero(elem.polynom_a[2])
        definition["MagneticMultipoleP"] =
            _pals_multipole_group(elem.polynom_b[2], elem.polynom_a[2], 1)
    end
    return definition
end

function _pals_element_definition(elem::RFCavity)
    phase = elem.freq > 0 ? elem.lag * elem.freq / _C_IO : 0.0
    return Dict{String,Any}(
        "kind" => "RFCavity",
        "length" => Float64(elem.L),
        "RFP" => Dict{String,Any}(
            "voltage" => Float64(elem.volt),
            "frequency" => Float64(elem.freq),
            "phase" => Float64(phase),
            "harmon" => Float64(elem.h),
        ),
    )
end

function _pals_element_definition(elem::Solenoid)
    return Dict{String,Any}(
        "kind" => "Solenoid",
        "length" => Float64(elem.L),
        "SolenoidP" => Dict{String,Any}("Ksol" => Float64(elem.ks)),
    )
end

function _pals_element_definition(elem::Corrector)
    return Dict{String,Any}(
        "kind" => "Kicker",
        "length" => Float64(elem.L),
        "KickerP" => Dict{String,Any}(
            "Kn0" => Float64(elem.xkick),
            "Ks0" => Float64(elem.ykick),
        ),
    )
end

function _pals_element_definition(elem::CrabCavity)
    return Dict{String,Any}(
        "kind" => "CrabCavity",
        "length" => Float64(elem.L),
        "RFP" => Dict{String,Any}(
            "voltage" => Float64(elem.volt),
            "frequency" => Float64(elem.freq),
            "phase" => Float64(elem.phi / (2π)),
        ),
    )
end

function _pals_element_definition(elem)
    throw(ArgumentError(
        "PALS writing is not implemented for TrackPad element $(typeof(elem))",
    ))
end
