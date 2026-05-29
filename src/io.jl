"""
    io.jl

Lattice I/O utilities for TrackPad.jl:

- [`read_pals`](@ref)  — read a PALS-standard YAML lattice file
- [`read_madx`](@ref)  — read a MAD-X sequence / input file
- [`write_pals`](@ref) — write a TrackPad `Lattice` to a PALS YAML file
"""

import YAML

export read_pals, read_madx, write_pals

const _C_IO = 2.99792458e8  # m/s

# ============================================================
# ==================  PALS YAML Reader  ======================
# ============================================================

"""
    read_pals(filename; sequence=nothing, beam_energy=1.0e9,
              mass=M_ELECTRON, charge=-1.0) -> (Lattice, Beam)

Read a PALS-standard YAML lattice file and return a `(Lattice, Beam)` pair.

# Arguments
- `filename`    : path to a `.yaml` / `.yml` file
- `sequence`    : name of the `BeamLine` to use as root sequence;
                  if `nothing`, the first `BeamLine` found is used.
- `beam_energy` : reference kinetic energy [eV]  (default 1 GeV)
- `mass`        : particle rest mass [eV]         (default `M_ELECTRON`)
- `charge`      : particle charge sign            (default `-1.0`)
"""
function read_pals(filename::AbstractString;
                   sequence::Union{String,Symbol,Nothing} = nothing,
                   beam_energy::Real = 1.0e9,
                   mass::Real        = M_ELECTRON,
                   charge::Real      = -1.0)
    data = YAML.load_file(filename; dicttype=Dict{String,Any})
    facility = _pals_facility(data)

    elem_defs = Dict{String,Dict{String,Any}}()
    beamlines = Dict{String,Vector{String}}()
    _pals_collect!(facility, elem_defs, beamlines)

    seq_name   = _choose_sequence(sequence, beamlines)
    elem_names = _pals_expand(seq_name, beamlines, elem_defs)

    elements = AbstractElement[_pals_build_element(n, elem_defs[n]) for n in elem_names]
    lat  = Lattice(elements; name = Symbol(seq_name))
    beam = Beam(Float64(beam_energy); mass=Float64(mass), charge=Float64(charge))
    return lat, beam
end

function _pals_facility(data::Dict)
    if haskey(data, "PALS")
        pals = data["PALS"]
        if isa(pals, Dict) && haskey(pals, "facility")
            return pals["facility"]
        end
        return pals
    end
    return data
end

function _pals_collect!(facility, elem_defs, beamlines)
    if isa(facility, AbstractVector)
        for item in facility
            isa(item, Dict) || continue
            for (name, val) in item
                isa(val, Dict) || continue
                kind = get(val, "kind", "")
                if kind == "BeamLine"
                    beamlines[name] = _pals_line_to_names(get(val, "line", Any[]))
                else
                    elem_defs[name] = val
                end
            end
        end
    elseif isa(facility, Dict)
        for (name, val) in facility
            isa(val, Dict) || continue
            kind = get(val, "kind", "")
            if kind == "BeamLine"
                beamlines[name] = _pals_line_to_names(get(val, "line", Any[]))
            else
                elem_defs[name] = val
            end
        end
    end
end

function _pals_line_to_names(line)
    names = String[]
    for item in line
        if isa(item, String)
            push!(names, item)
        elseif isa(item, Dict)
            for (k, _) in item; push!(names, string(k)); end
        end
    end
    return names
end

function _choose_sequence(sequence, seqmap)
    isempty(seqmap) && error("No BeamLine / sequence found in file")
    if sequence === nothing
        return first(keys(seqmap))
    end
    sname = string(sequence)
    haskey(seqmap, sname) || error("Sequence '$sname' not found in file")
    return sname
end

function _pals_expand(seq_name, beamlines, elem_defs; _depth=0)
    _depth > 50 && error("BeamLine expansion depth exceeded (circular reference?)")
    names = String[]
    for n in beamlines[seq_name]
        if haskey(beamlines, n)
            append!(names, _pals_expand(n, beamlines, elem_defs; _depth=_depth+1))
        elseif haskey(elem_defs, n)
            push!(names, n)
        else
            @warn "PALS: unknown element '$n' in '$seq_name' — skipped"
        end
    end
    return names
end

function _pals_build_element(name::String, d::Dict)
    kind = get(d, "kind", "Drift")
    len  = Float64(get(d, "length", 0.0))
    mmp  = get(d, "MagneticMultipoleP", Dict{String,Any}())
    bp   = get(d, "BendP",             Dict{String,Any}())
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
        k1  = Float64(get(mmp, "Kn1", 0.0))
        k1s = Float64(get(mmp, "Ks1", 0.0))
        pa  = k1s != 0 ? [k1s, 0.0, 0.0, 0.0] : nothing
        return Quadrupole(len, k1; name=Symbol(name), polynom_a=pa)
    elseif k == "SEXTUPOLE"
        k2  = Float64(get(mmp, "Kn2", 0.0))
        k2s = Float64(get(mmp, "Ks2", 0.0))
        pa  = k2s != 0 ? [0.0, k2s, 0.0, 0.0] : nothing
        return Sextupole(len, k2; name=Symbol(name), polynom_a=pa)
    elseif k == "OCTUPOLE"
        k3 = Float64(get(mmp, "Kn3", 0.0))
        return Octupole(len, k3; name=Symbol(name))
    elseif k == "SBEND"
        angle = Float64(get(bp, "angle_ref", 0.0))
        e1    = Float64(get(bp, "e1",        0.0))
        e2    = Float64(get(bp, "e2",        0.0))
        fint1 = Float64(get(bp, "edge1_int", 0.0))
        fint2 = Float64(get(bp, "edge2_int", fint1))
        hgap  = Float64(get(bp, "hgap",      0.0))
        k1    = Float64(get(mmp, "Kn1",      0.0))
        pb    = k1 != 0 ? [0.0, k1, 0.0, 0.0] : nothing
        return SBend(len, angle, e1, e2; name=Symbol(name),
                     fint1=fint1, fint2=fint2, gap=2*hgap, polynom_b=pb)
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
        return RFCavity(len, volt, freq, lag; name=Symbol(name), h=h)
    elseif k == "CRABCAVITY"
        volt  = Float64(get(rfp, "voltage",   0.0))
        freq  = Float64(get(rfp, "frequency", 0.0))
        phase = Float64(get(rfp, "phase",     0.0))
        return CrabCavity(len; name=Symbol(name), volt=volt, freq=freq, phi=phase*2π)
    elseif k == "SOLENOID"
        ks = Float64(get(solp, "Ksol", 0.0))
        return Solenoid(len, ks; name=Symbol(name))
    elseif k == "KICKER"
        hkick = Float64(get(kp, "hkick", 0.0))
        vkick = Float64(get(kp, "vkick", 0.0))
        return Corrector(len, hkick, vkick; name=Symbol(name))
    elseif k == "HKICKER"
        xkick = Float64(get(kp, "hkick", 0.0))
        return HKicker(; name=Symbol(name), L=len, xkick=xkick)
    elseif k == "VKICKER"
        ykick = Float64(get(kp, "vkick", 0.0))
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
        @warn "PALS: unsupported element kind '$kind' for '$name' — using Drift"
        return Drift(len; name=Symbol(name))
    end
end


# ============================================================
# ==================  MAD-X Reader  ==========================
# ============================================================

"""
    read_madx(filename; sequence=nothing, beam_energy=1.0e9,
              mass=M_ELECTRON, charge=-1.0) -> (Lattice, Beam)

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
                   beam_energy::Real = 1.0e9,
                   mass::Real        = M_ELECTRON,
                   charge::Real      = -1.0)
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

    _madx_parse!(tokens, vars, type_defs, line_defs, sequences, use_period)

    # Choose root sequence / line
    seq_name = if sequence !== nothing
        string(sequence)
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

    # Build TrackPad elements (skip unknowns with a warning)
    elements = AbstractElement[]
    unknown_types = Set{String}()
    for n in elem_names
        if !haskey(type_defs, n)
            n in unknown_types || @warn "MAD-X: element '$n' has no definition — skipped"
            push!(unknown_types, n)
            continue
        end
        push!(elements, _madx_build_element(n, type_defs[n], type_defs))
    end

    lat  = Lattice(elements; name = Symbol(seq_name))
    beam = Beam(Float64(beam_energy); mass=Float64(mass), charge=Float64(charge))
    return lat, beam
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
                (isletter(ch) || isdigit(ch) || ch == '_') || break
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
        if i+1 <= n && tokens[i+1] in ("=", ":=")
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

function _madx_parse!(tokens, vars, type_defs, line_defs, sequences, use_period)
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
            _madx_parse_def!(s, name, vars, type_defs, line_defs, sequences)
            continue
        end
        # BEAM command
        if tok == "beam"
            _next!(s); _madx_parse_beam!(s, vars); continue
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

function _madx_parse_def!(s, name, vars, type_defs, line_defs, sequences)
    kw = uppercase(_next!(s))

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

function _madx_parse_beam!(s, vars)
    params = _madx_parse_params!(s, vars)
    haskey(params, "energy") && (vars["_beam_energy_gev"] = Float64(params["energy"]))
    haskey(params, "pc")     && (vars["_beam_pc_gev"]     = Float64(params["pc"]))
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
function _madx_build_element(name::String, d::Dict, type_defs::Dict)
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
        return Quadrupole(L, k1; name=sname, polynom_a=pa)
    elseif mtype == "SEXTUPOLE"
        pa = k2s != 0 ? [0.0, k2s, 0.0, 0.0] : nothing
        return Sextupole(L, k2; name=sname, polynom_a=pa)
    elseif mtype == "OCTUPOLE"
        pa = k3s != 0 ? [0.0, 0.0, k3s, 0.0] : nothing
        return Octupole(L, k3; name=sname, polynom_a=pa)
    elseif mtype == "SBEND"
        pb = k1 != 0 ? [0.0, k1, 0.0, 0.0] : nothing
        pa = k1s != 0 ? [k1s, 0.0, 0.0, 0.0] : nothing
        return SBend(L, angle, e1, e2; name=sname,
                     fint1=fint, fint2=fint2, gap=gap,
                     polynom_a=pa, polynom_b=pb)
    elseif mtype == "RBEND"
        return RBend(L, angle; name=sname, fint1=fint, fint2=fint2, gap=gap)
    elseif mtype == "RFCAVITY"
        return RFCavity(L, volt, freq, lag_m; name=sname, h=h_)
    elseif mtype == "CRABCAVITY"
        phi = lag_c * 2π
        return CrabCavity(L; name=sname, volt=volt, freq=freq, phi=phi)
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
        return ThinMultipole(L, pa, pb; name=sname)
    elseif mtype == "DIPEDGE"
        return Marker(; name=sname)
    else
        @warn "MAD-X: unknown type '$mtype' for '$name' — using Drift/Marker"
        return L > 0 ? Drift(L; name=sname) : Marker(; name=sname)
    end
end


# ============================================================
# ==================  PALS YAML Writer  ======================
# ============================================================

"""
    write_pals(filename, lat; beam=nothing, sequence_name="RING")

Write a TrackPad `Lattice` to a PALS-standard YAML file.

Each unique element (by name) is written once under `PALS.facility`, followed by
a single `BeamLine` entry listing the full sequence.
"""
function write_pals(filename::AbstractString, lat::Lattice;
                    beam::Union{Beam,Nothing} = nothing,
                    sequence_name::String = "RING")
    buf = IOBuffer()
    println(buf, "# PALS lattice file — written by TrackPad.jl")
    println(buf, "PALS:")
    println(buf, "  facility:")

    seen = Set{Symbol}()
    for elem in lat.elements
        n = _elem_name(elem)
        n in seen && continue
        push!(seen, n)
        _write_pals_element(buf, elem)
    end

    println(buf, "")
    println(buf, "  - $(sequence_name):")
    println(buf, "      kind: BeamLine")
    names = [string(_elem_name(e)) for e in lat.elements]
    println(buf, "      line: [$(join(names, ", "))]")
    write(filename, String(take!(buf)))
end

_elem_name(elem) = hasproperty(elem, :name) ? getproperty(elem, :name) : :UNKNOWN

function _write_pals_element(buf, elem::Drift)
    println(buf, "  - $(_elem_name(elem)):")
    println(buf, "      kind: Drift")
    println(buf, "      length: $(elem.L)")
end

function _write_pals_element(buf, elem::Marker)
    println(buf, "  - $(_elem_name(elem)):")
    println(buf, "      kind: Marker")
end

function _write_pals_element(buf, elem::Quadrupole)
    n = _elem_name(elem)
    println(buf, "  - $n:")
    println(buf, "      kind: Quadrupole")
    println(buf, "      length: $(elem.L)")
    println(buf, "      MagneticMultipoleP:")
    println(buf, "        Kn1: $(elem.k1)")
    abs(elem.polynom_a[2]) > 0 && println(buf, "        Ks1: $(elem.polynom_a[2])")
end

function _write_pals_element(buf, elem::Sextupole)
    n = _elem_name(elem)
    println(buf, "  - $n:")
    println(buf, "      kind: Sextupole")
    println(buf, "      length: $(elem.L)")
    println(buf, "      MagneticMultipoleP:")
    println(buf, "        Kn2: $(elem.k2)")
end

function _write_pals_element(buf, elem::Octupole)
    n = _elem_name(elem)
    println(buf, "  - $n:")
    println(buf, "      kind: Octupole")
    println(buf, "      length: $(elem.L)")
    println(buf, "      MagneticMultipoleP:")
    println(buf, "        Kn3: $(elem.k3)")
end

function _write_pals_element(buf, elem::SBend)
    n = _elem_name(elem)
    println(buf, "  - $n:")
    println(buf, "      kind: SBend")
    println(buf, "      length: $(elem.L)")
    println(buf, "      BendP:")
    println(buf, "        angle_ref: $(elem.angle)")
    elem.e1    != 0 && println(buf, "        e1: $(elem.e1)")
    elem.e2    != 0 && println(buf, "        e2: $(elem.e2)")
    elem.fint1 != 0 && println(buf, "        edge1_int: $(elem.fint1)")
    elem.fint2 != 0 && println(buf, "        edge2_int: $(elem.fint2)")
    elem.gap   != 0 && println(buf, "        hgap: $(elem.gap / 2)")
end

function _write_pals_element(buf, elem::RFCavity)
    n     = _elem_name(elem)
    freq  = elem.freq
    phase = freq > 0 ? elem.lag * freq / _C_IO : 0.0
    println(buf, "  - $n:")
    println(buf, "      kind: RFCavity")
    println(buf, "      length: $(elem.L)")
    println(buf, "      RFP:")
    println(buf, "        voltage: $(elem.volt)")
    println(buf, "        frequency: $(freq)")
    println(buf, "        phase: $(phase)")
    println(buf, "        harmon: $(elem.h)")
end

function _write_pals_element(buf, elem::Solenoid)
    n = _elem_name(elem)
    println(buf, "  - $n:")
    println(buf, "      kind: Solenoid")
    println(buf, "      length: $(elem.L)")
    println(buf, "      SolenoidP:")
    println(buf, "        Ksol: $(elem.ks)")
end

function _write_pals_element(buf, elem::Corrector)
    n = _elem_name(elem)
    println(buf, "  - $n:")
    println(buf, "      kind: Kicker")
    println(buf, "      length: $(elem.L)")
    println(buf, "      KickerP:")
    println(buf, "        hkick: $(elem.xkick)")
    println(buf, "        vkick: $(elem.ykick)")
end

function _write_pals_element(buf, elem::CrabCavity)
    n     = _elem_name(elem)
    freq  = elem.freq
    phase = freq > 0 ? elem.phi / (2π) : 0.0
    println(buf, "  - $n:")
    println(buf, "      kind: CrabCavity")
    println(buf, "      length: $(elem.L)")
    println(buf, "      RFP:")
    println(buf, "        voltage: $(elem.volt)")
    println(buf, "        frequency: $(freq)")
    println(buf, "        phase: $(phase)")
end

# Fallback for any other element type
function _write_pals_element(buf, elem)
    n    = _elem_name(elem)
    L    = hasproperty(elem, :L) ? getproperty(elem, :L) : 0.0
    kind = string(nameof(typeof(elem)))
    println(buf, "  - $n:")
    println(buf, "      kind: $kind")
    L != 0 && println(buf, "      length: $L")
end
