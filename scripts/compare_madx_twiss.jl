"""
Compare TrackPad periodic optics with a MAD-X TFS TWISS table.

Usage:

    julia --project=. scripts/compare_madx_twiss.jl lattice.madx lattice.twiss [steps]
"""

using TrackPad

length(ARGS) in (2, 3) ||
    error("usage: compare_madx_twiss.jl lattice.madx lattice.twiss [steps]")

const MADX_PATH = ARGS[1]
const TWISS_PATH = ARGS[2]
const NUM_INT_STEPS = length(ARGS) == 3 ? parse(Int, ARGS[3]) : 40

function parse_tfs(path)
    headers = Dict{String,Any}()
    columns = String[]
    rows = NamedTuple[]

    for line in eachline(path)
        stripped = strip(line)
        isempty(stripped) && continue
        if startswith(stripped, "@")
            fields = [match.match for match in eachmatch(r"\"[^\"]*\"|\S+", stripped)]
            key = uppercase(fields[2])
            value = strip(join(fields[4:end], " "), '"')
            headers[key] = something(tryparse(Float64, value), value)
        elseif startswith(stripped, "*")
            columns = uppercase.(split(stripped)[2:end])
        elseif startswith(stripped, "\"")
            fields = [match.match for match in eachmatch(r"\"[^\"]*\"|\S+", stripped)]
            values = Any[
                startswith(field, '"') ? strip(field, '"') : parse(Float64, field)
                for field in fields
            ]
            data = Dict(zip(columns, values))
            push!(rows, (
                name=String(data["NAME"]),
                keyword=String(data["KEYWORD"]),
                s=Float64(data["S"]),
                L=Float64(data["L"]),
                betax=Float64(data["BETX"]),
                alphax=Float64(data["ALFX"]),
                betay=Float64(data["BETY"]),
                alphay=Float64(data["ALFY"]),
            ))
        end
    end
    return headers, rows
end

headers, rows = parse_tfs(TWISS_PATH)
sequence = string(get(headers, "SEQUENCE", ""))
lat, beam = read_madx(
    MADX_PATH;
    sequence=isempty(sequence) ? nothing : sequence,
    num_int_steps=NUM_INT_STEPS,
)
twiss = periodic_twiss(lat, beam; h=1.0e-7)
chrom = getchrom(
    lat, beam;
    h=1.0e-7, dpp=1.0e-5, centered=true, closed_orbit=true,
)

length(rows) == length(lat) + 2 || error(
    "TWISS table has $(length(rows)) rows for $(length(lat)) elements; " *
    "expected start and end rows",
)
reference = rows[1:end-1]

metrics = (
    elements=length(lat),
    circumference=sum(get_length(element) for element in lat.elements),
    reference_circumference=Float64(headers["LENGTH"]),
    tunex=twiss.tunex,
    reference_tunex=mod(Float64(headers["Q1"]), 1.0),
    tuney=twiss.tuney,
    reference_tuney=mod(Float64(headers["Q2"]), 1.0),
    chromx=chrom[1],
    reference_chromx=Float64(headers["DQ1"]),
    chromx_difference=chrom[1] - Float64(headers["DQ1"]),
    chromy=chrom[2],
    reference_chromy=Float64(headers["DQ2"]),
    chromy_difference=chrom[2] - Float64(headers["DQ2"]),
    max_abs_s=maximum(abs(twiss.s[i] - reference[i].s)
                      for i in eachindex(twiss.s)),
    max_abs_betax=maximum(abs(twiss.betax[i] - reference[i].betax)
                          for i in eachindex(twiss.s)),
    max_abs_alphax=maximum(abs(twiss.alphax[i] - reference[i].alphax)
                           for i in eachindex(twiss.s)),
    max_abs_betay=maximum(abs(twiss.betay[i] - reference[i].betay)
                          for i in eachindex(twiss.s)),
    max_abs_alphay=maximum(abs(twiss.alphay[i] - reference[i].alphay)
                           for i in eachindex(twiss.s)),
)

println("MAD-X/TrackPad TWISS comparison (num_int_steps=$NUM_INT_STEPS)")
for (key, value) in pairs(metrics)
    println(rpad(string(key), 28), value)
end
