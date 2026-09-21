```@meta
CurrentModule = TrackPad
```

# [Lattice File I/O](@id io_guide)

TrackPad's file boundary is deliberately small:

```julia
lat, beam = read_pals("model.yaml")
lat, beam = read_madx("model.madx")
write_pals("model.yaml", lat; beam=beam)
```

Every reader reduces one selected path to an ordered [`Lattice`](@ref) and one
[`Beam`](@ref TrackPad.Beam). TrackPad does not retain machine topology, route graphs,
source-occurrence metadata, controls, or consumer-specific data.

A returned lattice has one of two boundaries:

- `periodic=false`: an open line from start to end;
- `periodic=true`: a ring whose end reference point is its start.

Use [`isperiodic`](@ref) to inspect it. Multi-turn tracking and periodic optics
reject an open line.

## PALS

### Reading

```julia
lat, beam = read_pals(
    "machine.yaml";
    lattice="collider",
    branch="electron_ring",
)
```

Selection is needed only when a file contains more than one possible path. The
selected names are not retained in the tracking model.

| Keyword | Default | Meaning |
|---------|---------|---------|
| `lattice` | `nothing` | Select a PALS `Lattice`; otherwise use the last `use` statement or last lattice |
| `branch` | `nothing` | Select one branch; otherwise use the first branch |
| `sequence` | `nothing` | Compatibility alias for a lattice or standalone `BeamLine` |
| `beam_energy` | `nothing` | Override the reference total energy [eV] |
| `mass` | `nothing` | Override rest-mass energy [eV] |
| `charge` | `nothing` | Override signed charge in elementary-charge units |
| `strict` | `true` | Reject unsupported elements and true direction reversal |

The built-in reader handles selected branches, `use`, nested BeamLines,
repetition, inline definitions, inheritance, and `BeginningEle.ReferenceP`.
`E_tot_ref` is the total energy and `pc_ref` the momentum, both mapping onto `Beam` directly.

Expressions, includes, controllers, forks, and full PALS bookkeeping are not
implemented. A complete external parser or consumer adapter must reduce those
models to the documented subset before TrackPad reads them.

### Supported Elements

The current PALS conversion covers:

- drift, marker, monitor, and instrument;
- patch/reference-frame transform;
- quadrupole, sextupole, octupole, and thin multipole;
- sector and rectangular bends;
- RF and crab cavities;
- solenoid and kickers;
- the thin Gaussian beam-beam kick.

Wigglers are not mapped: PALS carries no parameter group for the period
length, peak field and harmonics a `Wiggler` needs, so a PALS wiggler is an
unsupported element (an error with `strict=true`, a drift with a warning
otherwise). Build the `Wiggler` directly and substitute it.

Normalized strengths `Kn1`, `Kn2`, `Kn3` and skew counterparts are accepted.
Integrated forms such as `Kn2L` are divided by nonzero element length. Named
`k2` and `k3` values are not factorial-scaled on import.

With `strict=false`, an unsupported element is replaced by a drift and a warning
is emitted. This mode is only for inspecting incomplete files, not physics
production.

### Writing

```julia
write_pals("output.yaml", lat; beam=beam)
```

The writer takes periodicity from `lat.periodic` and emits a `BeginningEle`, one
ordered `BeamLine`, one `Lattice`, and `use`. Optional naming keywords are
available for external readability:

```julia
write_pals(
    "output.yaml",
    lat;
    beam=beam,
    lattice_name="machine",
    branch_name="ring",
)
```

Writing supports the documented element subset and rejects unsupported TrackPad
elements. It is not a lossless serializer for every integration, aperture,
alignment, radiation, or advanced-element field.

## MAD-X

### Reading

```julia
lat, beam = read_madx("model.madx"; sequence="ring")
```

The built-in reader is a compact importer for `LINE` models and simple
sequences. It supports element definitions, nested lines, integer repetition,
basic arithmetic, common mathematical functions, `BEAM`, and `USE`.

`USE, PERIOD=...` marks the result periodic. Otherwise the result is an open
line unless explicitly overridden:

```julia
line, beam = read_madx("transport.madx"; periodic=false)
ring, beam = read_madx("ring.madx"; periodic=true)
```

The `periodic` override states a physical boundary; it does not validate floor
or reference-frame closure.

Mapped element types include drifts, markers, monitors, quadrupoles,
sextupoles, octupoles, `SBEND`, `RBEND`, RF and crab cavities, solenoids,
kickers, thin multipoles, and marker-like `DIPEDGE`.

### Unit Conversion

| MAD-X parameter | MAD-X unit | TrackPad field | TrackPad unit |
|----------------|------------|----------------|---------------|
| `L` | m | `L` | m |
| `K1`, `K2`, `K3` | m^-2 and higher | named strength | same normalization |
| `ANGLE`, `E1`, `E2` | rad | bend fields | rad |
| `HGAP` | half-gap [m] | `gap` | full gap [m] |
| `VOLT` | MV | `volt` | V |
| `FREQ` | MHz | `freq` | Hz |
| `LAG` | cycles | `lag` | `LAG*c/freq` [m] |
| `ENERGY` | GeV | `Beam.energy` | total energy [eV] |
| `PC` | GeV | `Beam(pc=...)` | momentum ``P_0c`` [eV] |

The RF conversion includes the signed reference charge. For an electron ring
above transition, MAD-X `LAG=0.5` remains the stable no-acceleration setting.

### MAD-X Limits

TrackPad does not implement full MAD-X placement or language semantics. In
particular, it does not reconstruct drifts from arbitrary `AT` positions or
support complete `REFER`, `FROM`, `CALL`, macro, include, deferred-expression,
table, or reversed-line behavior. Use MAD-X/cpymad or another complete parser
to produce one ordered path for production models that require these features.

For cross-code validation, read the MAD-X model with [`read_madx`](@ref), run
the appropriate TrackPad optics API, and compare against externally generated
MAD-X tables in the consumer or analysis package.

## API

```@docs
read_pals
read_madx
write_pals
```
