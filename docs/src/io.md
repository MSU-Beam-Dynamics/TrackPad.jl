# Lattice File I/O

TrackPad can read and write accelerator lattice files in two formats:

- **PALS** (Particle Accelerator Language Standard) — a YAML-based open standard
- **MAD-X** — the widely-used Methodical Accelerator Design format (`.seq` / `.mad`)

Both readers return a `(Lattice, Beam)` pair ready for tracking.

---

## PALS Format

[PALS](https://pals-project.readthedocs.io/en/latest/) is an open, YAML-based lattice
description standard. Each element is a YAML mapping with a `kind` key and optional
parameter group sub-mappings.

### Reading a PALS file

```julia
using TrackPad

lat, beam = read_pals("my_ring.yaml")
```

Selection and beam keyword arguments:

| Keyword | Default | Meaning |
|---------|---------|---------|
| `lattice` | `nothing` | PALS `Lattice` name; otherwise the last `use` statement or last lattice is selected |
| `branch` | `nothing` | Branch name; otherwise the first branch is selected |
| `sequence` | `nothing` | Compatibility alias accepting either a lattice or standalone `BeamLine` name |
| `beam_energy` | `nothing` | Override reference kinetic energy [eV] |
| `mass` | `nothing` | Override rest mass [eV] |
| `charge` | `nothing` | Override particle charge |
| `strict` | `true` | Reject unsupported elements and direction reversal |

Without explicit beam overrides, `BeginningEle.ReferenceP` supplies the
reference species and either `pc_ref` or `E_tot_ref`. Files without reference
information retain the 1 GeV electron fallback.

### Branch resolution

```julia
resolved = resolve_pals(
    "collider.yaml";
    lattice="eic",
    branch="electron_ring",
)
compiled = compile_branch(resolved)

compiled.lattice
compiled.beam
compiled.source_index[("QF", 2)]
```

`resolve_pals` handles canonical `Lattice.branches`, `use`, nested BeamLines,
inline definitions, inheritance, and repetition. `compile_branch` turns one
resolved path into TrackPad's flat executable lattice and retains the
`(element name, occurrence) => lattice index` map needed by EICViBE diagnostics
and live parameter updates.

EICViBE can bypass YAML and use the same compiler directly:

```julia
compiled = compile_branch(
    [
        Dict("name" => "D1", "kind" => "Drift", "length" => 1.0),
        Dict(
            "name" => "QF",
            "kind" => "Quadrupole",
            "length" => 0.4,
            "MagneticMultipoleP" => Dict("Kn1" => 0.8),
        ),
    ];
    machine_name="eic",
    branch_name="electron_ring",
    periodic=true,
    reference=Dict("species_ref" => "electron", "pc_ref" => 10.0e9),
)
```

This dictionary interface is the intended JuliaCall boundary. EICViBE remains
responsible for machine topology, route selection, multipass traversal, and
control semantics.

!!! warning "Full PALS expansion"
    TrackPad's built-in resolver does not evaluate PALS expressions,
    controllers, includes, forks, floor coordinates, or reference bookkeeping.
    Load PALSJulia and use the optional extension for those files:

    ```julia
    using PALSJulia, TrackPad

    resolved = resolve_pals_full(
        "collider.yaml";
        lattice="eic",
        branch="electron_ring",
    )
    compiled = compile_branch(resolved)
    ```

    `resolve_pals_full` runs PALSJulia/pals-cpp expansion and converts its
    `full_expanded` tree directly, without writing intermediate YAML.

### Writing a PALS file

```julia
write_pals(
    "output.yaml",
    lat;
    beam=beam,
    lattice_name="machine",
    branch_name="ring",
    periodic=true,
)
```

The writer emits `BeginningEle`, canonical `kind: Bend`, one `BeamLine`, one
`Lattice`, and a final `use` statement. Duplicate TrackPad element names receive
occurrence suffixes so different parameter values are not conflated.

### PALS file example

```yaml
PALS:
  facility:
    - D1:
        kind: Drift
        length: 2.07

    - QF:
        kind: Quadrupole
        length: 0.5
        MagneticMultipoleP:
          Kn1: 1.2            # m⁻²  (normalised gradient)

    - QD:
        kind: Quadrupole
        length: 0.5
        MagneticMultipoleP:
          Kn1: -1.2

    - B1:
        kind: Bend
        length: 1.0
        BendP:
          angle_ref: 0.3927   # rad  (π/8)
          e1: 0.1963
          e2: 0.1963
          hgap: 0.02          # m  (half-gap)

    - CAV:
        kind: RFCavity
        length: 0.5
        RFP:
          voltage: 1.0e6      # V
          frequency: 500.0e6  # Hz
          phase: 0.0          # rad/2π (cycles)
          harmon: 1

    - RING:
        kind: BeamLine
        line: [QF, D1, B1, D1, QD, D1, B1, D1, CAV]

    - MACHINE:
        kind: Lattice
        branches: [RING]

    - use: MACHINE
```

### Parameter group reference

| Parameter group | Key | Unit | TrackPad field |
|----------------|-----|------|----------------|
| *(top-level)* | `length` | m | `L` |
| `MagneticMultipoleP` | `Kn1` | m⁻² | `k1` |
| `MagneticMultipoleP` | `Kn2` | m⁻³ | `k2` |
| `MagneticMultipoleP` | `Kn3` | m⁻⁴ | `k3` |
| `MagneticMultipoleP` | `Ks1` | m⁻² | `polynom_a[2]` |
| `BendP` | `angle_ref` | rad | `angle` |
| `BendP` | `e1`, `e2` | rad | `e1`, `e2` |
| `BendP` | `edge1_int` | — | `fint1` |
| `BendP` | `edge2_int` | — | `fint2` |
| `BendP` | `hgap` | m | `gap/2` |
| `RFP` | `voltage` | V | `volt` |
| `RFP` | `frequency` | Hz | `freq` |
| `RFP` | `phase` | rad/2π | `lag` (×C/freq → m) |
| `RFP` | `harmon` | — | `h` |
| `SolenoidP` | `Ksol` | m⁻¹ | `ks` |
| `KickerP` | `Kn0` | rad | `xkick` |
| `KickerP` | `Ks0` | rad | `ykick` |

---

## MAD-X Format

MAD-X input files (`.seq`, `.mad`) define elements and sequences using a
Fortran-like syntax.  TrackPad includes a built-in text parser — no external
MAD-X installation is required.

### Reading a MAD-X file

```julia
lat, beam = read_madx("my_ring.seq")
```

`BEAM, PARTICLE=...` and either `PC=` or `ENERGY=` are used when explicit
`beam_energy`, `mass`, and `charge` overrides are absent. The `sequence`
argument is case-insensitive. `num_int_steps` controls the split-integrator
resolution of imported multipole magnets and defaults to 10; increase it when
comparing against analytic MAD-X linear maps.

TrackPad includes a comparison utility for MAD-X TFS TWISS output:

```bash
julia --project=. scripts/compare_madx_twiss.jl model.madx model.twiss 80
```

The optional final argument is `num_int_steps`.
The report includes periodic Twiss functions, fractional tunes, and
closed-orbit-centered chromaticities. The chromaticity comparison uses a
centered momentum step of `1e-5`.

!!! warning "MAD-X sequence subset"
    The built-in reader is intended for compact `LINE` files and simple
    sequences. For production MAD-X models with positional sequence geometry,
    `REFER`, `FROM`, `CALL`, macros, or deferred expressions, import through
    EICViBE's `cpymad` path and pass the normalized branch to `compile_branch`.

### Supported MAD-X constructs

| Construct | Example |
|-----------|---------|
| Variable assignment | `circum := 120.0;` |
| Element definition | `QF: QUADRUPOLE, L=0.5, K1=1.2;` |
| Sequence definition | `RING: SEQUENCE, REFER=CENTRE, L=circum;` |
| Element placement | `QF, AT=5.0;` |
| `BEAM` command | `BEAM, PARTICLE=ELECTRON, ENERGY=3.0;` |
| Arithmetic | `L_arc := 2*pi*R / 8;` |
| Builtins | `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `abs`, `exp`, `log` |
| Comments | `! this is a comment` and `/* block comment */` |

### Unit conventions (MAD-X → TrackPad)

| MAD-X parameter | MAD-X unit | TrackPad field | TrackPad unit |
|----------------|-----------|----------------|--------------|
| `L` | m | `L` | m |
| `K1`, `K2`, `K3` | m⁻² … | `k1`, `k2`, `k3` | m⁻² … |
| `ANGLE` | rad | `angle` | rad |
| `E1`, `E2` | rad | `e1`, `e2` | rad |
| `FINT` / `FINTX` | — | `fint1` / `fint2` | — |
| `HGAP` | m (half-gap) | `gap` | m (full gap = 2×HGAP) |
| `VOLT` | MV | `volt` | V (×10⁶) |
| `FREQ` | MHz | `freq` | Hz (×10⁶) |
| `LAG` | cycles (0–1) | `lag` | m (×C/freq) |
| `KS` | m⁻¹ | `ks` | m⁻¹ |
| `HKICK`, `VKICK` | rad | `xkick`, `ykick` | rad |

!!! note "FINTX = −1 convention"
    In MAD-X, `FINTX = -1` (the default) means the exit fringe integral equals
    the entrance value `FINT`.  TrackPad respects this: when `FINTX < 0`,
    `fint2 = fint1`.

!!! note "RF phase convention"
    TrackPad stores the RF cavity phase offset as a longitudinal displacement
    `lag` [m] such that the tracking phase is
    `φ = 2π f (z − lag) / c`.
    The conversion from MAD-X `LAG` [cycles] is
    `lag = LAG × c / f`.

### MAD-X file example

```madx
! FODO cell ring
L_cell := 10.0;
L_quad  := 0.5;

QF: QUADRUPOLE, L=L_quad, K1= 1.2;
QD: QUADRUPOLE, L=L_quad, K1=-1.2;
DR: DRIFT, L=(L_cell - 2*L_quad) / 2;
CAV: RFCAVITY, L=0.2, VOLT=1.0, FREQ=500.0, LAG=0.0, HARMON=1;

RING: SEQUENCE, REFER=CENTRE, L=L_cell;
  QF,  AT=L_quad/2;
  DR,  AT=L_quad + L_cell/4;
  QD,  AT=L_cell/2;
  DR,  AT=L_quad + 3*L_cell/4;
  CAV, AT=L_cell - 0.1;
ENDSEQUENCE;
```

---

## API Reference

```@docs
PALSResolvedElement
PALSResolvedBranch
CompiledBranch
resolve_pals
resolve_pals_full
compile_branch
read_pals
read_madx
write_pals
```
