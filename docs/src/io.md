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

Optional keyword arguments:

| Keyword | Default | Meaning |
|---------|---------|---------|
| `sequence` | `nothing` | Name of the `BeamLine` to use as root; first one found if `nothing` |
| `beam_energy` | `1.0e9` | Reference kinetic energy [eV] |
| `mass` | `M_ELECTRON` | Particle rest mass [eV] |
| `charge` | `-1.0` | Particle charge sign |

### Writing a PALS file

```julia
write_pals("output.yaml", lat; sequence_name="RING")
```

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
        kind: SBend
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
| `KickerP` | `hkick` | rad | `xkick` |
| `KickerP` | `vkick` | rad | `ykick` |

---

## MAD-X Format

MAD-X input files (`.seq`, `.mad`) define elements and sequences using a
Fortran-like syntax.  TrackPad includes a built-in text parser — no external
MAD-X installation is required.

### Reading a MAD-X file

```julia
lat, beam = read_madx("my_ring.seq")
```

Optional keyword arguments match those of `read_pals`.

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
read_pals
read_madx
write_pals
```
