```@meta
CurrentModule = TrackPad
```

# [Physics and Data Conventions](@id conventions)

This page is normative for TrackPad's public interfaces. Use it when comparing
TrackPad with MAD-X, JuTrack, Bmad, Xsuite, or another tracking code. A numerical
disagreement should first be checked against these conventions before changing
an integrator or finite-difference step.

## Phase-Space Coordinates

TrackPad uses

```math
\mathbf{r} = (x, p_x, y, p_y, z, \delta).
```

| Index | Name | Unit | Definition |
|-------|------|------|------------|
| 1 | `x` | m | Horizontal displacement |
| 2 | `px` | 1 | Canonical horizontal momentum normalized by reference momentum |
| 3 | `y` | m | Vertical displacement |
| 4 | `py` | 1 | Canonical vertical momentum normalized by reference momentum |
| 5 | `z` | m | Longitudinal path/time coordinate, approximately ``-c\Delta t`` |
| 6 | `delta` | 1 | Relative momentum deviation ``(p-p_0)/p_0`` |

Scalar coordinates are normally `SVector{6,T}`. Particle batches are `N x 6`
matrices: one particle per row and the same six coordinates in columns.

The exact drift Hamiltonian uses

```math
p_z^2 = 1 + 2\delta/\beta + \delta^2 - p_x^2 - p_y^2.
```

`USE_EXACT_HAMILTONIAN` is enabled by default. Changing it changes map and
chromaticity comparisons.

## Reference Beam

`Beam(energy; mass, charge)` uses:

| Field | Unit | Meaning |
|-------|------|---------|
| `energy` | eV | Reference kinetic energy |
| `mass` | eV | Rest-mass energy ``mc^2`` |
| `charge` | elementary charge | Signed reference-particle charge |
| `gamma` | 1 | ``(energy + mass)/mass`` |
| `beta` | 1 | Reference speed divided by ``c`` |

`Beam(3e9)` is therefore a 3 GeV kinetic-energy electron beam, not a 3 GeV
total-energy beam. PALS `pc_ref` and `E_tot_ref`, and MAD-X `PC` and `ENERGY`,
are converted to kinetic energy by the readers.

## Magnet Strengths

Named multipole constructors take standard normalized strengths:

| Constructor field | Unit |
|-------------------|------|
| `Quadrupole.k1` | m^-2 |
| `Sextupole.k2` | m^-3 |
| `Octupole.k3` | m^-4 |

Pass `k2` and `k3` without factorial scaling. Tracking inserts `k2/2!` and
`k3/3!` into the polynomial kick internally. PALS `Kn1`, `Kn2`, and `Kn3` map
directly to these named strengths. Integrated PALS values `Kn1L`, `Kn2L`, and
`Kn3L` are divided by element length during compilation.

The raw `polynom_a` and `polynom_b` arrays are the skew and normal polynomial
coefficients consumed by Horner evaluation, ordered from dipole through
octupole. They are lower-level fields and are not interchangeable with named
`k2`/`k3` unless the appropriate factorial normalization is applied.

## Bend Geometry

For `SBend(L, angle, e1, e2)`:

- `L` is the reference arc length in metres.
- `angle` is the signed total bend angle in radians.
- Body curvature is `irho = angle/L` when `L != 0`.
- `e1` and `e2` are entrance and exit pole-face angles in radians.
- `gap` is the full magnet gap in metres.

`RBend(L, angle)` is a convenience constructor using the same body map with
`e1 = e2 = angle/2`. Importers convert chord/rectangular geometry to TrackPad's
arc-length representation before construction.

## RF Phase and Charge

For `RFCavity`, the tracking phase is

```math
\phi = 2\pi f (z - lag)/c - philag.
```

| Field | Unit | Meaning |
|-------|------|---------|
| `volt` | V | Peak cavity voltage; numerically eV per unit elementary charge |
| `freq` | Hz | RF frequency |
| `lag` | m | Longitudinal phase offset |
| `philag` | rad | Additional phase subtracted from the RF phase |
| `energy` | eV | Reference kinetic energy used to normalize the kick |
| `charge` | elementary charge | Signed reference-particle charge |

A directly constructed `RFCavity` defaults to `energy=0`, which intentionally
disables its kick. Set both `energy=beam.energy` and `charge=beam.charge` for a
physical direct-construction model. PALS and MAD-X readers set them from the
reference beam.

PALS/MAD-X phase values expressed in cycles are converted with
`lag = phase * c/freq`. For an electron ring above transition, the stable
no-acceleration MAD-X setting remains `LAG=0.5` because the signed charge is
included in the kick.

## Loss Convention

For real-valued scalar tracking, `check_lost(r)` returns true when:

- `x` or `y` is NaN;
- `abs(x)` or `abs(y)` exceeds 1 m; or
- `abs(px)` or `abs(py)` exceeds 1.

Some maps also produce NaN coordinates when longitudinal momentum becomes
nonphysical. Matrix tracking uses integer flags (`0` alive, `1` lost). Packed
GPU tracking writes nonfinite coordinates for lost particles. These are current
global safety limits, not element apertures.

## Optics and Finite Differences

- `one_turn_map`/`findm66` compute numerical Jacobians.
- `gettune` extracts uncoupled tunes from transverse 2 x 2 blocks.
- `twissline` computes uncoupled periodic Twiss functions at element boundaries.
- `getchrom` defaults to the JuTrack-compatible forward momentum difference;
  use `centered=true` for a centered derivative.
- Coupled optics are not yet represented by `TwissLineResult`.

Finite-difference defaults are part of compatibility behavior. Specify `h`,
`dpp`, reference orbit, and centered/closed-orbit choices when reporting a
cross-code comparison.

## Time Dependence

`TimeContext(real_time; turn)` carries both physical time and turn index.
`linepass` and `ringpass` resolve `TimeVaryingElement` fields at the supplied
context. Materialize a time-dependent lattice before GPU adaptation because
device kernels cannot carry host closures.

## CPU and GPU Contract

Scalar CPU tracking defines behavior. `GPULattice` supports only the element
types and settings listed in the [GPU guide](@ref gpu_guide). Unsupported
physics throws `ArgumentError` during packing. Metal is `Float32` only; CUDA
supports `Float32` and `Float64`. CPU/GPU comparison tolerances must match the
selected precision.

## Lattice Boundary

`Lattice(elements)` is an open line. `Lattice(elements; periodic=true)` is a
ring whose end reference point is the same as its start. `linepass` and
`transfer_map` work for either boundary. Multi-turn tracking, one-turn maps,
tune, chromaticity, periodic Twiss, and closed-orbit searches require a
periodic lattice.

## File Import Boundary

PALS and MAD-X readers reduce a selected path directly to `(Lattice, Beam)`.
Machine topology, route graphs, source-occurrence identity, controls, sessions,
and result schemas remain outside TrackPad. See
[Lattice File I/O](@ref io_guide).
