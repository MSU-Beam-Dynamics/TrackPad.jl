```@meta
CurrentModule = TrackPad
```

# [Elements](@id elements_guide)

All element types are immutable parametric structs.  Every element `Elem{T,N}`
has a numeric type parameter `T` (usually `Float64`) and a name type `N`
(usually `Symbol`).  Common optional keyword arguments are listed in the
[Common Optional Fields](@ref common_fields) section at the bottom of this page.

## Constructor Quick Reference

Physical strengths are positional arguments. This is the safest reference for
generated code:

| Element | Constructor |
|---------|-------------|
| Drift | `Drift(L; name=:DRIFT)` |
| Marker | `Marker(; name=:MARKER)` |
| Patch | `Patch(; x_offset, y_offset, z_offset, x_pitch, y_pitch, tilt, t_offset, name)` |
| Quadrupole | `Quadrupole(L, k1; name=:QUAD, num_int_steps=10, ...)` |
| Sextupole | `Sextupole(L, k2; name=:SEXT, num_int_steps=10, ...)` |
| Octupole | `Octupole(L, k3; name=:OCT, num_int_steps=10, ...)` |
| Sector bend | `SBend(L, angle, e1=0, e2=0; name=:SBEND, ...)` |
| Rectangular bend | `RBend(L, angle; name=:SBEND, ...)` |
| RF cavity | `RFCavity(L, volt, freq, lag=0; energy, charge, name=:RFCA)` |
| Corrector | `Corrector(L, hkick, vkick; name=:CORRECTOR)` |
| Solenoid | `Solenoid(L, ks; name=:SOLENOID)` |

`L` is in metres. Strength and phase conventions are defined in
[Physics and Data Conventions](@ref conventions).

## Drift Space

```@docs
Drift
```

## Marker

```@docs
Marker
```

## Reference-Frame Patch

```@docs
Patch
```

`Patch` is an active zero-length reference-frame transformation, not a marker.

## Quadrupole

```@docs
Quadrupole
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `L` | magnetic length [m] |
| `k1` | normalised gradient [m⁻²]; positive focusses horizontally |
| `num_int_steps` | number of symplectic integration slices (default: 10) |
| `max_order` | highest multipole order used from `polynom_a/b` |
| `rad` | compatibility flag; radiation support is element-specific |
| `fringe_entrance`, `fringe_exit` | `1` to enable soft-edge fringe fields |
| `polynom_a` / `polynom_b` | lower-level skew / normal polynomial coefficients from dipole through octupole |

## Sextupole

```@docs
Sextupole
```

## Octupole

```@docs
Octupole
```

## Sector Bending Magnet

```@docs
SBend
```

## Rectangular and Combined-Function Bends

```@docs
RBend
ERBend
LBend
```

## RF Cavity

```@docs
RFCavity
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `L` | cavity length [m] |
| `volt` | peak RF voltage [V] |
| `freq` | RF frequency [Hz] |
| `lag` | RF phase lag represented as a longitudinal offset [m] |
| `h` | harmonic number |
| `charge` | reference-particle charge in units of elementary charge |

## Corrector

```@docs
Corrector
HKicker
VKicker
```

## Solenoid

```@docs
Solenoid
```

## Thin Multipole

```@docs
ThinMultipole
```

## Crab Cavity

```@docs
CrabCavity
```

## Accelerating Cavity

```@docs
AccelCavity
```

## Longitudinal RF Map

```@docs
LongitudinalRFMap
```

## Space-Charge Elements

SC-enabled variants mirror their regular counterparts and accept extra aperture
and mesh parameters `a`, `b` [m] and `Nl`, `Nm`, `Nsteps`.

```@docs
DriftSC
QuadrupoleSC
SextupoleSC
OctupoleSC
SBendSC
RBendSC
```

## Geometric / Reference Elements

```@docs
Translation
YRotation
LorentzBoost
InvLorentzBoost
```

## Beam–Beam Kick

```@docs
StrongThinGaussianBeam
StrongGaussianBeam
```

## Wake / Impedance Elements

```@docs
LongitudinalRLCWake
LongitudinalWake
```

## Wiggler

```@docs
Wiggler
```

---

## [Common Optional Fields](@id common_fields)

Many thick magnetic elements accept the following keyword arguments for
misalignment and aperture modelling. Defaults are zero/disabled. Availability
varies by constructor; do not assume every field is accepted by every element.

| Keyword | Type | Description |
|---------|------|-------------|
| `name` | `Symbol` | element name (used by [`findelem`](@ref)) |
| `t1` | `SVector{6}` | entrance translation misalignment vector |
| `t2` | `SVector{6}` | exit translation misalignment vector |
| `r1` | `SMatrix{6,6}` | entrance rotation misalignment matrix |
| `r2` | `SMatrix{6,6}` | exit rotation misalignment matrix |
| `r_apertures` | `SVector{6}` | rectangular aperture half-widths (0 = no cut) |
| `e_apertures` | `SVector{6}` | elliptical aperture half-axes (0 = no cut) |
