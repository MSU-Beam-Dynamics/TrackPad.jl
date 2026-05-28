```@meta
CurrentModule = TrackPad
```

# Elements

All element types are immutable parametric structs.  Every element `Elem{T,N}`
has a numeric type parameter `T` (usually `Float64`) and a name type `N`
(usually `Symbol`).  Common optional keyword arguments are listed in the
[Common Optional Fields](@ref common_fields) section at the bottom of this page.

## Drift Space

```@docs
Drift
```

## Marker

```@docs
Marker
```

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
| `rad` | `1` to enable synchrotron-radiation kicks |
| `fringe_entrance`, `fringe_exit` | `1` to enable soft-edge fringe fields |
| `polynom_a` / `polynom_b` | skew / normal multipole coefficients `[B₁, B₂, B₃, B₄]` |

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
| `volt` | peak RF voltage [eV] |
| `freq` | RF frequency [Hz] |
| `lag` | RF phase lag [rad] (0 = on-crest for electrons) |
| `h` | harmonic number |

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

```@raw html
<a id="common_fields"></a>
```

## Common Optional Fields

All thick magnetic elements accept the following keyword arguments for
misalignment and aperture modelling.  Defaults are zero / disabled.

| Keyword | Type | Description |
|---------|------|-------------|
| `name` | `Symbol` | element name (used by [`findelem`](@ref)) |
| `t1` | `SVector{6}` | entrance translation misalignment vector |
| `t2` | `SVector{6}` | exit translation misalignment vector |
| `r1` | `SMatrix{6,6}` | entrance rotation misalignment matrix |
| `r2` | `SMatrix{6,6}` | exit rotation misalignment matrix |
| `r_apertures` | `SVector{6}` | rectangular aperture half-widths (0 = no cut) |
| `e_apertures` | `SVector{6}` | elliptical aperture half-axes (0 = no cut) |
