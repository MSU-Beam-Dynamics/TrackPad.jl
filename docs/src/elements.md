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
| Quadrupole | `Quadrupole(L, k1; name=:QUAD, num_int_steps=4, ...)` |
| Sextupole | `Sextupole(L, k2; name=:SEXT, num_int_steps=2, ...)` |
| Octupole | `Octupole(L, k3; name=:OCT, num_int_steps=1, ...)` |
| Sector bend | `SBend(L, angle, e1=0, e2=0; name=:SBEND, ...)` |
| Rectangular bend | `RBend(L, angle; name=:SBEND, ...)` |
| RF cavity | `RFCavity(L, volt, freq, lag=0; energy, charge, name=:RFCA)` |
| Crab cavity | `CrabCavity(L; volt, freq, phi, energy, charge, name=:CRABCAVITY)` |
| Accelerating cavity | `AccelCavity(L; volt, freq, phis, energy, charge, name=:ACCELCAVITY)` |
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
| `num_int_steps` | number of symplectic integration slices (see below) |
| `max_order` | highest multipole order used from `polynom_a/b` |
| `rad` | compatibility flag; radiation support is element-specific |
| `fringe_entrance`, `fringe_exit` | `1` to enable soft-edge fringe fields |
| `polynom_a` / `polynom_b` | lower-level skew / normal polynomial coefficients from dipole through octupole |

### [Integration steps](@id integration_steps)

`num_int_steps` slices a thick magnet for the 4th-order Yoshida integrator.
Cost is linear in it — roughly 42 ns per step per particle for a quadrupole —
and accuracy converges at 4th order, so the defaults are set per element type
rather than uniformly (a sextupole or octupole body is a pure nonlinear kick
with no linear focusing, so it converges at once; the count matters for
elements with a linear part):

| element | default | single-pass error at the default |
|---|---|---|
| `Quadrupole`, `QuadrupoleSC` | 4 | 3.4e-10 (L=0.3 m, k1=0.7 m⁻²) |
| `SBend`, `SBendSC`, `ExactSBend` | 4 | 1.4e-12 (L=1.2 m, 0.05 rad) |
| `Sextupole`, `SextupoleSC` | 2 | 2.1e-15 (L=0.2 m, k2=3 m⁻³) |
| `Octupole`, `OctupoleSC` | 1 | 2e-16 (L=0.2 m, k3=30 m⁻⁴) |
| `ThinMultipole` | 1 | exact (thin kick) |

Errors are against a 400-step reference. Raise it for strong, long magnets: a
quadrupole with L=1.0 m and k1=2.0 m⁻² is at 4.3e-6 with 4 steps and needs 10
or more for 1e-7.

Per-element error is not the whole story, because a ring accumulates it into
the lattice functions. On a 240-element arc (40 quadrupoles at L=0.5 m,
k1=0.9 m⁻²), measured against a 200-step reference:

| `num_int_steps` | Δtune | Δchromaticity | relative cost |
|---|---|---|---|
| 2 | 2.9e-03 | 1.3e-02 | 0.5x |
| 4 (default) | 1.8e-04 | 8.2e-04 | 1.0x |
| 6 | 3.5e-05 | 1.6e-04 | 1.5x |
| 8 | 1.1e-05 | 5.1e-05 | 2.0x |
| 10 | 4.5e-06 | 2.1e-05 | 2.5x |

Convergence is 4th order, so each doubling buys ~16x. The default trades a
~2e-4 tune offset for ~3x throughput; raise it to 8 or 10 for precision optics
matching or resonance work, where that offset is larger than the effect being
studied. Every step count is symplectic, so the residual is a bounded
map distortion rather than a growth. Importers (`read_madx`, `read_pals`) leave
`num_int_steps` at `nothing` by default, which keeps these per-type values; pass
an explicit number to force one count on every thick magnet.

## Sextupole

```@docs
Sextupole
```

## Octupole

```@docs
Octupole
```

## Sector Bending Magnet

`SBend` is the AT-style expanded bend (`BndMPoleSymplectic4Pass`), which is
what `read_madx`/`read_pals` construct and what matches pyAT bit for bit.
`ExactSBend` below is Forest's exact sector bend with the hard-edge dipole
fringe, the model behind PTC, MAD-X `twiss` and Xsuite's `'full'` edges. The
two agree on the linear optics but differ on the chromaticity by a few percent
when the bending radius is a few tens of metres; see
[Bend models and what other codes compute](@ref) before comparing
chromaticities with another code.

```@docs
SBend
```

## Rectangular and Combined-Function Bends

```@docs
RBend
ERBend
ExactSBend
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
| `energy` | reference total energy [eV] (`beam.energy`) |
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
SpaceCharge
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

The strong-beam field and its normalisation are exposed for checking and for
building custom kicks:

```@docs
gaussian_beam_field
faddeeva_w
beambeam_amplitude
classical_radius
CLASSICAL_RADIUS_EV_M
```

## Wake / Impedance Elements

`LongitudinalRLCWake` (RLC resonator model) and `LongitudinalWake` (tabulated
`times`/`wakefields` samples) provide longitudinal wake **Green functions**.
The wake potential is not evaluated per particle: tracking convolves the Green
function with a cloud-in-cell histogram of `r[5]` (`nbins` uniform bins over
the alive-particle range, padded by one bin width on each side), reconstructs
the potential at bin edges, and applies the interpolated potential as an
energy kick at each macroparticle's own coordinate. Use
`physical_wake_scale(beam, bunch_charge, nmacro)` to obtain the physically
normalized `scale` for a Green function in V/C. See the [wake convolution
convention](@ref conventions) for the normative sign, delay, and `scale`
normalization.

For `LongitudinalWake`, `times` must start at zero and increase strictly; with
the default `fliphalf=-1`, they tabulate nonnegative source-to-test delays.
Values outside the tabulated interval are linearly extrapolated.

Because the kick is collective, these elements require multi-particle
tracking: `track!` (serial; `threaded=true` rejects them) or `linepass!`.
Single-particle tracking, TPSA maps, and GPU tracking reject them. Choose `nbins` so that each bin is
populated by many macroparticles; sparsely populated edge bins have their
kick smoothed at the bin scale.

## Translation, Apertures, and Bend Multipoles

`Translation(0; dx=dx, dy=dy, ds=ds)` is a rigid frame displacement: the lateral offsets
redefine the origin and the longitudinal `ds` acts exactly like a `Drift(ds)`
slice, keeping a synchronous particle's ``z`` unchanged. This intentionally
diverges from JuTrack's uncompensated longitudinal term.

Elements carrying `r_apertures` / `e_apertures` enforce them during
multi-particle tracking: macroparticles outside the rectangular or
elliptical boundary are flagged as lost at that element (coordinates are
kept). GPU encoding still rejects nonzero aperture settings.

The bends raise their multipole kick order automatically when higher
multipoles appear in `polynom_b`, and `Quadrupole`, `Sextupole`, `Octupole`,
and the bends apply the Forest (13.29) entrance/exit fringe correction when
their fringe flags are set. `ThinMultipole` fringe flags are accepted but
inert, mirroring JuTrack. Wiggler harmonics use complete six-value blocks;
vertical-field (`Bx`) blocks require nonzero ``k_x`` and ``k_z``, while
horizontal-field (`By`) blocks require nonzero ``k_y`` and ``k_z``. Invalid
blocks are rejected during construction rather than allowed to produce NaNs.

```@docs
LongitudinalRLCWake
LongitudinalWake
wakefieldfunc_RLCWake
wakefieldfunc
physical_wake_scale
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
