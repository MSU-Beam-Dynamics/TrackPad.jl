```@meta
CurrentModule = TrackPad
```

# [Physics and Data Conventions](@id conventions)

This page is normative for TrackPad's public interfaces. Use it when comparing
TrackPad with MAD-X, JuTrack, Bmad, Xsuite, or another tracking code. A numerical
disagreement should first be checked against these conventions before changing
an integrator or finite-difference step.

## Phase-Space Coordinates

TrackPad stores the phase-space vector

```math
\mathbf{r} = (x, p_x, y, p_y, z, \delta_E),
```

in the following order:

| Index | Julia name | Unit | Definition |
|:-----:|------------|------|------------|
| 1 | `x` | ``\mathrm{m}`` | Horizontal displacement ``x`` |
| 2 | `px` | ``1`` | Canonical horizontal momentum ``p_x=P_x/P_0`` |
| 3 | `y` | ``\mathrm{m}`` | Vertical displacement ``y`` |
| 4 | `py` | ``1`` | Canonical vertical momentum ``p_y=P_y/P_0`` |
| 5 | `z` | ``\mathrm{m}`` | Canonical coordinate ``z=s/\beta_0-ct=-c(t-t_0)`` |
| 6 | `delta` | ``1`` | Relative energy variable ``\delta_E=(E-E_0)/(P_0c)`` |

Here ``P_0``, ``E_0``, and ``t_0`` are the reference momentum, total energy,
and arrival time at the same longitudinal reference position. The transverse
momenta are canonical momenta; they equal mechanical momenta in a field-free
region. Scalar coordinates are normally `SVector{6,T}`. Particle batches are
``N\times6`` matrices with one particle per row.

### Longitudinal canonical pair

With ``\Delta t=t-t_0``, the energy-based canonical pair is

```math
(\sigma_E,\delta_E)
=\left(-c\Delta t,\frac{\Delta E}{P_0c}\right).
```

At a fixed reference position ``s``, the reference particle arrives at
``ct_0=s/\beta_0``. TrackPad stores ``z=\sigma_E`` directly: a particle
arriving early has ``z>0``, while a late particle has ``z<0``. The symplectic
two-form in TrackPad's coordinate order is

```math
\omega
=\mathrm{d}x\wedge\mathrm{d}p_x
+\mathrm{d}y\wedge\mathrm{d}p_y
+\mathrm{d}z\wedge\mathrm{d}\delta_E.
```

This sign is observable: a particle that takes longer than the reference
particle to cross a drift acquires negative `z`. TrackPad's coordinate agrees
directly with the MAD-X canonical coordinate:

```math
T_{\mathrm{MAD\text{-}X}}=z_{\mathrm{TrackPad}}=-c\Delta t.
```

An equally valid momentum-based canonical pair is

```math
(\sigma_P,\delta_P)
=\left(-\beta_0c\Delta t,\frac{P-P_0}{P_0}\right).
```

TrackPad does **not** store ``\delta_P``. The exact conversion between its
sixth coordinate and relative momentum is

```math
1+\delta_P
=\sqrt{1+\frac{2\delta_E}{\beta_0}+\delta_E^2},
```

and only to first order is ``\delta_E=\beta_0\delta_P+O(\delta_P^2)``.
Accordingly, ``\sigma_P=\beta_0\sigma_E`` at linear order. The common
replacement ``\delta_E\simeq\delta_P`` is an ultrarelativistic approximation,
not TrackPad's definition.

### Exact drift

For a straight element, define the normalized longitudinal mechanical momentum

```math
\pi_s=\frac{P_s}{P_0}
=\sqrt{1+\frac{2\delta_E}{\beta_0}+\delta_E^2-p_x^2-p_y^2}.
```

The exact drift of length ``L`` is

```math
\begin{aligned}
x_f &= x_i + L\frac{p_x}{\pi_s},\\
y_f &= y_i + L\frac{p_y}{\pi_s},\\
z_f &= z_i - L\left[
\frac{\beta_0^{-1}+\delta_E}{\pi_s}-\beta_0^{-1}
\right],\\
\delta_{E,f} &= \delta_{E,i}.
\end{aligned}
```

This equation fixes both the meaning of `delta` and the sign of `z`.
`USE_EXACT_HAMILTONIAN` is a compile-time constant (`true`); all tracking,
optics, TPSA and AD paths use the exact drift. The JuTrack-compatible
ultrarelativistic approximation is still available for cross-code comparison
through `drift6(r, L, beti, Val(false))`; it changes nonlinear maps and
chromaticity.

### Beam covariance and emittance

Macroparticle matrices follow the same canonical order as scalar tracking. A
``6\times6`` covariance matrix therefore uses

```math
X=(x,p_x,y,p_y,z,\delta_E)^\mathsf{T},
\qquad
\Sigma=\left\langle(X-\langle X\rangle)
(X-\langle X\rangle)^\mathsf{T}\right\rangle.
```

The transverse geometric emittance of an uncoupled plane is represented by

```math
\Sigma_x=\epsilon_x
\begin{pmatrix}
\beta_x & -\alpha_x\\
-\alpha_x & (1+\alpha_x^2)/\beta_x
\end{pmatrix},
```

with the analogous expression for ``y`` and, when requested, the canonical
``(z,\delta_E)`` plane. Because ``p_x`` and ``p_y`` are normalized canonical
momenta, interpreting them as slopes is only a paraxial approximation.

The matched-distribution API defines ordinary and crab dispersion through

```math
\begin{pmatrix}x\\p_x\\y\\p_y\end{pmatrix}
=
\begin{pmatrix}x_\beta\\p_{x,\beta}\\y_\beta\\p_{y,\beta}\end{pmatrix}
+D_c z+D_E\delta_E.
```

Thus ordinary dispersion is differentiated with respect to TrackPad's energy
coordinate. If external optics uses
``\delta_P=(P-P_0)/P_0``, convert its first-order response using
``D_E=D_P/\beta_0``. Signs of crab response follow TrackPad's positive-early
``z`` convention.

Projected emittances are ``\sqrt{\det\Sigma_{ii}}`` for the three diagonal
canonical ``2\times2`` blocks. They are generally not invariants in a coupled,
dispersive, or crabbed beam. `eigenemittances` instead returns the sorted paired
singular values of ``\sqrt{\Sigma}J\sqrt{\Sigma}``.

### Finite-``\beta_0`` coverage

Maps originally expressed in the momentum variable use the exact conversion

```math
p=\frac{P}{P_0}
=\sqrt{1+\frac{2\delta_E}{\beta_0}+\delta_E^2},
\qquad
\frac{\mathrm d\delta_P}{\mathrm d\delta_E}
=\frac{\beta_0^{-1}+\delta_E}{p}.
```

The second factor is required in the longitudinal update. TrackPad applies
this conversion in the standard and exact bend bodies, `Solenoid`,
`Corrector`, `LBend`, the non-radiating `Wiggler`, and the RF-family maps.
The scalar CPU, packed GPU, and PolySeries implementations use the same
conversion where each element is supported.

## Strong beam-beam kick

`StrongThinGaussianBeam` and `StrongGaussianBeam` apply the transverse field of
a bi-Gaussian strong beam through the Bassetti–Erskine formula (closed
round-beam form when `σx = σy`), evaluated with a pure-arithmetic Faddeeva
function so the same code runs in the packed GPU kernel:

```math
Δp_x = A\,E_x,\qquad Δp_y = A\,E_y,\qquad
E_r^{\rm round} = \frac{2}{r}\left(1 - e^{-r^2/2σ^2}\right)
```

so the linear kick is `A x/σ²` and the far field is the `2A/r` Coulomb field of
the whole charge. The coupling `A` (`amplitude` for the thin element,
`kick_scale × zslice_npar[i]` per slice for the sliced one) is

```math
A = N\, r_{0,w}\, q_w q_s / γ_w
```

with `N` strong particles, and the classical radius `r_{0,w}`, charge `q_w` and
Lorentz factor `γ_w` of the *weak* (tracked) beam; `beambeam_amplitude(beam, N,
q_s)` builds it. Positive `A` (like charges) defocuses; the linear beam-beam
parameter is `ξ = β* A / (4π σ²)`. `StrongGaussianBeam` adds Hirata's
synchro-beam mapping: slice `i` meets a weak particle of coordinate `z` at
`s* = (z - zslice_center[i])/2` from the IP (both `z` and `zslice_center`
positive toward the head of their bunch), with drift–kick–drift-back in the
transverse plane, constant beam size along the bunch (no hourglass) and no
energy update. TPSA maps support round strong beams only.

The following specialized paths remain approximations or require a separate
model-specific normalization audit:

- the linearized drift `drift6(r, L, beti, Val(false))`, which deliberately uses
  the ultrarelativistic ``p\simeq1+\delta_E`` Hamiltonian;
- the Brown/SOLEIL/THOMX bend fringe-field models; and
- synchrotron-radiation and other collective kicks (longitudinal wakes follow
  the normative convention below).

Do not infer finite-energy validity solely from historical JuTrack parity.
JuTrack comparisons are exact only in the shared ultrarelativistic limit when
the reference implementation mixes ``\delta_P`` and ``\delta_E`` conventions.

### Longitudinal wake convolution

`LongitudinalRLCWake` and `LongitudinalWake` supply Green functions, not wake
potentials. With the delay argument

```math
t = \frac{z_{\mathrm{test}} - z_{\mathrm{source}}}{c},
```

the Green functions vanish for ``t > 0``: a particle feels only sources ahead
of it. The longitudinal wake potential is the discrete convolution of the
Green function ``W`` with the bunch profile obtained from a cloud-in-cell
histogram of ``r_5 = z`` over all alive macroparticles:

```math
V_k = \sum_j N_j \, W\!\left(\frac{z_k - z_j}{c}\right),
```

where ``N_j`` is the (possibly fractional) cloud-in-cell weight in histogram
bin ``j`` and ``z_k``, ``z_j`` are uniform bin-center coordinates. The grid
uses `nbins` bins per element and is padded by one nominal bin width beyond
the alive-particle range on each side, so every deposition neighbor and the
edge reconstruction stay inside the grid. As in JuTrack, ``V`` is averaged
onto bin edges and every macroparticle receives the edge-interpolated value at
its own coordinate. TrackPad deliberately uses a translation-invariant range
and linear cloud-in-cell weights; JuTrack instead uses a zero-centered range
and quadratic neighbor weights. The resulting kick is

```math
\Delta\delta_i = -\texttt{scale} \cdot V(z_i).
```

The convolution includes each particle's own bin (self term at zero delay).
Each macroparticle carries equal charge; `scale` absorbs the macroparticle
charge, the ``1/(P_0 c)`` kick normalization, and any additional user factor.
`physical_wake_scale(beam, bunch_charge, nmacro)` returns the physically
normalized value for a Green function in V/C:

```math
\texttt{scale}
= \frac{\widehat q\,Q_{\mathrm{macro}}}{P_0c},
\qquad
Q_{\mathrm{macro}}
= \frac{Q_{\mathrm{bunch}}}{N_{\mathrm{macro}}},
```

where ``\widehat q=q/e`` is the signed reference-particle charge stored in
`beam.charge`, ``Q_{\mathrm{bunch}}`` and ``Q_{\mathrm{macro}}`` are signed
charges in C, and ``P_0c=\beta_0E_0`` is in eV. Multiplying a V/C Green
function by ``Q_{\mathrm{macro}}`` gives volts; multiplying by ``\widehat q``
gives the test-particle energy change in eV. Thus like-sign source and test
particles give a positive scale and a positive wake decelerates through the
minus sign in the tracking kick.

Because the kick is collective, single-particle tracking (`linepass`,
`ringpass`, TPSA maps) rejects these elements; use the ``N\times 6``
multi-particle APIs. GPU tracking rejects them during `GPULattice`
construction.

### Compatibility with earlier TrackPad versions

Earlier TrackPad versions inherited JuTrack's noncanonical positive-delay
coordinate ``z_{\mathrm{old}}=c(t-t_0)``. Existing particle arrays and stored
maps must be converted with

```math
\mathbf r_{\mathrm{new}}=C\mathbf r_{\mathrm{old}},
\qquad
C=\operatorname{diag}(1,1,1,1,-1,1),
```

and a first-order map must be converted as
``M_{\mathrm{new}}=CM_{\mathrm{old}}C``. This is an intentional breaking
correction. JuTrack comparisons in TrackPad's tests apply this transformation
explicitly.

At finite ``\beta_0``, this coordinate sign conversion does not reproduce old
tracking results from bends, RF cavities, solenoids, correctors, linear bends,
or wigglers. Those maps now use the exact ``\delta_E``-to-momentum conversion
described above, so there is no coordinate-only conversion for an already
tracked trajectory.

Collective longitudinal-wake kicks changed numerics: earlier versions used
nearest-bin deposition with a piecewise-constant per-bin kick over an
unpadded grid; current versions use cloud-in-cell deposition and a
bin-edge-interpolated kick on a padded grid. The edge interpolation follows
JuTrack, while the translation-invariant range and linear deposition do not.
Both schemes approach the same continuum wake potential when bins are densely
populated; like JuTrack's histogram scheme, both smooth the kick of
sparsely populated edge bins at the bin scale, so choose `nbins` such that
many macroparticles populate each bin. Stored single-pass results differ at
the order of the old bin width. No input conversion is required.

Several element behaviors were corrected against JuTrack and are breaking
for previously stored single-pass results:

- `Translation` now applies its longitudinal `ds` shift with TrackPad's
  drift-consistent update (a synchronous particle keeps its ``z``). Earlier
  versions copied JuTrack's uncompensated ``\pm ds\,(\beta_0^{-1}+\delta)``
  term, which mixed axis conventions. With ``dx = dy = 0``,
  `Translation(0; ds=ds)` equals `Drift(ds)`. This intentionally diverges
  from JuTrack's map. Scalar and PolySeries tracking use the same convention.
- Rectangular (`r_apertures`) and elliptical (`e_apertures`) apertures are
  now enforced in multi-particle CPU tracking: macroparticles outside an
  aperture are flagged as lost at that element while keeping their evolved
  coordinates. Earlier versions stored the fields but never applied them.
- `SBend`, `RBend`, `ExactSBend`, `ERBend`, and `SBendSC` raise their kick
  expansion order automatically when higher multipoles are present in
  `polynom_b` (JuTrack behavior); earlier versions required a manual
  `max_order` and silently dropped gradients otherwise.
- The Forest (13.29) multipole entrance/exit fringe correction is applied by
  `Quadrupole`, `Sextupole`, `Octupole`, and the bends when their fringe
  flags are nonzero. `ThinMultipole` fringe flags stay inert on both sides
  of the migration (JuTrack's Float64 thin-multipole pass ignores them too).
- Wiggler harmonics must contain complete six-value blocks and nonzero
  denominator wave-vector entries: ``k_x,k_z\neq0`` for vertical-field `Bx`
  blocks and ``k_y,k_z\neq0`` for horizontal-field `By` blocks. Invalid
  configurations are rejected during construction.

## Reference Beam

`Beam(energy; mass, charge)` uses the following quantities:

| Field | Unit | Meaning |
|-------|------|---------|
| `energy` | ``\mathrm{eV}`` | Reference kinetic energy ``K_0`` |
| `mass` | ``\mathrm{eV}`` | Rest-mass energy ``m_0c^2`` |
| `charge` | ``e`` | Signed reference-particle charge ``q/e`` |
| `gamma` | ``1`` | ``\gamma_0=E_0/(m_0c^2)=1+K_0/(m_0c^2)`` |
| `beta` | ``1`` | ``\beta_0=P_0c/E_0=v_0/c`` |

The total energy and reference momentum satisfy

```math
E_0=K_0+m_0c^2,
\qquad
P_0c=\sqrt{E_0^2-m_0^2c^4}.
```

`Beam(3e9)` is therefore a ``3\,\mathrm{GeV}`` kinetic-energy electron beam,
not a ``3\,\mathrm{GeV}`` total-energy beam. PALS `pc_ref` and `E_tot_ref`,
and MAD-X `PC` and `ENERGY`, are converted to kinetic energy by the readers.

## Magnet Strengths

Named multipole constructors take normalized strengths:

| Constructor field | Unit |
|-------------------|------|
| `Quadrupole.k1` | ``\mathrm{m}^{-2}`` |
| `Sextupole.k2` | ``\mathrm{m}^{-3}`` |
| `Octupole.k3` | ``\mathrm{m}^{-4}`` |

Pass ``K_2`` and ``K_3`` without factorial scaling. Tracking inserts
``K_2/2!`` and ``K_3/3!`` into the polynomial kick. PALS `Kn1`, `Kn2`, and
`Kn3` map directly to these named strengths. Integrated PALS values `Kn1L`,
`Kn2L`, and `Kn3L` are divided by element length during compilation.

The raw `polynom_a` and `polynom_b` arrays contain skew and normal coefficients
``A_n`` and ``B_n``, ordered from dipole (``n=0``) through octupole
(``n=3``). The thin kick evaluates

```math
\mathcal{B}(x,y)=\sum_{n=0}^{N}(B_n+\mathrm{i}A_n)(x+\mathrm{i}y)^n,
```

and applies

```math
\Delta p_x=-L\,\Re\mathcal{B},
\qquad
\Delta p_y=+L\,\Im\mathcal{B}.
```

These lower-level coefficients are not interchangeable with named ``K_2`` or
``K_3`` unless the factorial normalization is applied.

## Bend Geometry

For `SBend(L, angle, e1, e2)`:

- ``L`` is the reference arc length in metres.
- ``\theta=\mathtt{angle}`` is the signed total bend angle in radians.
- The body curvature is ``h=1/\rho=\theta/L`` when ``L\ne0``.
- ``e_1`` and ``e_2`` are entrance and exit pole-face angles in radians.
- `gap` is the full magnet gap in metres.

`RBend(L, angle)` uses the same body map with
``e_1=e_2=\theta/2``. Importers convert chord/rectangular geometry to
TrackPad's arc-length representation before construction.

## RF Phase and Charge

For `RFCavity`, the implemented phase is

```math
\phi=-2\pi f\frac{z+\ell_{\mathrm{lag}}}{c}-\phi_{\mathrm{lag}},
```

where `lag` stores ``\ell_{\mathrm{lag}}`` and `philag` stores
``\phi_{\mathrm{lag}}``.

| Field | Unit | Meaning |
|-------|------|---------|
| `volt` | ``\mathrm{V}`` | Peak cavity voltage ``V`` |
| `freq` | ``\mathrm{Hz}`` | RF frequency ``f`` |
| `lag` | ``\mathrm{m}`` | Longitudinal phase offset ``\ell_{\mathrm{lag}}`` |
| `philag` | ``\mathrm{rad}`` | Additional phase ``\phi_{\mathrm{lag}}`` subtracted from ``\phi`` |
| `energy` | ``\mathrm{eV}`` | Reference kinetic energy ``K_0`` used by the kick normalization |
| `charge` | ``e`` | Signed reference-particle charge ``q/e`` |

Let ``\widehat q=q/e`` be the signed charge value stored in `charge`. TrackPad
computes ``P_0c`` from the reference kinetic energy and ``\beta_0`` and applies
the canonical kick

```math
\delta_E^+
=\delta_E^-
-\frac{\widehat q V}{P_0c}\sin\phi.
```

`CrabCavity` and `AccelCavity` use the same ``\widehat qV/(P_0c)``
normalization and expose `charge` for the signed reference charge. Their phase
depends on ``-kz`` because ``z=-c\Delta t``.

`LorentzBoost` and `InvLorentzBoost` form a canonical inverse pair. TrackPad
keeps JuTrack's crossing-angle coordinate transformation but derives the
momentum transformation from symplecticity; consequently these maps no longer
have raw numerical parity with JuTrack's non-symplectic momentum scaling.

A directly constructed `RFCavity` defaults to `energy=0`, which intentionally
disables its kick. Set `energy=beam.energy` and `charge=beam.charge` for direct
construction. PALS and MAD-X readers set both fields from the reference beam.

PALS/MAD-X phase values expressed in cycles are converted as
``\ell_{\mathrm{lag}}=\mathrm{phase}\,c/f``. For an electron ring above
transition, the stable no-acceleration MAD-X setting remains `LAG=0.5` because
the signed reference charge is included in the kick.

For `LongitudinalRFMap`, with ``k=2\pi f/c`` and
``\eta=\alpha_c-1/\gamma_0^2``, the first-order energy-coordinate slip is

```math
z^+=z^--\frac{2\pi h\eta}{\beta_0 k}\,\delta_E.
```

The additional ``1/\beta_0`` relative to a momentum-coordinate formula follows
from ``\delta_E=\beta_0\delta_P`` at the reference particle.

## Loss Convention

For real-valued scalar tracking, `check_lost(r)` returns `true` when:

- ``x`` or ``y`` is `NaN`;
- ``\lvert x\rvert`` or ``\lvert y\rvert`` exceeds ``1\,\mathrm{m}``; or
- ``\lvert p_x\rvert`` or ``\lvert p_y\rvert`` exceeds ``1``.

Some maps also produce `NaN` coordinates when ``\pi_s^2\le0``. Matrix tracking
uses integer flags (`0` alive, `1` lost). Packed GPU tracking writes nonfinite
coordinates for lost particles. These are current global safety limits, not
element apertures.

## Optics and Finite Differences

- `one_turn_map` and `findm66` compute numerical Jacobians.
- `gettune` extracts uncoupled tunes from transverse ``2\times2`` blocks.
- `twissline` and `periodic_twiss` compute uncoupled periodic Twiss functions;
  `transport_twiss` propagates supplied line optics. Results default to stored
  element boundaries. `sample_integrator_steps=true` exposes configured thick-
  element steps, and `max_step` refines long drifts and bends. During bend
  refinement, entrance edge/fringe/frame maps remain on the first piece and
  exit maps remain on the last piece.
- `getchrom` defaults to the JuTrack-compatible forward difference; use
  `centered=true` for a centered derivative.
- Coupled optics are not yet represented by `TwissLineResult`.

The legacy keywords `dp` and `dpp` denote offsets and steps in TrackPad's sixth
coordinate ``\delta_E``, despite their names. Thus `getchrom` returns
``\mathrm{d}Q/\mathrm{d}\delta_E``. At the reference energy,

```math
\frac{\mathrm{d}Q}{\mathrm{d}\delta_P}
=\beta_0\frac{\mathrm{d}Q}{\mathrm{d}\delta_E}.
```

Finite-difference defaults are part of compatibility behavior. Specify ``h``,
``\Delta\delta_E``, the reference orbit, and centered/closed-orbit choices when
reporting a cross-code comparison.

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
