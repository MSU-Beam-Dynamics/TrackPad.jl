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

## Element Maps

The maps below fix the physics that every tracking path — scalar CPU, packed
GPU, TPSA and AD — implements identically.

### [Exact drift](@id exact_drift)

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

This equation fixes both the meaning of `delta` and the sign of `z`. The
bracket is evaluated as

```math
\frac{\beta_0^{-1}+\delta_E}{\pi_s}-\beta_0^{-1}
=\frac{(1-\beta_0^{-2})\,\delta_E(2\beta_0^{-1}+\delta_E)+\beta_0^{-2}(p_x^2+p_y^2)}
      {\pi_s\left[(\beta_0^{-1}+\delta_E)+\beta_0^{-1}\pi_s\right]},
```

which is the same number without the subtraction of two ``O(L)`` terms. The
direct form loses ``\sim\varepsilon L`` per drift, and that roundoff, summed
over a ring, was the floor of every finite-difference ``M_{56}`` (momentum
compaction) at ``10^{-6}``; with this form `periodic_twiss(...; method=:fd)`
and `method=:tpsa` agree on ``\alpha_c`` to ``10^{-12}``. The exact sector
bend body (`ExactSBend`) is written the same way: its ``x`` update is Forest's
``[\pi_{s,f}-(\pi_{s,i}-1-hx)\cos\theta+p_x\sin\theta-1]/h`` with every
``\pi_s-1`` formed as ``(p^2-1-p_x^2-p_y^2)/(\pi_s+1)``, so that the
``\varepsilon/h\approx10^{-14}`` m per step of the literal expression (which
corrupts finite-difference optics of exact-bend rings at ``10^{-4}``) does not
occur.

`USE_EXACT_HAMILTONIAN` is a compile-time constant (`true`); all tracking,
optics, TPSA and AD paths use the exact drift. The JuTrack-compatible
ultrarelativistic approximation is still available for cross-code comparison
through `drift6(r, L, beti, Val(false))`; it changes nonlinear maps and
chromaticity.

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
`Corrector`, `LBend`, the non-radiating `Wiggler`, and the RF-family maps. In
the standard (AT-style) bend the dispersion kick is ``h\,\delta_P\,L`` with
``\delta_P=\delta_E(2\beta_0^{-1}+\delta_E)/(1+p)`` exactly, not the
linearised ``h\,\delta_E L/\beta_0`` of AT and JuTrack; the two agree for
``\beta_0=1`` but the linearised kick makes ``Q(\delta_P)`` of a
``\beta_0=0.875`` ring differ from the ``\beta_0\to1`` ring at
``O(\delta_P^2)`` (``9\times10^{-6}`` at ``\delta_P=2\times10^{-3}``),
which is not physical: with normalised strengths the ``\delta_P`` dynamics
of a ring without RF is species independent.
The scalar CPU, packed GPU, and PolySeries implementations use the same
conversion where each element is supported.

### Dipole edge

The AT-style edge is the thin kick ``p_x \mathrel{+}= f_x x``,
``p_y \mathrel{-}= f_y y``, i.e. the flow of
``H=-f_x x^2/2 + f_y y^2/2``. Every fringe model but Brown-without-integral
makes ``f_y`` depend on momentum — ``1/(1+\delta_E)`` for Brown with a fringe
integral and for SOLEIL, ``p_x/(1+\delta_E)`` for THOMX — so the map must also
carry

```math
z \mathrel{+}= \frac{y^2}{2}\frac{\partial f_y}{\partial\delta_E},
\qquad
x \mathrel{+}= \frac{y^2}{2}\frac{\partial f_y}{\partial p_x}.
```

AT and JuTrack omit both, which is the sole source of their edge
non-symplecticity. `SYMPLECTIC_BEND_EDGE` is a compile-time constant (`true`)
that includes them; `Val(false)` as a seventh argument to
`edge_fringe_entrance`/`edge_fringe_exit` restores the AT map for cross-code
comparison. Measured symplectic violation of the edge Jacobian at
``y\sim10^{-3}`` (finite-difference floor ``7.5\times10^{-12}``):

| fringe model | AT | TrackPad |
|---|---|---|
| none, or Brown with ``\mathrm{fint}\cdot\mathrm{gap}=0`` | 7.5e-12 | 7.5e-12 |
| Brown (1) | 3.5e-07 | 7.5e-12 |
| SOLEIL (2) | 1.2e-05 | 7.5e-12 |
| THOMX (3) | 1.1e-04 | 1.7e-06 |

THOMX keeps a ``p_x`` dependence, so its corrected form is a first-order
splitting rather than an exact flow. The correction is ``O(y^2)`` and confined
to ``z`` (and ``x`` for THOMX), so the transverse map at fixed momentum is
unchanged bit-for-bit: tunes, Twiss, chromaticity and dispersion do not move,
and neither does 4-D tracking. It matters only where ``z`` feeds back — 6-D
tracking with RF, synchro-betatron studies, momentum aperture.

### Strong beam-beam kick

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

Because the kick is collective, single-particle tracking (`track` on an
`SVector`, `linepass`, TPSA maps) rejects these elements; use the
``N\times 6`` multi-particle form of `track!`, which is serial for them
(`threaded=true` rejects them). GPU tracking rejects them during `GPULattice`
construction.

## Reference Beam

`Beam(energy; mass, charge)` uses the following quantities:

| Field | Unit | Meaning |
|-------|------|---------|
| `energy` | ``\mathrm{eV}`` | Reference **total** energy ``E_0`` |
| `mass` | ``\mathrm{eV}`` | Rest-mass energy ``m_0c^2`` |
| `charge` | ``e`` | Signed reference-particle charge ``q/e`` |
| `gamma` | ``1`` | ``\gamma_0=E_0/(m_0c^2)`` |
| `beta` | ``1`` | ``\beta_0=P_0c/E_0=v_0/c`` |

The kinetic energy and reference momentum follow from

```math
K_0=E_0-m_0c^2,
\qquad
P_0c=\sqrt{E_0^2-m_0^2c^4}=\beta_0E_0,
```

and are returned by `kinetic_energy(beam)` and `p0c(beam)`. `Beam(3e9)` is a
``3\,\mathrm{GeV}`` total-energy electron beam, the same meaning as MAD-X
`ENERGY`. For low-energy machines, where the kinetic energy or the momentum is
the natural specification, use `Beam(kinetic=K_0, mass=m)` or
`Beam(pc=P_0c, mass=m)`; a positional value below the rest mass is rejected,
which catches a kinetic energy passed by mistake. PALS `E_tot_ref`/`pc_ref` and
MAD-X `ENERGY`/`PC` map onto these forms directly.

Before 0.2 the positional argument was the kinetic energy, as in JuTrack;
`Beam(kinetic=K)` reproduces the old `Beam(K)` bit for bit.

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

### [Bend models and what other codes compute](@id bend_models)

`SBend` is AT's `BndMPoleSymplectic4Pass`: the Hamiltonian expanded to second
order in the transverse coordinates, so the body has no ``(1+hx)`` factor on
the kinetic term. `ExactSBend` is Forest's exact sector bend, and by default
applies the hard-edge dipole fringe at both faces
(`fringe_bend_entrance = fringe_bend_exit = 1`, nonzero even for
``e_1=e_2=0``). Which one a reference code corresponds to:

| Code / model | TrackPad equivalent |
|---|---|
| pyAT `BndMPoleSymplectic4Pass` | `SBend` (bit-for-bit) |
| PTC `exact=true`, MAD-X `SBEND` (twiss), pyAT `ExactSectorBendPass` | `ExactSBend` |
| Xsuite `Bend` with `edge_*_model='full'` | `ExactSBend` |
| Xsuite `Bend` with `edge_*_model='linear'` (its default) | `ExactSBend(...; fringe_bend_entrance=0, fringe_bend_exit=0)` |

On a 24-cell ring of 7.5° bends (``\rho=27.5`` m) the three differ on the
chromaticity alone: `SBend` gives ``\xi=(12.942, 5.193)``, the exact bend
without fringe ``(13.431, 5.382)`` and with fringe ``(13.431, 5.449)``; tunes,
Twiss functions, dispersion, ``\alpha_c`` and the radiation integrals agree
to ``10^{-7}`` or better. The expanded body misses the term
``h\,x\,(p_x^2+p_y^2)/2(1+\delta)`` whose closed-orbit value
``h D\,\delta`` acts as a chromatic drift-length change,
``\Delta\xi\simeq\frac{1}{4\pi}\oint\gamma\,hD\,\mathrm ds`` — a few
percent of ``\xi`` at this bending radius, negligible for
``\rho\gtrsim100`` m. Use `ExactSBend` when comparing chromaticities with
MAD-X, PTC or Xsuite on compact rings. `test/verify_ring_optics.jl` pins the
numbers of all four codes on this ring.

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
| `energy` | ``\mathrm{eV}`` | Reference total energy ``E_0`` used by the kick normalization (pass `beam.energy`) |
| `charge` | ``e`` | Signed reference-particle charge ``q/e`` |

Let ``\widehat q=q/e`` be the signed charge value stored in `charge`. TrackPad
computes ``P_0c=\beta_0E_0`` from the element's total energy and the beam's
``\beta_0`` and applies the canonical kick

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

For `LongitudinalRFMap`, with ``k=2\pi f/c``, circumference ``C=2\pi h/k`` and
``\eta=\alpha_c-1/\gamma_0^2``, the first-order slip in the stored coordinates is

```math
z^+=z^--\frac{2\pi h\eta}{\beta_0^2 k}\,\delta_E
     =z^--\frac{C\eta}{\beta_0^2}\,\delta_E.
```

Relative to the momentum-coordinate formula ``\Delta(ct)=C\eta\,\delta_P``
this carries two factors of ``1/\beta_0``: one because ``z=s/\beta_0-ct``
measures path length in units of ``\beta_0 c\,t``, and one because
``\delta_E=\beta_0\delta_P`` at the reference particle. The formula is
verified in the tests against the path length of the tracked off-momentum
closed orbit of a ``\beta_0=0.875`` proton ring.

## [Loss Convention](@id loss_convention)

Two mechanisms mark a particle lost.

*Global safety limits.* For real-valued coordinates `check_lost(r)` is `true`
when ``x`` or ``y`` is `NaN`, ``\lvert x\rvert`` or ``\lvert y\rvert`` exceeds
``1\,\mathrm{m}``, or ``\lvert p_x\rvert`` or ``\lvert p_y\rvert`` exceeds
``1``. Kernels also return `NaN` coordinates when ``\pi_s^2\le0`` (a particle
that cannot propagate forward). For non-`Real` coordinates — TPSA series, dual
numbers — `check_lost` is always `false`.

*Element apertures.* `r_apertures` (rectangular) and `e_apertures` (elliptical)
are tested after each element in every tracking path when set.

How a loss is reported depends on the path:

| path | report |
|---|---|
| `track`, `track!` (scalar or matrix, any backend) | `NaN` coordinates, particle skipped thereafter; the optional `lost` vector of `track!` also carries a flag per particle (`check_apertures=false` tracks through apertures as if absent) |
| `linepass` (scalar, JuTrack-compatible) | returns immediately; an aperture loss returns `NaN`, a coordinate-limit loss the coordinates the particle held |
| `linepass!` (matrix, JuTrack-compatible) | integer flag `1` in `lost_flags`; an aperture loss keeps the evolved coordinates |

A TPSA map is always computed as if the apertures were absent.

## Optics and Finite Differences

- `one_turn_map` and `findm66` compute numerical Jacobians.
- `gettune` extracts uncoupled tunes from transverse ``2\times2`` blocks.
- `twiss` computes uncoupled optics as a `TwissResult`: without an `entrance`
  the periodic optics of a ring about its closed orbit — Twiss functions,
  dispersion, momentum compaction, slip factor and chromaticity, plus optional
  second-order chromaticity, radiation integrals and amplitude detuning
  (`periodic_twiss`, alias `twissline`); with an `entrance` (an `optics4DUC`
  or another `TwissResult`) the propagation of those optics, dispersion
  included, through a line (`transport_twiss`). Results default to element
  boundaries; `slices`, `max_step` and `sample_integrator_steps` split
  elements of finite length into pieces of one integration step each, never
  fewer than configured. During bend refinement, entrance edge/fringe/frame maps remain on
  the first piece and exit maps remain on the last piece.
- `getchrom` returns ``\mathrm{d}Q/\mathrm{d}\delta_P`` about the off-momentum
  closed orbit, which is the chromaticity of the ring; there is no on-axis
  option, because with sextupoles present the on-axis trajectory does not
  measure a well-defined quantity — see [Tunes and chromaticity](@ref chromaticity)
  in the guide. The
  finite difference is centered by default (`centered=false` for one-sided).
- Momentum compaction is defined through the closed-orbit path length,
  ``\alpha_c=(1/C)\,\mathrm{d}C/\mathrm{d}\delta_P``, and the slip factor is
  ``\eta=\alpha_c-1/\gamma_0^2``. In the stored coordinates the off-momentum
  closed orbit slips by ``\Delta z=-C\,\eta\,\delta_E/\beta_0^2`` per turn,
  which is the map `LongitudinalRFMap` applies; `transition_gamma` returns
  ``1/\sqrt{\alpha_c}``.
- The radiation integrals sample each bend body internally (64 slices or the
  bend's integration-step count, composite Simpson, with the end nodes
  extrapolated past the pole-face maps), so they do not depend on the output
  sampling; sampling only at element boundaries would overestimate ``I_1``,
  ``I_4`` and ``I_5`` by ``(hL)^2 D''/12D\approx4\%`` on a 7.5° bend. The
  pole-face contribution to ``I_4`` is not included.
- Every quantity in `TwissResult` is available from truncated Taylor
  maps (`method=:tpsa`, the default) or from finite differences of tracking
  (`method=:fd`). The two are cross-checked in the test suite; `:fd` is the
  route for lattices whose elements have no series map (`LBend`).
- Coupled optics are not yet represented by `TwissResult`.

### The interface speaks ``\delta_P``; the state vector stores ``\delta_E``

These are separate decisions and TrackPad makes them differently on purpose.

*Coordinate vectors* — anything tracked, and every `reference`, `orb`, `x0` or
returned orbit — are canonical coordinates, so their sixth entry is always the
stored ``\delta_E``. So is the sixth row and column of any Jacobian returned by
`findm66`, `fastfindm66` or `one_turn_map`.

*Scalar momentum parameters and derived energy derivatives* use
``\delta_P=(P-P_0)/P_0``, the convention MAD-X, elegant and AT speak at their
interfaces. That covers the `dp` argument of `fastfindm66`, `findm66`,
`twissring`, `twissline`, `periodicEdwardsTengTwiss` and
`find_closed_orbit_4d`; the `dp`, `dpp` and `dpp2` of `getchrom` and
`periodic_twiss`; and the values returned by `getchrom` and `periodic_twiss`
(dispersion, chromaticities, `alphac`, `slip`). Every one of these takes
`wrt=:deltae` to switch to the stored coordinate for cross-code work against a
code that stores it.

Offsets are converted through the exact relation, not the linear one:

```math
(1+\delta_P)^2=1+\frac{2\delta_E}{\beta_0}+\delta_E^2,
\qquad
\frac{\mathrm{d}Q}{\mathrm{d}\delta_P}
=\beta_0\frac{\mathrm{d}Q}{\mathrm{d}\delta_E}
\ \text{at the reference energy.}
```

`deltap_from_deltae` and `deltae_from_deltap` are exported for converting by
hand; both are written to stay accurate at finite-difference-sized arguments,
where a naive ``\sqrt{1+\varepsilon}-1`` loses half its digits. The two
conventions differ by the local ``\beta``: 1.4e-8 relative for a 3 GeV
electron, 12.5% for a 1 GeV proton.

!!! note "Changed in 0.2"
    `getchrom` and the dispersion previously were ``\delta_E``
    derivatives, and `dp` arguments were ``\delta_E`` offsets. Code that
    multiplied a result by `beam.beta` by hand should drop that factor; code
    that needs the old behavior should pass `wrt=:deltae`. `getchrom` now
    always measures about the off-momentum closed orbit; the former
    `closed_orbit=false` on-axis launch has been removed (see below for how to
    reproduce JuTrack's number by hand).

Finite-difference defaults are part of compatibility behavior. Specify ``h``,
``\Delta\delta_P``, the reference orbit and the centered/one-sided choice when
reporting a cross-code comparison, or use `method=:tpsa`, which has none of
them. The reference-orbit choice dominates any comparison with a code that
launches on-axis: it shifts ``\xi`` by a few percent of the sextupole
contribution, orders of magnitude more than any step-size effect, and it does
not converge away as the steps are refined.

## Differences from JuTrack

TrackPad began as a rewrite of JuTrack and keeps JuTrack-style entry points
(`findm66`, `twissring`, `fastfindm66`, …), but several conventions were
corrected on the way and are deliberately **not** JuTrack-compatible. Tests that
compare the two codes apply the corresponding transformations explicitly. They
do not run JuTrack: its side of every comparison is frozen in
`test/jutrack_reference.jl`, so TrackPad has no JuTrack dependency (see
`test/jutrack_reference/README.md` for how that data is regenerated).

- **Longitudinal sign.** JuTrack's sixth-pair coordinate is the noncanonical
  positive-delay ``c(t-t_0)``; TrackPad's ``z=-c(t-t_0)`` is canonical and
  positive for an early particle. Coordinates convert with
  ``C=\operatorname{diag}(1,1,1,1,-1,1)`` and first-order maps with
  ``M_{\mathrm{TrackPad}}=C\,M_{\mathrm{JuTrack}}\,C``.
- **Finite ``\beta_0``.** Bends, RF cavities, solenoids, correctors, linear
  bends and wigglers use the exact ``\delta_E``-to-momentum conversion above,
  including the dispersion kick of the standard bend (JuTrack: ``\delta_E/\beta_0``).
  JuTrack's ultrarelativistic ``\delta_E\simeq\delta_P`` maps are recovered
  only in that limit.
- **Roundoff.** The drift ``z`` update and the exact-bend ``x`` update are
  evaluated without cancelling ``O(1)`` terms (see [Exact drift](@ref exact_drift)); JuTrack's
  literal forms leave ``\sim\varepsilon L`` and ``\sim\varepsilon/h`` per
  step. Parity tests therefore compare to JuTrack's roundoff
  (``10^{-14}``–``10^{-13}``), not to the bit.
- **Chromaticity and dispersion** are ``\delta_P`` derivatives about the
  off-momentum closed orbit; JuTrack differentiates the stored coordinate along
  an on-axis launch, and TrackPad offers no option for that. To reproduce
  JuTrack's number, difference the tunes of
  `findm66(lat, ±dp, beam; reference=SVector(0,0,0,0,0,0), wrt=:deltae)`
  by hand, as `test/verify_optics.jl` does.
- **Dipole edges** carry the longitudinal term that makes the AT-style fringe
  map symplectic (`SYMPLECTIC_BEND_EDGE`); JuTrack and AT omit it.
- **`Translation`** applies its longitudinal `ds` with the drift-consistent
  update, so a synchronous particle keeps its ``z`` and `Translation(0; ds=ds)`
  equals `Drift(ds)`. JuTrack's uncompensated ``\pm ds\,(\beta_0^{-1}+\delta)``
  term is not reproduced.
- **Apertures** (`r_apertures`, `e_apertures`) are enforced in every tracking
  path; see [Loss Convention](@ref loss_convention).
- **Wake deposition** uses cloud-in-cell deposition and a bin-edge-interpolated
  kick on a padded grid; JuTrack uses nearest-bin deposition. Both converge to
  the same continuum wake potential, so choose `nbins` such that many
  macroparticles populate each bin.

Behaviours that follow JuTrack and are worth knowing:

- `SBend`, `RBend`, `ExactSBend`, `ERBend` and `SBendSC` raise their kick
  expansion order automatically when `polynom_b` carries higher multipoles.
- The Forest (13.29) multipole fringe correction is applied by `Quadrupole`,
  `Sextupole`, `Octupole` and the bends when their fringe flags are nonzero;
  `ThinMultipole` fringe flags are inert.
- Wiggler harmonics must contain complete six-value blocks with nonzero
  denominator wave-vectors (``k_x,k_z\neq0`` for `Bx` blocks, ``k_y,k_z\neq0``
  for `By` blocks); invalid tables are rejected at construction.

## Time Dependence

`TimeContext(real_time; turn)` carries both physical time and turn index.
`track`, `track!` and `linepass` resolve `TimeVaryingElement` fields at the
supplied context, once per turn at `(time + (n-1)*dt_turn, turn + n - 1)`. Materialize a time-dependent lattice before GPU adaptation because
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
