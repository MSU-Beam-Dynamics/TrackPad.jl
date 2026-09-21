# Changelog

All notable changes to TrackPad.jl are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.2.0] — 2026-09-18

First public release. Earlier development snapshots (0.1.0) were never
tagged or registered; the notes below describe what changed relative to them,
for anyone who used the repository directly.

### Breaking

- **`Beam(energy)` takes the total energy.** The positional argument is now
  `E0` in eV — what MAD-X `ENERGY` and PALS `E_tot_ref` mean — instead of the
  kinetic energy inherited from JuTrack. `Beam(kinetic=K, mass=m)` and
  `Beam(pc=P0c, mass=m)` cover low-energy machines; `kinetic_energy(beam)` and
  `p0c(beam)` read the derived quantities back. A positional energy below the
  rest mass throws, which catches a kinetic energy passed by mistake. The
  `energy` field of `RFCavity`, `CrabCavity`, `AccelCavity` and `Wiggler` is
  likewise total (the Wiggler already was, so the elements are now consistent
  with each other), and `P0c = β0 E0` replaces the kinetic-energy formula in
  the RF normalisation on CPU and GPU. Readers map MAD-X `ENERGY`/`PC` and PALS
  `E_tot_ref`/`pc_ref` onto the new forms directly; `examples/data/fodo.madx`
  now says `ENERGY=3.0`. For a 3 GeV electron the shift is 1.7e-4 in γ0.
  `Beam(kinetic=3.0e9)` reproduces the old `Beam(3.0e9)` bit for bit.
- **Momentum convention at the interface.** Every `dp` argument
  (`getchrom`, `find_closed_orbit_4d`, `findm66`, `fastfindm66`,
  `findm66_refpts`, `fastfindm66_refpts`, `twissring`, `twissline`,
  `periodicEdwardsTengTwiss`), the `dpp` step of `getchrom`, and the values
  returned by `getchrom` and the dispersion of `periodic_twiss` are now relative momentum
  `δP = (P-P0)/P0`, matching MAD-X, elegant and AT. Coordinate vectors, `orb`,
  `reference` and returned Jacobians keep the stored `δE = (E-E0)/(P0c)`.
  Conversion is exact, via `(1+δP)^2 = 1 + 2δE/β0 + δE^2`. Pass `wrt=:deltae`
  to any of these for the previous behaviour, and drop hand-applied `beam.beta`
  factors. Invisible for ultrarelativistic beams (β0 = 1 − 1.4e-8 at 3 GeV
  electrons), a 12.5 % effect for a 1 GeV proton.
- **Chromaticity is always measured about the off-momentum closed orbit.**
  `getchrom` (and the chromaticities, dispersion and momentum compaction of
  `periodic_twiss`) re-solve the 4-D closed orbit at every momentum step, so
  the sextupole feed-down is sampled where the beam actually sits — the
  chromaticity of the ring. The previous default, an on-axis launch (JuTrack's
  convention), samples the feed-down along a trajectory that betatron-
  oscillates about the closed orbit and is off by a few percent of the
  sextupole contribution (on a ring corrected to ξ = 0 it reported
  (+0.30, −0.47)); the `closed_orbit` keyword is removed and there is no
  on-axis option. `getchrom` throws on a lattice with no stable off-momentum
  closed orbit and on a non-periodic lattice. The finite difference is now
  centered by default (`centered=false` for one-sided).
- **PolySeries.jl is a hard dependency** and Taylor maps are the default
  `method` of `getchrom` and `periodic_twiss` (see
  Added). Code that relied on the finite-difference numbers to the bit should
  pass `method=:fd`; installations must add PolySeries by URL until it is
  registered.
- **One tracking interface.** `track(lat, r_or_coords, beam; nturns=1, ...)`
  and `track!(coords, lat, beam; nturns=1, lost=nothing, threaded=false, ...)`
  replace the line/ring split: one particle or an `N x 6` matrix, one pass or
  many turns (`nturns > 1` requires a periodic lattice), serial or threaded,
  and `track!(coords, gpulattice; nturns)` for a packed `GPULattice`. Lost
  particles come back as `NaN` from every path, with the optional `lost`
  vector carrying a flag per particle. **`ringpass`, `ringpass!` and
  `batch_ringpass!` are removed** — `track(...; nturns=n)` and
  `track!(...; nturns=n)` replace them. `linepass`, `linepass!`,
  `cpu_batch_linepass!` and `batch_linepass!` remain as the single-pass /
  positional spellings, `linepass`/`linepass!` keeping JuTrack's loss
  convention (a flag, coordinates left as they were). `linepass!` now accepts
  any flag vector and a `Beam` whose precision differs from the coordinates.
- **One Twiss interface.** `twiss(lat, beam; entrance=nothing, kwargs...)`
  replaces the pair `TwissLineResult`/`TransportTwissResult` with a single
  `TwissResult` for rings and lines. Without `entrance` it is the periodic
  solution (`periodic_twiss`, `twissline`), with `entrance` — an `optics4DUC`
  (now read with its `eta`/`etap`) or another `TwissResult` to chain from —
  the propagation through a line (`transport_twiss`), dispersion included.
  The result gained the fields `periodic`, `method`, `dx`, `dpx`, `dy`, `dpy`,
  `length` (was `circumference`), `alphac`, `slip`, `chromx`, `chromy`, and
  the optional `chrom2x`, `chrom2y` (`second_order=true`),
  `radiation = (I1, …, I5)` (`radiation_integrals=true`) and `detuning`
  (2×2 `∂Q/∂J`, `detuning=true`); ring-only fields are `nothing` for a line,
  and a line's `tunex`/`tuney` are its total phase advance over 2π. Both
  cases accept the same keywords.
- **Sampling keywords.** `slices=n` splits every element of finite length
  into at least `n` pieces for smooth optics inside thick elements; `max_step`
  now also splits thick multipoles, not only drifts and bends. Neither ever
  integrates an element in fewer steps than configured (`refine_lattice` has
  the same keywords).
- **`periodic_dispersion` and `DispersionLineResult` are removed.** The
  dispersion is the `dx`, `dpx`, `dy`, `dpy` of `periodic_twiss`, computed from
  the same Jacobians as the Twiss functions with the same `wrt`, `method` and
  sampling keywords; the wrapper only re-packaged those arrays at the full
  cost of `periodic_twiss`. Replace `periodic_dispersion(lat, beam; kw...).dx`
  by `periodic_twiss(lat, beam; kw...).dx`.
- **`LongitudinalRFMap` slip corrected to `Δz = −C η δE / β0²`** (was `/β0`):
  the stored `z = s/β0 − ct` needs a second `1/β0` on top of the `δE → δP`
  conversion. Ultrarelativistic rings are unaffected; a `β0 = 0.875` proton
  ring was slipping 12.5 % too little. The corrected map matches the path
  length of the tracked off-momentum closed orbit.
- **Integration-step defaults** are per element type: `Quadrupole`,
  `QuadrupoleSC`, `SBend`, `SBendSC`, `ExactSBend` use 4 (was 10);
  `Sextupole`, `SextupoleSC` use 2 (was 10); `Octupole`, `OctupoleSC` use 1
  (was 10; a thin-kick element, converged at one step). On a
  240-element arc this is a 2.9x speed-up for a 1.8e-4 tune offset against a
  200-step reference; raise `num_int_steps` for precision optics. `read_madx`
  now defaults `num_int_steps` to `nothing` (per-type defaults) like
  `read_pals`; an explicit value still forces one count everywhere.
- **AT-style dipole edges are symplectic.** Edge maps with a fringe integral
  (Brown, SOLEIL, THOMX) now carry the longitudinal term their Hamiltonian
  implies, `z += y²/2 ∂fy/∂δ` (and `x += y²/2 ∂fy/∂px` for THOMX), which AT and
  JuTrack omit. Jacobian symplecticity error goes from 3.5e-7 / 1.2e-5 to the
  finite-difference floor. The correction is `O(y²)` and confined to `z`, so
  tunes, Twiss, chromaticity, dispersion and RF-free tracking are bit-for-bit
  unchanged; it matters only where `z` feeds back (6-D tracking with RF).
  `const SYMPLECTIC_BEND_EDGE = true`; `Val(false)` restores the AT map.
- Removed the exports `drift6!` and `strthinkick!`, which referred to functions
  that did not exist.
- Single-particle `linepass`/`ringpass` now enforce element apertures and
  return `NaN` coordinates on an aperture loss (`check_apertures=false` tracks
  through apertures as before). `cpu_batch_linepass!` enforces apertures and
  resolves turn-dependent parameters every turn; both previously ignored them.
- `find_closed_orbit_4d`/`_6d` throw when the Newton iteration does not
  converge (`strict=false` warns and returns the last iterate); they previously
  returned unconverged orbits silently.
- `ParamSweepLattice` rejects sweeping a bend's curvature slot, which the
  kernel derives from `angle/L`; the sweep would otherwise have been ignored.
- The longitudinal coordinate is the canonical `z = s/β0 − ct` (positive for
  an early particle); JuTrack's positive-delay convention is recovered with
  `diag(1,1,1,1,-1,1)`. All maps use the exact `δE`-to-momentum conversion at
  finite β0.

- Julia 1.12 or newer is required (PolySeries.jl, the TPSA companion, needs
  it).

### Added

- `Wiggler` takes `mass` and `charge` (electron defaults) and normalises its
  map to the reference particle — kinematics through `E/m`, the wiggler
  parameter through `|q|·Bmax·lw/m`, radiation through the particle's
  classical radius — instead of hard-coding the electron. Electron results
  are unchanged bit for bit.
- `deltap_from_deltae`, `deltae_from_deltap`: exact, cancellation-free
  conversions between the stored and momentum conventions.
- `periodic_twiss` computes, besides the Twiss functions, the dispersion,
  momentum compaction and slip factor (from the closed-orbit path length),
  chromaticity, and on request the second-order chromaticity (3-point
  stencil), the synchrotron-radiation integrals I1–I5 over the bend bodies,
  and the amplitude-dependent tune shift `∂Q_i/∂J_j` from tracking at several
  actions with a NAFF tune estimate. `transition_gamma(tw)` derives γ_tr.
- **Ring optics default to Taylor maps.** `getchrom` and `periodic_twiss`
  take `method=:tpsa` (default) or `method=:fd`. With
  `:tpsa` every quantity is an exact derivative of a truncated Taylor map
  about the closed orbit: closed orbit and transfer matrices from the order-1
  map, chromaticities from the order-2 map including the closed-orbit term
  Σ D_k ∂Q/∂x_k, detuning from the order-3 map — no step sizes and no noise
  floor. `:fd` (finite differences of tracking, the previous behaviour) agrees
  to its truncation error (tunes 1e-12, β 1e-11, ξ ~1e-6 absolute, ξ₂ ~1e-3),
  is 1.5–4× faster for the linear/chromatic quantities and 5–20× slower for
  the detuning; it remains the route for elements without a series map
  (`LBend`). Each quantity uses the lowest order that contains it (1 for
  closed orbit and Jacobians, 2 for ξ, 3 for ∂Q/∂J), a reference that already
  closes skips the Newton solve, and off-momentum orbits are seeded from the
  linear dispersion.
  **PolySeries.jl is therefore a hard dependency** (formerly the optional
  `TrackPadPolySeriesExt`; its code is now `src/tpsa_polyseries.jl` and
  `tpsa_map`, `polyseries_variables`, `polyseries_one_turn_map` are always
  available). Until PolySeries is registered, add it by URL before TrackPad.
- The hard-edge dipole fringe of `ExactSBend` has a Taylor map (PolySeries
  now provides `atan`), so the default `ExactSBend` works with `method=:tpsa`
  and `tpsa_map`.
- Strong beam–beam elements `StrongThinGaussianBeam` and `StrongGaussianBeam`
  with the Bassetti–Erskine field (Weideman Faddeeva), physical amplitude from
  `beambeam_amplitude`/`classical_radius`, Hirata synchro-beam slicing, a
  round-beam power series for TPSA maps, and a GPU kernel branch.
- `cpu_batch_linepass!` threads over contiguous particle chunks; function
  barriers in both matrix trackers remove one dynamic dispatch and one heap
  allocation per particle-element (2.3x faster, allocation-free, bit-identical
  results).
- Matched macroparticle generation (`matched_gaussian`, `matched_covariance`,
  `gaussian_distribution`) with exact finite-sample moments, and the covariance
  diagnostics `beam_covariance`, `projected_emittances`, `eigenemittances`.
- Sampled optics (`sample_integrator_steps`, `max_step`) and lattice glyphs for
  Makie (`plot_lattice!`, `lattice_plot_data`).
- Documentation: every exported symbol is documented and the build enforces it
  (`checkdocs = :exports`); new Performance page; conventions page
  restructured with a "Differences from JuTrack" section; MIT license,
  citation metadata, CI and documentation workflows.

### Fixed

- The standard bend's dispersion kick used the linearised `δE/β0` for `δP`
  (AT's and JuTrack's `β0 = 1` form). It now uses the exact
  `δP = δE(2/β0 + δE)/(1 + P/P0)`, so the off-momentum optics of a finite-β0
  ring no longer depend on the species at `O(δ²)` (9e-6 in the tune at
  δP = 2e-3 for a β0 = 0.875 proton ring; zero for electrons).
- `ExactSBend`'s body formed `x_new = (pz_new − pzmx cos + px sin − 1)/h`
  literally, losing ~1e-14 m per step to cancellation; over a ring this
  corrupted finite-difference Jacobians at 1e-4 (tune noise 1e-5, chromaticity
  off by units). The algebra is unchanged but every `pz − 1` is now formed as
  `(p² − 1 − px² − py²)/(pz + 1)`; finite differences and Taylor maps now
  agree on exact-bend rings as they do on standard-bend rings.
- The drift's `z` update `L[(1/β0 + δE)/pz − 1/β0]` is evaluated through its
  conjugate form without subtracting two `O(L)` numbers (CPU, GPU and exact
  bend). This removes the 1e-6 floor of the finite-difference momentum
  compaction: `method=:fd` and `method=:tpsa` now agree on `αc` to 1e-12.
  JuTrack parity tests compare to JuTrack's roundoff (1e-14) instead of the
  bit.
- Radiation integrals sampled `D` and `H` only at element boundaries, which
  overestimates `I1`, `I4`, `I5` by `(hL)² D''/12D` — 4 % on a 7.5° bend and
  inconsistent with the ring's own `αc·C`. Each bend body is now sliced
  internally and integrated with Simpson; `I1 = αc·C` holds to 1e-9 and
  `I1`–`I5` match MAD-X `synch_1..5` to 1e-8.
- Cross-code reference: `test/verify_ring_optics.jl` pins tunes, Twiss,
  dispersion, `αc`, `ξ`, `ξ₂`, `I1`–`I5` and `∂Q/∂J` of a 24-cell ring
  against MAD-X 5.09.03, PTC, Xsuite 0.114 and pyAT (see the conventions page
  for which bend model each code corresponds to).
- The PALS reader's `Wiggler` branch called `Wiggler(len)` with no parameters
  and always threw an unrelated "lw must be positive" error. PALS carries no
  wiggler parameter group TrackPad maps, so a PALS wiggler is now handled as an
  unsupported element: a clear error with `strict=true`, a drift with a warning
  otherwise. The I/O guide no longer lists wigglers as supported.
- `StrongGaussianBeam` was being handed a kinetic `beam.energy` in its
  `total_energy` slot by callers; with `Beam.energy` now total the two agree.
- The PolySeries extension re-implemented every tracking kernel for `CTPS`
  coordinates; the copies were proven bit-identical to the generic core and
  removed (790 → 162 lines). Only the round-beam series, the `ExactSBend`
  fringe guard, the `LBend` rejection and the TPSA utilities remain.
- `pass!(::Marker, r)` failed for TPSA coordinates because its default `beti`
  was built from the coordinate type (`one(S)`), which `CTPS` cannot provide.
- Twiss and dispersion functions resolve time-varying elements at `t = 0`
  instead of failing on `timed` lattices; kernels are generic in the coordinate
  type so `Float32` and dual coordinates flow through every element.
- Longitudinal wake deposition uses cloud-in-cell weights and a
  bin-edge-interpolated kick on a padded grid.

## [0.1.0] — unreleased

Internal development snapshots migrated from JuTrack.jl: element kernels,
PALS and MAD-X readers, PolySeries and Enzyme extensions, CUDA and Metal
backends.
