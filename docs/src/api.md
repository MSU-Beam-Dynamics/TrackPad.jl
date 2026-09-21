```@meta
CurrentModule = TrackPad
```

# API Reference

```@docs
TrackPad
```

## Reference Beam

```@docs
Beam
Beam(energy::T; mass::T, charge::T) where T
kinetic_energy
p0c
beti
M_ELECTRON
M_PROTON
```

## Beam Distributions

```@docs
gaussian_distribution
matched_gaussian
matched_covariance
match_moments!
beam_covariance
projected_emittances
eigenemittances
```

## Element Type Hierarchy

Concrete elements are documented on the [Elements](@ref elements_guide) page;
these are the abstract parents used for dispatch and for user extensions.

```@docs
AbstractElement
AbstractMagnet
AbstractDrift
AbstractCavity
AbstractTransferMap
AbstractTransverseMap
AbstractLongitudinalRFMap
```

## Lattice Assembly

```@docs
Lattice
isperiodic
total_length
get_length
spos
findelem
materialize_lattice
```

## Tracking

### High-Level Interface

```@docs
track
track!
linepass
linepass!
```

### Low-Level Interface

```@docs
pass!
check_lost
```

### Physics Helpers

```@docs
drift6
strthinkick
symplectic4_pass
```

## Linear Optics

### Optics Types

```@docs
AbstractOptics
AbstractOptics2D
AbstractOptics4D
optics2D
optics4DUC
TwissResult
transition_gamma
refine_lattice
```

### Map Computation

```@docs
transfer_map
one_turn_map
fastfindm66
findm66
fastfindm66_refpts
findm66_refpts
```

### Twiss Analysis

```@docs
twiss
periodic_twiss
transport_twiss
twissline
twissring
twissPropagate
periodicEdwardsTengTwiss
gettune
getchrom
```

### Lattice Illustrations

```@docs
LatticeGlyph
lattice_plot_data
plot_lattice!
```

### Closed Orbit

```@docs
find_closed_orbit_6d
find_closed_orbit_4d
```

### Momentum Convention

The optics interface takes and returns relative momentum ``\delta_P``, while
the tracked state stores ``\delta_E``; see [Conventions](@ref conventions).
These exact conversions are what the optics functions apply internally.

```@docs
deltap_from_deltae
deltae_from_deltap
```

## TPSA Maps

PolySeries is a dependency of TrackPad: the TPSA utilities below and the
default `method=:tpsa` of the optics functions are always available.

```@docs
tpsa_map
polyseries_variables
polyseries_one_turn_map
```

## GPU Acceleration

!!! note "Extension packages"
    GPU support requires loading a backend package alongside TrackPad.
    See the [GPU guide](@ref gpu_guide) for usage examples.

```@docs
GPULattice
ParamSweepLattice
gpu_adapt
batch_linepass!
param_sweep_linepass!
cpu_batch_linepass!
batch_jacobian!
batch_hessian_vector_product!
batch_hessian!
```

## [Time Dependence](@id time_dependence)

```@docs
TimeContext
TimeFunction
TimeDependentParam
Time
RealTime
Turn
TimeVaryingElement
timed
materialize
teval
time_lower
static_timecheck
```
