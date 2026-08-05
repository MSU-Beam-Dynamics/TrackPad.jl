```@meta
CurrentModule = TrackPad
```

# API Reference

## Beam

```@docs
Beam
Beam(energy::T; mass::T, charge::T) where T
beti
M_ELECTRON
M_PROTON
```

## Lattice

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
linepass
linepass!
ringpass
ringpass!
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
TwissLineResult
TransportTwissResult
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
periodic_twiss
transport_twiss
twissline
twissring
twissPropagate
periodicEdwardsTengTwiss
gettune
getchrom
```

### Closed Orbit

```@docs
find_closed_orbit_6d
find_closed_orbit_4d
```

## TPSA Maps

!!! note "Extension package"
    `tpsa_map` is provided by the `TrackPadPolySeriesExt` extension and is
    activated when `PolySeries` is loaded.

```@docs
tpsa_map
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
batch_ringpass!
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
