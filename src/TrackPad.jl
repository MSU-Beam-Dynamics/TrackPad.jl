"""
    TrackPad.jl

High-performance particle accelerator tracking engine.

TrackPad is designed for:
- Automatic Differentiation (AD) support via parametric types
- GPU acceleration via KernelAbstractions.jl
- Allocation-free tracking kernels using StaticArrays
- TPSA (Truncated Power Series Algebra) for high-order maps

Migrated from JuTrack.jl with enhanced architecture for modern computing.
"""
module TrackPad

# Core dependencies
using LinearAlgebra
using StaticArrays
using Adapt

# Element definitions (types and constructors)
include("elements.jl")

# Time-dependent parameter utilities
include("time_dependence.jl")

# Tracking physics (pass! functions for each element)
include("tracking.jl")

# Lattice and beam definitions
include("lattice.jl")

# Linear optics utilities
include("optics.jl")

# Re-export key symbols
export AbstractElement, AbstractMagnet, AbstractDrift, AbstractCavity
export AbstractTransferMap, AbstractTransverseMap, AbstractLongitudinalRFMap
export Marker, Drift, Quadrupole, Sextupole, Octupole
export SBend, RFCavity, ThinMultipole, Solenoid, Corrector
export DriftSC, QuadrupoleSC, SextupoleSC, OctupoleSC, SBendSC
export RBend, RBendSC, ERBend, LBend
export SpaceCharge, Translation, YRotation
export HKicker, VKicker
export Wiggler, CrabCavity, AccelCavity, LongitudinalRFMap
export LorentzBoost, InvLorentzBoost
export StrongThinGaussianBeam, StrongGaussianBeam
export LongitudinalRLCWake, LongitudinalWake, wakefieldfunc_RLCWake, wakefieldfunc
export TimeContext, TimeFunction, TimeDependentParam, Time, RealTime, Turn
export teval, time_lower, static_timecheck
export TimeVaryingElement, timed, materialize

export Beam, Lattice
export pass!, linepass, linepass!, ringpass, ringpass!
export total_length, spos, findelem, get_length, materialize_lattice
export AbstractOptics, AbstractOptics2D, AbstractOptics4D, optics2D, optics4DUC
export TwissLineResult, one_turn_map, findm66, fastfindm66, findm66_refpts, fastfindm66_refpts
export twissPropagate, periodicEdwardsTengTwiss, twissring
export gettune, getchrom, twissline
export find_closed_orbit_4d, find_closed_orbit_6d
export check_lost, drift6, strthinkick, symplectic4_pass

# Physical constants
export M_ELECTRON, M_PROTON

# Stubs for TPSA extension (TrackPadPolySeriesExt)
function polyseries_variables end
function polyseries_one_turn_map end
export polyseries_variables, polyseries_one_turn_map

end # module TrackPad
