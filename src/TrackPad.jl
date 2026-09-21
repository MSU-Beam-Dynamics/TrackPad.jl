"""
    TrackPad

Symplectic 6-D particle tracking and linear optics for accelerator lattices.

The core is a set of allocation-free element kernels on `SVector{6}`
coordinates `(x, px, y, py, z, δE)` that are generic in the coordinate type, so
one implementation serves scalar `Float64`/`Float32` tracking, truncated
power series maps (PolySeries `CTPS` coordinates, the backend of the default
optics method), automatic differentiation (Enzyme, via `TrackPadEnzymeExt`)
and packed multi-particle tracking on CPU threads or GPUs (KernelAbstractions;
`TrackPadCUDAExt`, `TrackPadMetalExt`).

Start with `Beam`, `Lattice`, `track`/`track!`, and the optics functions
`gettune`, `getchrom`, `twiss` (`periodic_twiss`/`transport_twiss`). Ring optics are
computed from Taylor maps by default (`method=:tpsa`) or from finite
differences of tracking (`method=:fd`). Physics and data conventions are
documented in the manual; the interface speaks the relative momentum
`δP = (P-P0)/P0` while the tracked state stores `δE`.
"""
module TrackPad

# Core dependencies
using LinearAlgebra
using StaticArrays
using Adapt
using PolySeries

# Element definitions (types and constructors)
include("elements.jl")

# Time-dependent parameter utilities
include("time_dependence.jl")

# Tracking physics (pass! functions for each element)
include("tracking.jl")

# Lattice and beam definitions
include("lattice.jl")

# Backend-independent lattice illustration geometry
include("lattice_plot.jl")

# Linear optics utilities
include("optics.jl")

# Matched macroparticle distributions and moment diagnostics
include("distributions.jl")

# TPSA map API and the PolySeries backend (default optics method)
include("tpsa.jl")
include("tpsa_polyseries.jl")

# Lattice I/O: PALS YAML and MAD-X readers/writers
include("io.jl")

# GPU-accelerated tracking (KernelAbstractions - Metal / CUDA / CPU)
include("gpu.jl")

# Re-export key symbols
export AbstractElement, AbstractMagnet, AbstractDrift, AbstractCavity
export AbstractTransferMap, AbstractTransverseMap, AbstractLongitudinalRFMap
export Marker, Patch, Drift, Quadrupole, Sextupole, Octupole
export SBend, RFCavity, ThinMultipole, Solenoid, Corrector
export DriftSC, QuadrupoleSC, SextupoleSC, OctupoleSC, SBendSC
export RBend, RBendSC, ERBend, LBend
export SpaceCharge, Translation, YRotation
export HKicker, VKicker
export Wiggler, CrabCavity, AccelCavity, LongitudinalRFMap
export LorentzBoost, InvLorentzBoost
export StrongThinGaussianBeam, StrongGaussianBeam
export classical_radius, beambeam_amplitude, gaussian_beam_field, faddeeva_w, CLASSICAL_RADIUS_EV_M
export LongitudinalRLCWake, LongitudinalWake, wakefieldfunc_RLCWake, wakefieldfunc
export physical_wake_scale
export TimeContext, TimeFunction, TimeDependentParam, Time, RealTime, Turn
export teval, time_lower, static_timecheck
export TimeVaryingElement, timed, materialize

export Beam, kinetic_energy, p0c, Lattice, isperiodic, refine_lattice
export pass!, track, track!, linepass, linepass!
export total_length, spos, findelem, get_length, materialize_lattice
export LatticeGlyph, lattice_plot_data, plot_lattice!
export AbstractOptics, AbstractOptics2D, AbstractOptics4D, optics2D, optics4DUC
export TwissResult
export transfer_map, one_turn_map, findm66, fastfindm66, findm66_refpts, fastfindm66_refpts
export twissPropagate, periodicEdwardsTengTwiss, twissring
export gettune, getchrom, twiss, periodic_twiss, transport_twiss, twissline
export find_closed_orbit_4d, find_closed_orbit_6d
export gaussian_distribution, matched_gaussian, matched_covariance
export match_moments!, beam_covariance, projected_emittances, eigenemittances
export check_lost, drift6, strthinkick, symplectic4_pass
export tpsa_map

# GPU acceleration
export GPULattice, ParamSweepLattice
export gpu_adapt
export batch_linepass!, param_sweep_linepass!
export cpu_batch_linepass!

# Automatic differentiation (implemented by TrackPadEnzymeExt)
"""
    batch_jacobian!(jacobian, coordinates, gl; nturns=1)

Compute one tracking Jacobian per initial condition using Enzyme.
Requires `using Enzyme`.
"""
function batch_jacobian! end

"""
    batch_hessian_vector_product!(result, coordinates, vectors, gl; kwargs...)

Compute one tracking Hessian-vector product per initial condition.
Requires `using Enzyme`.
"""
function batch_hessian_vector_product! end

"""
    batch_hessian!(hessian, coordinates, gl; kwargs...)

Compute the full tracking Hessian for each initial condition.
Requires `using Enzyme`.
"""
function batch_hessian! end
export batch_jacobian!, batch_hessian_vector_product!, batch_hessian!

# Physical constants
export M_ELECTRON, M_PROTON

# TPSA utilities (implemented in tpsa_polyseries.jl)
"""
    polyseries_variables(T; order=1) -> SVector{6,CTPS{T}}
    polyseries_variables(x0::SVector{6,T}; order=1) -> SVector{6,CTPS{T}}

The six TPSA coordinate variables about the origin, or about the expansion
point `x0`, truncated at `order`. Sets the global PolySeries descriptor to
`(6, order)`.
"""
function polyseries_variables end

"""
    polyseries_one_turn_map(lat, beam; r0=zeros, order=1) -> SVector{6,CTPS}

Track the TPSA variables from [`polyseries_variables`](@ref) once through `lat`.
Equivalent to [`tpsa_map`](@ref) with `closed_orbit = r0`; kept for the
JuTrack-style call signature.
"""
function polyseries_one_turn_map end
export polyseries_variables, polyseries_one_turn_map

end # module TrackPad
