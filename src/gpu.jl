"""
    gpu.jl

GPU-accelerated particle tracking for TrackPad.jl.

Built on KernelAbstractions.jl for backend-agnostic kernels that run on:
  - Apple Metal  (Float32 only — Apple GPU constraint)
  - NVIDIA CUDA  (Float32 or Float64)
  - CPU          (fallback, multi-threaded via Threads.@threads)

Three packed-tracking use cases are supported:
  1. **Multi-particle tracking**   — `batch_linepass!(coords, gl)`:
     N particles through the same lattice.  One GPU thread per particle.

  2. **Parameter sweep**           — `param_sweep_linepass!(out, r0, ps)`:
     One (or N) particles through N different lattice configurations.
     One GPU thread per configuration.

  3. **Batched derivatives**       — Enzyme extension APIs for Jacobians,
     Hessian-vector products, and Hessians of supported packed tracking.

GPU TPSA is not implemented. TPSA power-series objects carry algebraic state
that cannot be packed into the current float parameter arrays.

## Typical workflow (Metal / Apple GPU)

```julia
using TrackPad, Metal, StaticArrays

lat  = read_madx("ring.madx")
beam = Beam(18e9)

# -- Use case 1: 1 million particles --
gl     = gpu_adapt(lat, beam, MetalBackend())   # Float32 on MtlArray
coords = MtlArray(zeros(Float32, 1_000_000, 6)) # N×6
batch_linepass!(coords, gl)                     # in-place

# -- Use case 3: scan quad strength over 1 000 values --
N       = 1000
k1_vals = MtlArray(Float32.(range(0.5, 0.6; length=N)))
qf_idx  = findfirst(e -> isa(e, Quadrupole) && string(e.name) == "QF", lat.elements)
ps      = ParamSweepLattice(gl, [(qf_idx, 2, k1_vals)])  # slot 2 = k1
out     = MtlArray(zeros(Float32, N, 6))
input   = MtlArray(repeat(zeros(Float32, 1, 6), N, 1))
param_sweep_linepass!(out, input, ps)
```
"""

using KernelAbstractions
using Adapt

# ============================================================
# Element type codes  (Int32 — safe for Metal kernels)
# ============================================================
const GPU_DRIFT  = Int32(1)
const GPU_MARKER = Int32(2)
const GPU_QUAD   = Int32(3)
const GPU_SEXT   = Int32(4)
const GPU_OCT    = Int32(5)
const GPU_SBEND  = Int32(6)
const GPU_RFCAV  = Int32(7)
const GPU_CORR   = Int32(8)
const GPU_SOL    = Int32(9)
const GPU_THIN   = Int32(10)
const GPU_PATCH  = Int32(11)

# ============================================================
# Parameter layout
# ============================================================
#
# fparams : Matrix{T},   shape (N_GPU_FPARAMS, n_elems)  — float params
# iparams : Matrix{Int32}, shape (2, n_elems)             — int params
#
# Float param slots (1-indexed):
#   All:       [1] = L
#   QUAD/SEXT/OCT:
#              [2]=k  [3..6]=pa[1..4]  [7..10]=pb[1..4]
#              (k is added to pb[2] for QUAD, pb[3]/2 for SEXT, pb[4]/6 for OCT)
#   SBEND:
#              [2]=angle  [3]=e1  [4]=e2  [5]=fint1  [6]=fint2  [7]=gap
#              [8]=irho   [9..12]=pa[1..4]  [13..16]=pb[1..4]
#              [17]=fringe_entrance  [18]=fringe_exit
#   RFCAV:
#              [2]=volt  [3]=freq  [4]=lag  [5]=energy  [6]=h  [7]=philag
#              [8]=charge
#   CORR:      [2]=xkick  [3]=ykick
#   SOL:       [2]=ks
#   THIN:      [2..5]=pa[1..4]  [6..9]=pb[1..4]
#   PATCH:     [2]=x_offset  [3]=y_offset  [4]=z_offset
#              [5]=x_pitch   [6]=y_pitch   [7]=tilt  [8]=t_offset
#
# Int param slots:
#   [1] = num_int_steps
#   [2] = (reserved / max_order — not used in kernel; always 3 hardcoded)
#
const N_GPU_FPARAMS = Int32(18)
const N_GPU_IPARAMS = Int32(2)

# ============================================================
# GPULattice  —  flat-array representation of a Lattice
# ============================================================

"""
    GPULattice{T, VI, MF, MI}

Lattice encoded as plain arrays for GPU kernel consumption.

- `T`  : float type (`Float32` for Metal, `Float64` for CUDA/CPU)
- `VI` : integer vector type (`Vector{Int32}`, `MtlArray{Int32}`, …)
- `MF` : float matrix type (`Matrix{T}`, `MtlArray{T,2}`, …)
- `MI` : int matrix type (`Matrix{Int32}`, …)

The source lattice's open/periodic boundary is retained for multi-pass
validation.

Construct with [`gpu_adapt`](@ref) or `GPULattice(lat, beam; dtype=Float32)`.
"""
struct GPULattice{T,
                  VI <: AbstractVector{Int32},
                  MF <: AbstractMatrix{T},
                  MI <: AbstractMatrix{Int32}}
    elem_types :: VI          # (n_elems,)
    fparams    :: MF          # (N_GPU_FPARAMS, n_elems)
    iparams    :: MI          # (N_GPU_IPARAMS, n_elems)
    n_elems    :: Int32
    beti       :: T           # 1/β from Beam
    periodic   :: Bool
end

Adapt.@adapt_structure GPULattice

# ============================================================
# ParamSweepLattice  —  N configurations for parameter scans
# ============================================================

"""
    ParamSweepLattice{T, VI, MF, MI, MV, VE, VV}

Lattice with N distinct parameter configurations.
All configurations share the base `fparams`, `elem_types`, and `iparams`.
Only the selected parameter values are stored per configuration:

- `variation_index[slot, element]` maps a parameter to a variation row, or zero
  when the base value is used.
- `varied_elements[element]` marks elements that need an override lookup.
- `variation_values[configuration, variation]` stores the selected values.

Construct with
```julia
ps = ParamSweepLattice(gl, [(elem_idx, param_slot, values_vector), ...])
```
"""
struct ParamSweepLattice{T,
                         VI  <: AbstractVector{Int32},
                         MF  <: AbstractMatrix{T},
                         MI  <: AbstractMatrix{Int32},
                         MV  <: AbstractMatrix{Int32},
                         VE  <: AbstractVector{Int32},
                         VV  <: AbstractMatrix{T}}
    elem_types :: VI
    fparams    :: MF          # shared base: (N_GPU_FPARAMS, n_elems)
    iparams    :: MI
    variation_index  :: MV    # (N_GPU_FPARAMS, n_elems)
    varied_elements  :: VE    # nonzero when an element has an override
    variation_values :: VV    # (n_configs, n_variations)
    n_elems    :: Int32
    n_configs  :: Int32
    n_variations :: Int32
    beti       :: T
end

Adapt.@adapt_structure ParamSweepLattice

struct _SweepParameterView{MF, MV, VV}
    base::MF
    variation_index::MV
    variation_values::VV
    config::Int32
end

@inline function Base.getindex(params::_SweepParameterView,
                               slot::Integer,
                               element::Integer)
    variation = @inbounds params.variation_index[slot, element]
    if iszero(variation)
        return @inbounds params.base[slot, element]
    end
    return @inbounds params.variation_values[params.config, variation]
end

# ============================================================
# Inline GPU physics helpers
# ============================================================
# All functions are @inline and use scalar arguments only.
# Metal/CUDA compile these to native GPU code via KernelAbstractions.
#
# Convention: (x, px, y, py, z, d), z=s/β0-c*t and d=ΔE/(P0*c).
# The exact Hamiltonian is always used: pz = sqrt(1+2δ/β+δ²−px²−py²)
#
# Lost-particle handling: if pz² ≤ 0 we clamp it to avoid sqrt(neg)
# on CPU backends; the resulting huge NormL drives coordinates to ±Inf
# which is detected as lost by the caller.

@inline function _gpu_drift(x::T, px::T, y::T, py::T, z::T, d::T,
                              L::T, beti::T) where T
    pz2 = one(T) + T(2)*d*beti + d*d - px*px - py*py
    pz2_s = pz2 > zero(T) ? pz2 : zero(T)           # clamp negative
    NormL = L / (sqrt(pz2_s) + T(1e-30))
    return x + NormL*px,
           px,
           y + NormL*py,
           py,
           z - (NormL*(beti + d) - L*beti),
           d
end

# Horner's method for polynomial evaluation (max_order = 3, 4 coeffs)
@inline function _gpu_horner(x::T, y::T,
                               pa0::T, pa1::T, pa2::T, pa3::T,
                               pb0::T, pb1::T, pb2::T, pb3::T) where T
    ReSum = pb3;  ImSum = pa3
    RS = ReSum*x - ImSum*y + pb2;  ImSum = ImSum*x + ReSum*y + pa2;  ReSum = RS
    RS = ReSum*x - ImSum*y + pb1;  ImSum = ImSum*x + ReSum*y + pa1;  ReSum = RS
    RS = ReSum*x - ImSum*y + pb0;  ImSum = ImSum*x + ReSum*y + pa0;  ReSum = RS
    return ReSum, ImSum
end

# Thin straight multipole kick
@inline function _gpu_strkick(px::T, py::T, x::T, y::T,
                                pa0::T, pa1::T, pa2::T, pa3::T,
                                pb0::T, pb1::T, pb2::T, pb3::T,
                                L::T) where T
    RS, IS = _gpu_horner(x, y, pa0, pa1, pa2, pa3, pb0, pb1, pb2, pb3)
    return px - L*RS, py + L*IS
end

# Thin bend kick (includes 1/ρ curvature term)
@inline function _gpu_bndkick(px::T, py::T, z::T, x::T, y::T, d::T,
                                pa0::T, pa1::T, pa2::T, pa3::T,
                                pb0::T, pb1::T, pb2::T, pb3::T,
                                L::T, irho::T, beti::T) where T
    RS, IS = _gpu_horner(x, y, pa0, pa1, pa2, pa3, pb0, pb1, pb2, pb3)
    px_new = px - L*(RS - (d*beti - x*irho)*irho)
    py_new = py + L*IS
    z_new  = z  - L*irho*x*beti
    return px_new, py_new, z_new
end

# Dipole edge fringe (Brown hard-edge model, method=1)
@inline function _gpu_edge(x::T, px::T, y::T, py::T, d::T,
                             irho::T, e::T, fint::T, gap::T) where T
    fc = if fint > zero(T) && gap > zero(T)
        se = sin(e); ce = cos(e)
        irho * gap * fint * (one(T) + se*se) / ce
    else
        zero(T)
    end
    fx = irho * tan(e)
    fy = irho * tan(e - fc / (one(T) + d))
    return x, px + x*fx, y, py - y*fy
end

# 4th-order Yoshida symplectic integrator — straight element
@inline function _gpu_symp4(x::T, px::T, y::T, py::T, z::T, d::T,
                              L::T, beti::T,
                              pa0::T, pa1::T, pa2::T, pa3::T,
                              pb0::T, pb1::T, pb2::T, pb3::T,
                              nsteps::Int32) where T
    SL = L / T(nsteps)
    L1 = SL * T(0.6756035959798286638)
    L2 = SL * T(-0.1756035959798286639)
    K1 = SL * T(1.351207191959657328)
    K2 = SL * T(-1.702414383919314656)
    s = Int32(0)
    while s < nsteps
        s += Int32(1)
        x, px, y, py, z, d = _gpu_drift(x, px, y, py, z, d, L1, beti)
        px, py = _gpu_strkick(px, py, x, y, pa0, pa1, pa2, pa3, pb0, pb1, pb2, pb3, K1)
        x, px, y, py, z, d = _gpu_drift(x, px, y, py, z, d, L2, beti)
        px, py = _gpu_strkick(px, py, x, y, pa0, pa1, pa2, pa3, pb0, pb1, pb2, pb3, K2)
        x, px, y, py, z, d = _gpu_drift(x, px, y, py, z, d, L2, beti)
        px, py = _gpu_strkick(px, py, x, y, pa0, pa1, pa2, pa3, pb0, pb1, pb2, pb3, K1)
        x, px, y, py, z, d = _gpu_drift(x, px, y, py, z, d, L1, beti)
    end
    return x, px, y, py, z, d
end

# 4th-order Yoshida symplectic integrator — bend element (uses bndthinkick)
@inline function _gpu_symp4_bend(x::T, px::T, y::T, py::T, z::T, d::T,
                                   L::T, beti::T,
                                   pa0::T, pa1::T, pa2::T, pa3::T,
                                   pb0::T, pb1::T, pb2::T, pb3::T,
                                   irho::T, nsteps::Int32) where T
    SL = L / T(nsteps)
    L1 = SL * T(0.6756035959798286638)
    L2 = SL * T(-0.1756035959798286639)
    K1 = SL * T(1.351207191959657328)
    K2 = SL * T(-1.702414383919314656)
    s = Int32(0)
    while s < nsteps
        s += Int32(1)
        x, px, y, py, z, d = _gpu_drift(x, px, y, py, z, d, L1, beti)
        px, py, z = _gpu_bndkick(px, py, z, x, y, d, pa0, pa1, pa2, pa3, pb0, pb1, pb2, pb3, K1, irho, beti)
        x, px, y, py, z, d = _gpu_drift(x, px, y, py, z, d, L2, beti)
        px, py, z = _gpu_bndkick(px, py, z, x, y, d, pa0, pa1, pa2, pa3, pb0, pb1, pb2, pb3, K2, irho, beti)
        x, px, y, py, z, d = _gpu_drift(x, px, y, py, z, d, L2, beti)
        px, py, z = _gpu_bndkick(px, py, z, x, y, d, pa0, pa1, pa2, pa3, pb0, pb1, pb2, pb3, K1, irho, beti)
        x, px, y, py, z, d = _gpu_drift(x, px, y, py, z, d, L1, beti)
    end
    return x, px, y, py, z, d
end

# Solenoid body map (linearised transverse dynamics, exact energy conversion)
@inline function _gpu_solenoid(x::T, px::T, y::T, py::T, z::T, d::T,
                                 L::T, ks::T, beti::T) where T
    if ks == zero(T)
        return _gpu_drift(x, px, y, py, z, d, L, beti)
    end
    momentum = sqrt(one(T) + T(2)*d*beti + d*d)
    p_norm = one(T) / momentum
    momentum_jacobian = (beti + d) * p_norm
    xpr = px * p_norm;  ypr = py * p_norm
    H   = ks * p_norm / T(2)
    Sn  = sin(L * H);  Cn = cos(L * H)
    x_n  =  x*Cn*Cn + xpr*Cn*Sn/H + y*Cn*Sn + ypr*Sn*Sn/H
    px_n = (-x*H*Cn*Sn + xpr*Cn*Cn - y*H*Sn*Sn + ypr*Cn*Sn) / p_norm
    y_n  = -x*Cn*Sn - xpr*Sn*Sn/H + y*Cn*Cn + ypr*Cn*Sn/H
    py_n = ( x*H*Sn*Sn - xpr*Cn*Sn - y*Cn*Sn*H + ypr*Cn*Cn) / p_norm
    z_n  = z - momentum_jacobian * L *
           (H*H*(x*x + y*y) + T(2)*H*(xpr*y - ypr*x) + xpr*xpr + ypr*ypr) / T(2)
    return x_n, px_n, y_n, py_n, z_n, d
end

# ============================================================
# Kernel: multi-particle tracking  (Use case 1)
# ============================================================
"""
One GPU thread tracks one particle through all elements of the lattice.

`coords` is a `N×6` matrix (mutated in-place).  Rows correspond to
particles; columns correspond to `[x, px, y, py, z, δ]`.

Lost particles are written back as `NaN` (all 6 coordinates).
"""
@kernel function _gpu_linepass_kernel!(coords,
                                        @Const(etypes),
                                        @Const(fparams),
                                        @Const(iparams),
                                        n_elems::Int32,
                                        beti)
    j = @index(Global, Linear)
    T = eltype(coords)

    x  = @inbounds coords[j, 1]
    px = @inbounds coords[j, 2]
    y  = @inbounds coords[j, 3]
    py = @inbounds coords[j, 4]
    z  = @inbounds coords[j, 5]
    d  = @inbounds coords[j, 6]

    x, px, y, py, z, d = _gpu_track_all(x, px, y, py, z, d,
                                          etypes, fparams, iparams,
                                          n_elems, T(beti))

    @inbounds coords[j, 1] = x
    @inbounds coords[j, 2] = px
    @inbounds coords[j, 3] = y
    @inbounds coords[j, 4] = py
    @inbounds coords[j, 5] = z
    @inbounds coords[j, 6] = d
end

# ============================================================
# Kernel: parameter sweep  (Use case 3)
# ============================================================
"""
One GPU thread processes one lattice configuration.

`in_r`     — `n_configs×6` initial coordinates (replicate if constant)
`out`      — `n_configs×6` output (written in-place)
`fparams` — shared `(N_GPU_FPARAMS, n_elems)` base parameters
`variation_index` — maps `(slot, element)` to a variation row
`varied_elements` — marks elements that require override lookup
`variation_values` — `(n_configs, n_variations)` selected parameter values
"""
@kernel function _gpu_param_sweep_kernel!(out,
                                           @Const(in_r),
                                           @Const(etypes),
                                           @Const(fparams),
                                           @Const(iparams),
                                           @Const(variation_index),
                                           @Const(varied_elements),
                                           @Const(variation_values),
                                           n_elems::Int32,
                                           beti)
    j = @index(Global, Linear)
    T = eltype(out)

    x  = @inbounds in_r[j, 1]
    px = @inbounds in_r[j, 2]
    y  = @inbounds in_r[j, 3]
    py = @inbounds in_r[j, 4]
    z  = @inbounds in_r[j, 5]
    d  = @inbounds in_r[j, 6]

    params = _SweepParameterView(fparams, variation_index, variation_values,
                                 Int32(j))
    x, px, y, py, z, d = _gpu_track_all_sweep(x, px, y, py, z, d,
                                                etypes, params, iparams,
                                                varied_elements,
                                                n_elems, T(beti))

    @inbounds out[j, 1] = x
    @inbounds out[j, 2] = px
    @inbounds out[j, 3] = y
    @inbounds out[j, 4] = py
    @inbounds out[j, 5] = z
    @inbounds out[j, 6] = d
end

# ============================================================
# Core tracking loop (called by both kernels)
# ============================================================

@inline function _gpu_track_all(x::T, px::T, y::T, py::T, z::T, d::T,
                                  etypes, fparams, iparams,
                                  n_elems::Int32, beti::T) where T
    for i in Int32(1):n_elems
        x, px, y, py, z, d = _gpu_apply_elem(x, px, y, py, z, d,
                                               etypes, fparams, iparams,
                                               i, beti)
    end
    return x, px, y, py, z, d
end

@inline function _gpu_track_all_sweep(x::T, px::T, y::T, py::T, z::T, d::T,
                                       etypes, params, iparams, varied_elements,
                                       n_elems::Int32, beti::T) where T
    for i in Int32(1):n_elems
        if iszero(@inbounds varied_elements[i])
            x, px, y, py, z, d = _gpu_apply_elem(
                x, px, y, py, z, d, etypes, params.base, iparams, i, beti)
        else
            x, px, y, py, z, d = _gpu_apply_elem(
                x, px, y, py, z, d, etypes, params, iparams, i, beti)
        end
    end
    return x, px, y, py, z, d
end

# ============================================================
# Element dispatch (shared between both kernels)
# ============================================================

@inline function _gpu_apply_elem(x::T, px::T, y::T, py::T, z::T, d::T,
                                   etypes, fparams, iparams,
                                   i::Int32, beti::T) where T
    etype  = @inbounds etypes[i]
    L      = @inbounds fparams[1, i]
    nsteps = @inbounds iparams[1, i]

    if etype == GPU_DRIFT
        x, px, y, py, z, d = _gpu_drift(x, px, y, py, z, d, L, beti)

    elseif etype == GPU_MARKER
        # no-op

    elseif etype == GPU_PATCH
        x, px, y, py, z, d = _patch_coordinates(
            x,
            px,
            y,
            py,
            z,
            d,
            beti,
            @inbounds(fparams[2, i]),
            @inbounds(fparams[3, i]),
            @inbounds(fparams[4, i]),
            @inbounds(fparams[5, i]),
            @inbounds(fparams[6, i]),
            @inbounds(fparams[7, i]),
            @inbounds(fparams[8, i]),
        )

    elseif etype == GPU_QUAD || etype == GPU_SEXT || etype == GPU_OCT
        k   = @inbounds fparams[2, i]
        pa0 = @inbounds fparams[3, i]; pa1 = @inbounds fparams[4, i]
        pa2 = @inbounds fparams[5, i]; pa3 = @inbounds fparams[6, i]
        pb0 = @inbounds fparams[7, i]
        pb1 = @inbounds fparams[8, i] + (etype == GPU_QUAD ? k       : zero(T))
        pb2 = @inbounds fparams[9, i] + (etype == GPU_SEXT ? k/T(2)  : zero(T))
        pb3 = @inbounds fparams[10,i] + (etype == GPU_OCT  ? k/T(6)  : zero(T))
        x, px, y, py, z, d = _gpu_symp4(x, px, y, py, z, d, L, beti,
                                          pa0, pa1, pa2, pa3,
                                          pb0, pb1, pb2, pb3, nsteps)

    elseif etype == GPU_SBEND
        angle  = @inbounds fparams[2, i]
        e1     = @inbounds fparams[3, i]; e2    = @inbounds fparams[4, i]
        fint1  = @inbounds fparams[5, i]; fint2 = @inbounds fparams[6, i]
        gap    = @inbounds fparams[7, i]
        irho   = @inbounds fparams[8, i]
        pa0    = @inbounds fparams[9, i];  pa1 = @inbounds fparams[10,i]
        pa2    = @inbounds fparams[11,i];  pa3 = @inbounds fparams[12,i]
        pb0    = @inbounds fparams[13,i];  pb1 = @inbounds fparams[14,i]
        pb2    = @inbounds fparams[15,i];  pb3 = @inbounds fparams[16,i]
        fringe_e = @inbounds fparams[17,i]
        fringe_x = @inbounds fparams[18,i]

        if fringe_e != zero(T)
            x, px, y, py = _gpu_edge(x, px, y, py, d, irho, e1, fint1, gap)
        end
        x, px, y, py, z, d = _gpu_symp4_bend(x, px, y, py, z, d, L, beti,
                                               pa0, pa1, pa2, pa3,
                                               pb0, pb1, pb2, pb3,
                                               irho, nsteps)
        if fringe_x != zero(T)
            x, px, y, py = _gpu_edge(x, px, y, py, d, irho, e2, fint2, gap)
        end

    elseif etype == GPU_RFCAV
        volt   = @inbounds fparams[2, i]
        freq   = @inbounds fparams[3, i]
        lag    = @inbounds fparams[4, i]
        energy = @inbounds fparams[5, i]
        philag = @inbounds fparams[7, i]
        charge = @inbounds fparams[8, i]
        if L > zero(T)
            x, px, y, py, z, d = _gpu_drift(x, px, y, py, z, d, L/T(2), beti)
        end
        if energy > zero(T)
            beta  = one(T) / beti
            invgamma = sqrt(max(zero(T), one(T) - beta*beta))
            p0c   = energy * (one(T) + invgamma) / beta
            kick  = charge * volt / p0c
            phase = -T(2*pi) * freq * (z + lag) / T(2.99792458e8) - philag
            d     = d - kick * sin(phase)
        end
        if L > zero(T)
            x, px, y, py, z, d = _gpu_drift(x, px, y, py, z, d, L/T(2), beti)
        end

    elseif etype == GPU_CORR
        xkick  = @inbounds fparams[2, i]
        ykick  = @inbounds fparams[3, i]
        momentum = sqrt(one(T) + T(2)*d*beti + d*d)
        p_norm = one(T) / momentum
        momentum_jacobian = (beti + d) * p_norm
        NormL  = L * p_norm
        z  = z - momentum_jacobian * NormL * p_norm *
                 (xkick*xkick/T(3) + ykick*ykick/T(3) +
                  px*px + py*py + px*xkick + py*ykick) / T(2)
        x  = x  + NormL * (px + xkick/T(2))
        y  = y  + NormL * (py + ykick/T(2))
        px = px + xkick
        py = py + ykick

    elseif etype == GPU_SOL
        ks = @inbounds fparams[2, i]
        x, px, y, py, z, d = _gpu_solenoid(x, px, y, py, z, d, L, ks, beti)

    elseif etype == GPU_THIN
        pa0 = @inbounds fparams[2, i]; pa1 = @inbounds fparams[3, i]
        pa2 = @inbounds fparams[4, i]; pa3 = @inbounds fparams[5, i]
        pb0 = @inbounds fparams[6, i]; pb1 = @inbounds fparams[7, i]
        pb2 = @inbounds fparams[8, i]; pb3 = @inbounds fparams[9, i]
        px, py = _gpu_strkick(px, py, x, y,
                               pa0, pa1, pa2, pa3,
                               pb0, pb1, pb2, pb3, one(T))
    end  # GPULattice construction guarantees a supported element type code.

    return x, px, y, py, z, d
end

# ============================================================
# Parameter extraction: Julia element → (type_code, fvec, ivec)
# ============================================================

function _gpu_elem_type(elem)::Int32
    elem isa Drift         && return GPU_DRIFT
    elem isa Marker        && return GPU_MARKER
    elem isa Quadrupole    && return GPU_QUAD
    elem isa Sextupole     && return GPU_SEXT
    elem isa Octupole      && return GPU_OCT
    elem isa SBend         && return GPU_SBEND
    elem isa RFCavity      && return GPU_RFCAV
    elem isa Corrector     && return GPU_CORR
    elem isa Solenoid      && return GPU_SOL
    elem isa ThinMultipole && return GPU_THIN
    elem isa Patch         && return GPU_PATCH
    throw(ArgumentError("GPU tracking does not support $(typeof(elem))"))
end

function _validate_gpu_element(elem)
    for field in (:t1, :t2, :r1, :r2, :r_apertures, :e_apertures, :kick_angle)
        if hasproperty(elem, field) && !iszero(getproperty(elem, field))
            throw(ArgumentError("GPU tracking does not support nonzero `$field` for $(typeof(elem))"))
        end
    end
    for field in (:rad, :fringe_entrance, :fringe_exit,
                  :fringe_quad_entrance, :fringe_quad_exit)
        if hasproperty(elem, field) && !iszero(getproperty(elem, field))
            throw(ArgumentError("GPU tracking does not support `$field` for $(typeof(elem))"))
        end
    end
    return nothing
end

function _gpu_fparams(::Type{T}, elem) where T
    fp = zeros(T, N_GPU_FPARAMS)
    et = _gpu_elem_type(elem)

    if et == GPU_DRIFT || et == GPU_MARKER
        fp[1] = T(hasproperty(elem, :L) ? elem.L : 0)

    elseif et == GPU_QUAD
        fp[1] = T(elem.L);  fp[2] = elem.max_order >= 1 ? T(elem.k1) : zero(T)
        for k in 1:4
            active = k - 1 <= elem.max_order
            fp[2+k] = active ? T(elem.polynom_a[k]) : zero(T)
            fp[6+k] = active && k != 2 ? T(elem.polynom_b[k]) : zero(T)
        end

    elseif et == GPU_SEXT
        fp[1] = T(elem.L);  fp[2] = elem.max_order >= 2 ? T(elem.k2) : zero(T)
        for k in 1:4
            active = k - 1 <= elem.max_order
            fp[2+k] = active ? T(elem.polynom_a[k]) : zero(T)
            fp[6+k] = active && k != 3 ? T(elem.polynom_b[k]) : zero(T)
        end

    elseif et == GPU_OCT
        fp[1] = T(elem.L);  fp[2] = elem.max_order >= 3 ? T(elem.k3) : zero(T)
        for k in 1:4
            active = k - 1 <= elem.max_order
            fp[2+k] = active ? T(elem.polynom_a[k]) : zero(T)
            fp[6+k] = active && k != 4 ? T(elem.polynom_b[k]) : zero(T)
        end

    elseif et == GPU_SBEND
        L = T(elem.L)
        fp[1]  = L
        fp[2]  = T(elem.angle)
        fp[3]  = T(elem.e1);   fp[4]  = T(elem.e2)
        fp[5]  = T(hasproperty(elem, :fint1) ? elem.fint1 : 0)
        fp[6]  = T(hasproperty(elem, :fint2) ? elem.fint2 : 0)
        fp[7]  = T(hasproperty(elem, :gap)   ? elem.gap   : 0)
        fp[8]  = L > zero(T) ? T(elem.angle) / L : zero(T)  # irho
        for k in 1:4
            active = k - 1 <= elem.max_order
            fp[8+k]  = active ? T(elem.polynom_a[k]) : zero(T)
            fp[12+k] = active ? T(elem.polynom_b[k]) : zero(T)
        end
        fp[17] = T(hasproperty(elem, :fringe_bend_entrance) ? elem.fringe_bend_entrance : 0)
        fp[18] = T(hasproperty(elem, :fringe_bend_exit)     ? elem.fringe_bend_exit     : 0)

    elseif et == GPU_RFCAV
        fp[1] = T(elem.L)
        fp[2] = T(elem.volt);  fp[3] = T(elem.freq)
        fp[4] = T(elem.lag);   fp[5] = T(elem.energy)
        fp[6] = T(hasproperty(elem, :h)      ? elem.h      : 0)
        fp[7] = T(hasproperty(elem, :philag) ? elem.philag : 0)
        fp[8] = T(hasproperty(elem, :charge) ? elem.charge : 1)

    elseif et == GPU_CORR
        fp[1] = T(elem.L)
        fp[2] = T(elem.xkick);  fp[3] = T(elem.ykick)

    elseif et == GPU_SOL
        fp[1] = T(elem.L);  fp[2] = T(elem.ks)

    elseif et == GPU_THIN
        for k in 1:4
            active = k - 1 <= elem.max_order
            fp[1+k] = active ? T(elem.polynom_a[k]) : zero(T)
            fp[5+k] = active ? T(elem.polynom_b[k]) : zero(T)
        end

    elseif et == GPU_PATCH
        fp[2] = T(elem.x_offset)
        fp[3] = T(elem.y_offset)
        fp[4] = T(elem.z_offset)
        fp[5] = T(elem.x_pitch)
        fp[6] = T(elem.y_pitch)
        fp[7] = T(elem.tilt)
        fp[8] = T(elem.t_offset)
    end

    return fp
end

function _gpu_iparams(elem)
    ip = ones(Int32, N_GPU_IPARAMS)  # default: 1 step
    if hasproperty(elem, :num_int_steps)
        ip[1] = Int32(max(1, elem.num_int_steps))
    end
    if hasproperty(elem, :max_order)
        ip[2] = Int32(elem.max_order)
    else
        ip[2] = Int32(3)
    end
    return ip
end

# ============================================================
# GPULattice constructor
# ============================================================

"""
    GPULattice(lat::Lattice, beam::Beam = Beam(1e9); dtype = Float32) -> GPULattice

Convert a CPU `Lattice` to a `GPULattice` using `dtype` for all float
parameters.  The result lives in CPU arrays and can be moved to a GPU
device with `Adapt.adapt(array_type, gl)`.

Metal requires `dtype = Float32`.  CUDA supports `Float32` or `Float64`.
"""
function GPULattice(lat::Lattice, beam::Beam = Beam(1e9); dtype::Type{T} = Float32) where T
    elems = lat.elements
    n     = length(elems)

    etypes_v  = Vector{Int32}(undef, n)
    fparams_m = zeros(T,     N_GPU_FPARAMS, n)
    iparams_m = zeros(Int32, N_GPU_IPARAMS, n)

    for (i, elem) in enumerate(elems)
        elem isa TimeVaryingElement && throw(ArgumentError(
            "GPU tracking does not support time-varying elements",
        ))
        raw = _resolve_for_time(elem, TimeContext(0.0))
        _validate_gpu_element(raw)
        etypes_v[i] = _gpu_elem_type(raw)
        fp = _gpu_fparams(T, raw)
        ip = _gpu_iparams(raw)
        for k in 1:N_GPU_FPARAMS;  fparams_m[k, i] = fp[k];  end
        for k in 1:N_GPU_IPARAMS;  iparams_m[k, i] = ip[k];  end
    end

    return GPULattice(
        etypes_v, fparams_m, iparams_m, Int32(n), T(beti(beam)), lat.periodic,
    )
end

# ============================================================
# ParamSweepLattice constructor
# ============================================================

"""
    ParamSweepLattice(gl::GPULattice, variations; mode=:aligned)

Build a parameter sweep from a base `GPULattice` by varying selected
float parameter slots.

`variations` is a vector of `(elem_idx, param_slot, values)` tuples where
`values` is a vector/array of parameter values.

- `mode=:aligned` associates values with the same index and requires equal
  lengths. This supports Monte Carlo trials and correlated scans.
- `mode=:cartesian` forms the Cartesian product of the variation vectors.

The base `18 × M` parameter table is shared. Only an `N × K` matrix of varied
values, an `18 × M` integer lookup, and an `M`-entry element mask are allocated,
where `K` is the number of varied parameters and `N` is the number of
configurations.

## Float parameter slots for common elements

| Element  | Slot | Meaning              |
|----------|------|----------------------|
| Quad     |  1   | `L`                  |
| Quad     |  2   | `k1`                 |
| Sext     |  2   | `k2`                 |
| SBend    |  2   | `angle`              |
| SBend    |  8   | `irho` (= angle/L)   |
| RFCavity |  2   | `volt` (V)           |
| RFCavity |  3   | `freq` (Hz)          |

## Example

```julia
# Scan quad QF over 1000 k1 values.
ps = ParamSweepLattice(gl, [(qf_idx, 2, k1_range)])

# Scan every QF/QD combination.
ps = ParamSweepLattice(
    gl,
    [(qf_idx, 2, qf_values), (qd_idx, 2, qd_values)];
    mode=:cartesian,
)
```
"""
function ParamSweepLattice(gl::GPULattice{T}, variations;
                           mode::Symbol=:aligned) where T
    mode in (:aligned, :cartesian) ||
        throw(ArgumentError("mode must be :aligned or :cartesian"))
    isempty(variations) &&
        throw(ArgumentError("at least one parameter variation is required"))

    n_elements = Int(gl.n_elems)
    n_variations = length(variations)
    n_variations <= typemax(Int32) ||
        throw(ArgumentError("too many parameter variations"))

    value_vectors = Vector{Vector{T}}(undef, n_variations)
    lengths = Vector{Int}(undef, n_variations)
    variation_index_cpu = zeros(Int32, Int(N_GPU_FPARAMS), n_elements)
    varied_elements_cpu = zeros(Int32, n_elements)

    for (variation, entry) in enumerate(variations)
        length(entry) == 3 ||
            throw(ArgumentError("each variation must be (element, slot, values)"))
        element, slot, values = entry
        element isa Integer ||
            throw(ArgumentError("variation element index must be an integer"))
        slot isa Integer ||
            throw(ArgumentError("variation parameter slot must be an integer"))
        1 <= element <= n_elements ||
            throw(BoundsError(gl.elem_types, element))
        1 <= slot <= N_GPU_FPARAMS ||
            throw(BoundsError(gl.fparams, (slot, element)))
        iszero(variation_index_cpu[slot, element]) ||
            throw(ArgumentError("parameter slot $slot of element $element is varied more than once"))

        vals = vec(T.(Array(values)))
        isempty(vals) &&
            throw(ArgumentError("parameter variation values must be nonempty"))
        value_vectors[variation] = vals
        lengths[variation] = length(vals)
        variation_index_cpu[slot, element] = Int32(variation)
        varied_elements_cpu[element] = one(Int32)
    end

    if mode === :aligned
        all(==(first(lengths)), lengths) ||
            throw(DimensionMismatch("all aligned variation vectors must have the same length"))
        n_configs = first(lengths)
        variation_values_cpu = Matrix{T}(undef, n_configs, n_variations)
        for variation in 1:n_variations
            variation_values_cpu[:, variation] .= value_vectors[variation]
        end
    else
        n_configs = prod(lengths; init=1)
        n_configs > 0 ||
            throw(ArgumentError("Cartesian parameter grid is empty"))
        variation_values_cpu = Matrix{T}(undef, n_configs, n_variations)
        for (config, index) in enumerate(CartesianIndices(Tuple(lengths)))
            for variation in 1:n_variations
                variation_values_cpu[config, variation] =
                    value_vectors[variation][index[variation]]
            end
        end
    end
    n_configs <= typemax(Int32) ||
        throw(ArgumentError("number of parameter configurations exceeds Int32 capacity"))

    backend = KernelAbstractions.get_backend(gl.fparams)
    variation_index = KernelAbstractions.allocate(
        backend, Int32, size(variation_index_cpu))
    varied_elements = KernelAbstractions.allocate(
        backend, Int32, size(varied_elements_cpu))
    variation_values = KernelAbstractions.allocate(
        backend, T, size(variation_values_cpu))
    copyto!(variation_index, variation_index_cpu)
    copyto!(varied_elements, varied_elements_cpu)
    copyto!(variation_values, variation_values_cpu)

    return ParamSweepLattice(
        gl.elem_types,
        gl.fparams,
        gl.iparams,
        variation_index,
        varied_elements,
        variation_values,
        gl.n_elems,
        Int32(n_configs),
        Int32(n_variations),
        gl.beti,
    )
end

# ============================================================
# gpu_adapt  —  builds CPUGPULattice; extensions override for GPU devices
# ============================================================

"""
    gpu_adapt(lat::Lattice, beam::Beam = Beam(1e9);
              dtype = Float32) -> GPULattice (CPU arrays)

Build a `GPULattice` on the CPU.  Use `batch_linepass!` to track on CPU
(multi-threaded), or let the Metal/CUDA extensions provide a device-side
version.

To get a device-ready lattice:
```julia
# Metal (Float32)
using TrackPad, Metal
gl_metal = gpu_adapt(lat, beam, MetalBackend())

# CUDA (Float64)
using TrackPad, CUDA
gl_cuda  = gpu_adapt(lat, beam, CUDABackend(); dtype=Float64)
```
"""
function gpu_adapt(lat::Lattice, beam::Beam = Beam(1e9);
                   dtype::Type{T} = Float32) where T
    return GPULattice(lat, beam; dtype = dtype)
end

# Overload with explicit backend (extensions may specialise this)
function gpu_adapt(lat::Lattice, beam::Beam, backend;
                   dtype::Type{T} = Float32) where T
    return GPULattice(lat, beam; dtype = dtype)
end


# ============================================================
# batch_linepass!  (Use case 1)
# ============================================================

_batch_workgroup(backend) =
    backend isa KernelAbstractions.CPU ? 32 : 256

"""
    batch_linepass!(coords::AbstractMatrix, gl::GPULattice,
                    nturns::Integer = 1) -> coords

Track `N` particles through the lattice for `nturns` turns.

`coords` is an `N×6` matrix **on the same device** as `gl`.
It is modified **in-place** and returned.

- On Metal: `coords` should be `MtlArray{Float32, 2}`.
- On CUDA:  `coords` should be `CuArray{Float64, 2}` or `CuArray{Float32, 2}`.
- On CPU:   `coords` can be any `Matrix{T}`.

Lost particles are written back as `NaN` (all 6 coordinates).
"""
function batch_linepass!(coords::AbstractMatrix{T},
                          gl::GPULattice{T},
                          nturns::Integer = 1) where T
    nturns >= 0 || throw(ArgumentError("nturns must be nonnegative"))
    nturns <= 1 || gl.periodic || throw(ArgumentError(
        "repeated batch tracking requires a periodic GPULattice",
    ))
    n_particles = Int32(size(coords, 1))
    backend = get_backend(coords)
    kernel = _gpu_linepass_kernel!(backend, _batch_workgroup(backend))
    for _ in 1:nturns
        kernel(coords, gl.elem_types, gl.fparams, gl.iparams,
               gl.n_elems, gl.beti;
               ndrange = n_particles)
        KernelAbstractions.synchronize(backend)
    end
    return coords
end

"""
    batch_ringpass!(coords, gl, nturns) -> coords

Track a particle batch for `nturns` through a ring encoded as `gl`.
This is the explicit multi-turn counterpart of [`batch_linepass!`](@ref).
"""
function batch_ringpass!(coords::AbstractMatrix{T},
                         gl::GPULattice{T},
                         nturns::Integer) where T
    gl.periodic || throw(ArgumentError(
        "batch_ringpass! requires a periodic GPULattice",
    ))
    return batch_linepass!(coords, gl, nturns)
end

# ============================================================
# param_sweep_linepass!  (Use case 3)
# ============================================================

"""
    param_sweep_linepass!(out::AbstractMatrix, in_r::AbstractMatrix,
                           ps::ParamSweepLattice) -> out

Track particles through N different lattice configurations simultaneously.

- `in_r`  — `N_configs × 6` initial coordinates (same device as `ps`)
- `out`   — `N_configs × 6` output (modified in-place)
- `ps`    — `ParamSweepLattice` with N configurations

If all configurations start from the same particle, replicate:
```julia
in_r = repeat(r0', n_configs, 1)   # or use MtlArray(repeat(...))
```
"""
function param_sweep_linepass!(out::AbstractMatrix{T},
                                 in_r::AbstractMatrix{T},
                                 ps::ParamSweepLattice{T}) where T
    n_configs = Int(ps.n_configs)
    size(in_r) == (n_configs, 6) ||
        throw(DimensionMismatch("in_r must have size N_configs×6"))
    size(out) == (n_configs, 6) ||
        throw(DimensionMismatch("out must have size N_configs×6"))
    backend   = get_backend(out)
    kernel    = _gpu_param_sweep_kernel!(backend, 256)
    kernel(out, in_r, ps.elem_types, ps.fparams, ps.iparams,
           ps.variation_index, ps.varied_elements, ps.variation_values,
           ps.n_elems, ps.beti;
           ndrange = n_configs)
    KernelAbstractions.synchronize(backend)
    return out
end

# ============================================================
# CPU multi-particle fallback  (no GPU needed)
# ============================================================

"""
    cpu_batch_linepass!(coords::Matrix{T}, lat::Lattice,
                        beam::Beam, nturns::Integer = 1) -> coords

CPU multi-threaded version of `batch_linepass!`.  Uses `Threads.@threads`
to parallelise over particles.  No GPU required.
"""
function cpu_batch_linepass!(coords::Matrix{T}, lat::Lattice,
                              beam::Beam = Beam(1e9),
                              nturns::Integer = 1) where T
    nturns >= 0 || throw(ArgumentError("nturns must be nonnegative"))
    nturns <= 1 || lat.periodic || throw(ArgumentError(
        "repeated CPU batch tracking requires a periodic lattice",
    ))
    n  = size(coords, 1)
    β_inv = T(beti(beam))
    for _ in 1:nturns
        Threads.@threads for i in 1:n
            r = SVector{6, T}(coords[i,1], coords[i,2], coords[i,3],
                               coords[i,4], coords[i,5], coords[i,6])
            ctx = TimeContext(zero(T))
            for elem in lat.elements
                elem_now = _resolve_for_time(elem, ctx)
                r = pass!(elem_now, r, β_inv)
                if check_lost(r)
                    r = SVector{6,T}(T(NaN), T(NaN), T(NaN), T(NaN), T(NaN), T(NaN))
                    break
                end
            end
            coords[i,1] = r[1]; coords[i,2] = r[2]
            coords[i,3] = r[3]; coords[i,4] = r[4]
            coords[i,5] = r[5]; coords[i,6] = r[6]
        end
    end
    return coords
end
