"""
    scripts/test_gpu.jl

Quick end-to-end test of the GPU tracking infrastructure:
  1. CPU batch via KernelAbstractions CPU backend
  2. Metal GPU Float32 (requires `using Metal`)
  3. Parameter sweep (CPU, then Metal)
  4. cpu_batch_linepass! fallback

Run with:
  julia --project=. scripts/test_gpu.jl
  julia --project=. -t 4 scripts/test_gpu.jl   # multi-threaded CPU
"""

using TrackPad, StaticArrays

# ──────────────────────────────────────────────────────────────
# 1. Build a simple FODO lattice
# ──────────────────────────────────────────────────────────────
println("=== Building lattice ===")
lat = Lattice([
    Quadrupole(0.5, 1.2; name=:QF, num_int_steps=4),
    Drift(1.0;           name=:D1),
    Quadrupole(0.5, -1.2; name=:QD, num_int_steps=4),
    Drift(1.0;           name=:D2),
])
beam = Beam(1.0e9)
println("  Elements: ", length(lat.elements))

N = 1000
xs = Float64.(range(-1e-3, 1e-3; length=N))

# ──────────────────────────────────────────────────────────────
# 2. Reference: single-particle CPU linepass
# ──────────────────────────────────────────────────────────────
println("\n=== CPU single-particle reference ===")
ref_coords = zeros(Float64, N, 6)
ref_coords[:, 1] .= xs
for i in 1:N
    r = SVector{6,Float64}(ref_coords[i,:]...)
    r = linepass(lat, r, beam)
    ref_coords[i, :] .= r
end
println("  Done.  Sample: x[1]=$(ref_coords[1,1]),  x[end]=$(ref_coords[end,1])")

# ──────────────────────────────────────────────────────────────
# 3. GPULattice (CPU backend via KernelAbstractions)
# ──────────────────────────────────────────────────────────────
println("\n=== GPULattice / batch_linepass! (CPU backend) ===")
gl   = GPULattice(lat, beam; dtype=Float64)
c64  = zeros(Float64, N, 6);  c64[:, 1] .= xs
batch_linepass!(c64, gl)
err = maximum(abs.(c64 .- ref_coords))
println("  Max diff vs reference: $err")
@assert err < 1e-14 "batch_linepass! (CPU) disagrees with reference"

# ──────────────────────────────────────────────────────────────
# 4. cpu_batch_linepass! fallback (Threads.@threads)
# ──────────────────────────────────────────────────────────────
println("\n=== cpu_batch_linepass! ($(Threads.nthreads()) threads) ===")
ct = zeros(Float64, N, 6);  ct[:, 1] .= xs
cpu_batch_linepass!(ct, lat, beam)
err_t = maximum(abs.(ct .- ref_coords))
println("  Max diff vs reference: $err_t")
@assert err_t < 1e-14 "cpu_batch_linepass! disagrees with reference"

# ──────────────────────────────────────────────────────────────
# 5. Parameter sweep (CPU backend)
# ──────────────────────────────────────────────────────────────
println("\n=== ParamSweepLattice / param_sweep_linepass! (CPU) ===")
qf_idx = 1  # QF is element 1
k1_range = collect(Float64, range(0.8, 1.6; length=N))
ps = ParamSweepLattice(gl, [(qf_idx, 2, k1_range)])
println("  ParamSweepLattice: ", ps.n_configs, " configs")

r0   = zeros(Float64, N, 6);  r0[:, 1] .= 1e-3  # all particles start at x=1mm
pout = zeros(Float64, N, 6)
param_sweep_linepass!(pout, r0, ps)
println("  Done.  x range after sweep: ", extrema(pout[:, 1]))
@assert !any(isnan, pout) "param_sweep: NaN detected"

# ──────────────────────────────────────────────────────────────
# 6. Metal GPU (only if Metal is available)
# ──────────────────────────────────────────────────────────────
try
    using Metal

    println("\n=== Metal GPU (Float32) ===")
    gl_metal = gpu_adapt(lat, beam, MetalBackend())
    println("  fparams type: ", typeof(gl_metal.fparams))

    cf32 = MtlArray(Float32.(zeros(N, 6)))
    xs32 = Float32.(xs)
    cf32_cpu = Array(cf32);  cf32_cpu[:, 1] .= xs32;  copyto!(cf32, cf32_cpu)

    batch_linepass!(cf32, gl_metal)
    Metal.synchronize()
    result = Array(cf32)
    err32 = maximum(abs.(result .- Float32.(ref_coords)))
    println("  Max diff Metal vs CPU reference: $err32  (Float32 roundoff expected)")
    @assert err32 < 1e-5 "Metal result deviates too much from CPU reference"

    # 6b. Parameter sweep on Metal
    println("\n=== ParamSweepLattice on Metal ===")
    k1_range32 = MtlArray(Float32.(k1_range))
    ps_metal = ParamSweepLattice(gl_metal, [(qf_idx, 2, k1_range32)])
    r0_metal = MtlArray(Float32.(r0))
    out_metal = MtlArray(zeros(Float32, N, 6))
    param_sweep_linepass!(out_metal, r0_metal, ps_metal)
    Metal.synchronize()
    pout_metal = Array(out_metal)
    err_ps = maximum(abs.(pout_metal .- Float32.(pout)))
    println("  Param sweep Metal vs CPU max diff: $err_ps")
    @assert err_ps < 1e-4 "Metal param sweep deviates from CPU"

    println("\nAll Metal tests PASSED ✓")
catch e
    if e isa ArgumentError && occursin("Metal", string(e))
        println("\n  [skipped — Metal.jl not available in this environment]")
    else
        rethrow(e)
    end
end

println("\n=== ALL TESTS PASSED ===")
