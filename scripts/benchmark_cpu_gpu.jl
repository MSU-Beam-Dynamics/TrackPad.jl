"""
Reproducible TrackPad CPU/CUDA throughput benchmark.

Compilation and input reset are excluded from device-resident timings. Set:

- `TRACKPAD_BENCH_PARTICLES` (default: 1_000_000)
- `TRACKPAD_BENCH_CELLS` (default: 25, four elements per cell)
- `TRACKPAD_BENCH_SAMPLES` (default: 5)
- `TRACKPAD_BENCH_NGPU` (default: all visible GPUs, capped at 4)
"""

using TrackPad
using CUDA
using Statistics

const N_PARTICLES = parse(Int, get(ENV, "TRACKPAD_BENCH_PARTICLES", "1000000"))
const N_CELLS = parse(Int, get(ENV, "TRACKPAD_BENCH_CELLS", "25"))
const N_SAMPLES = parse(Int, get(ENV, "TRACKPAD_BENCH_SAMPLES", "5"))
const REQUESTED_GPUS = parse(Int, get(ENV, "TRACKPAD_BENCH_NGPU", "4"))

function benchmark_samples(run!, reset!; samples=N_SAMPLES)
    times = Vector{Float64}(undef, samples)
    reset!()
    run!() # Compile and warm up outside the measured samples.
    for sample in 1:samples
        reset!()
        t0 = time_ns()
        run!()
        times[sample] = (time_ns() - t0) / 1e9
    end
    return (; minimum=minimum(times), median=median(times), samples=times)
end

function initial_coordinates(n::Int)
    coords = zeros(Float64, n, 6)
    for i in 1:n
        u = i / n
        coords[i, 1] = 8e-4 * (2u - 1)
        coords[i, 2] = 2e-4 * sin(i)
        coords[i, 3] = 6e-4 * cos(i)
        coords[i, 4] = -1.5e-4 * u
        coords[i, 5] = 3e-4 * (u - 0.5)
        coords[i, 6] = 4e-4 * (0.5 - u)
    end
    return coords
end

function benchmark_lattice(ncells::Int)
    elements = AbstractElement[]
    for _ in 1:ncells
        push!(elements, Quadrupole(0.2, 0.7; num_int_steps=4))
        push!(elements, Drift(0.35))
        push!(elements, Quadrupole(0.2, -0.7; num_int_steps=4))
        push!(elements, Drift(0.35))
    end
    return Lattice(elements)
end

function report(label, result, particle_elements)
    throughput = particle_elements / result.median
    println(label,
            ": median=", round(result.median; sigdigits=6), " s",
            ", min=", round(result.minimum; sigdigits=6), " s",
            ", throughput=", round(throughput / 1e9; sigdigits=5), " Gparticle-elements/s",
            ", samples=", result.samples)
end

CUDA.functional(true) || error("CUDA is not functional")
CUDA.allowscalar(false)

lat = benchmark_lattice(N_CELLS)
beam = Beam(3e9)
initial = initial_coordinates(N_PARTICLES)
n_elements = length(lat)
particle_elements = N_PARTICLES * n_elements

println("TrackPad CPU/CUDA benchmark")
println("particles=$N_PARTICLES elements=$n_elements samples=$N_SAMPLES Float64")
println("Julia threads=", Threads.nthreads())
println("CUDA devices visible=", length(collect(CUDA.devices())))

# CPU object-model tracking using Threads.@threads.
cpu_coords = similar(initial)
cpu_result = benchmark_samples(
    () -> cpu_batch_linepass!(cpu_coords, lat, beam),
    () -> copyto!(cpu_coords, initial),
)
report("CPU threaded element tracking", cpu_result, particle_elements)

# CPU KernelAbstractions path using the same flattened lattice as CUDA.
gl_cpu = GPULattice(lat, beam; dtype=Float64)
ka_cpu_coords = similar(initial)
ka_cpu_result = benchmark_samples(
    () -> batch_linepass!(ka_cpu_coords, gl_cpu),
    () -> copyto!(ka_cpu_coords, initial),
)
report("CPU KernelAbstractions", ka_cpu_result, particle_elements)

# One A100, with coordinates and lattice resident on the device.
device0 = first(collect(CUDA.devices()))
CUDA.device!(device0)
backend = CUDA.CUDABackend()
gl_cuda = gpu_adapt(lat, beam, backend; dtype=Float64)
initial_cuda = CUDA.CuArray(initial)
cuda_coords = similar(initial_cuda)
one_gpu_result = benchmark_samples(
    () -> CUDA.@sync(batch_linepass!(cuda_coords, gl_cuda)),
    () -> begin
        copyto!(cuda_coords, initial_cuda)
        CUDA.synchronize()
    end,
)
report("1 GPU device-resident", one_gpu_result, particle_elements)

# Include host-to-device input and device-to-host output copies, with buffers
# preallocated so allocator and garbage-collection noise is excluded.
host_output = similar(initial)
function one_gpu_with_transfers()
    copyto!(cuda_coords, initial)
    batch_linepass!(cuda_coords, gl_cuda)
    copyto!(host_output, cuda_coords)
    CUDA.synchronize()
end
transfer_result = benchmark_samples(one_gpu_with_transfers, () -> nothing; samples=max(3, N_SAMPLES))
report("1 GPU including transfers", transfer_result, particle_elements)

# Strong scaling: split the same total particle count across visible GPUs.
devices = collect(CUDA.devices())[1:min(REQUESTED_GPUS, length(collect(CUDA.devices())))]
if length(devices) > 1
    ranges = [floor(Int, (i - 1) * N_PARTICLES / length(devices)) + 1:floor(Int, i * N_PARTICLES / length(devices))
              for i in eachindex(devices)]
    states = map(zip(devices, ranges)) do (device, range)
        CUDA.device!(device)
        local_backend = CUDA.CUDABackend()
        local_gl = gpu_adapt(lat, beam, local_backend; dtype=Float64)
        local_initial = CUDA.CuArray(initial[range, :])
        local_coords = similar(local_initial)
        (; device, gl=local_gl, initial=local_initial, coords=local_coords)
    end

    function reset_multi!()
        @sync for state in states
            Threads.@spawn begin
                CUDA.device!(state.device)
                copyto!(state.coords, state.initial)
                CUDA.synchronize()
            end
        end
    end

    function run_multi!()
        @sync for state in states
            Threads.@spawn begin
                CUDA.device!(state.device)
                batch_linepass!(state.coords, state.gl)
            end
        end
    end

    multi_gpu_result = benchmark_samples(run_multi!, reset_multi!; samples=max(3, N_SAMPLES))
    report("$(length(devices)) GPU device-resident", multi_gpu_result, particle_elements)
    println("Speedup vs 64-thread CPU: ", round(cpu_result.median / multi_gpu_result.median; sigdigits=5), "x")
    println("Strong scaling vs 1 GPU: ", round(one_gpu_result.median / multi_gpu_result.median; sigdigits=5), "x")
end

println("1 GPU speedup vs threaded CPU: ", round(cpu_result.median / one_gpu_result.median; sigdigits=5), "x")
println("1 GPU speedup vs KA CPU: ", round(ka_cpu_result.median / one_gpu_result.median; sigdigits=5), "x")
