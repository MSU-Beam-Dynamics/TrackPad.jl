using Test
using TrackPad
using CUDA

CUDA.functional(true) || error("CUDA is not functional on this host")
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "verify_gpu.jl"))

function test_device_parity(actual, expected; rtol, atol, label)
    errors = abs.(actual .- expected)
    limits = atol .+ rtol .* abs.(expected)
    max_error, index = findmax(errors)
    @info "$label parity" max_error index expected=expected[index] actual=actual[index]
    @test all(errors .<= limits)
end

function run_cuda_case(::Type{T}) where T
    beam = Beam(T(3e9))
    lat = Lattice(supported_gpu_elements(T, beam.energy))
    coords0 = initial_coordinates(T, 4096)
    backend = CUDA.CUDABackend()

    gl_cpu = GPULattice(lat, beam; dtype=T)
    gl_cuda = gpu_adapt(lat, beam, backend; dtype=T)
    @test gl_cuda.elem_types isa CUDA.CuArray
    @test gl_cuda.fparams isa CUDA.CuArray
    @test gl_cuda.iparams isa CUDA.CuArray

    expected = copy(coords0)
    batch_linepass!(expected, gl_cpu)
    coords_cuda = CUDA.CuArray(coords0)
    CUDA.@sync batch_linepass!(coords_cuda, gl_cuda)
    actual = Array(coords_cuda)
    atol = T === Float32 ? T(2e-6) : T(2e-13)
    rtol = T === Float32 ? T(2e-5) : T(2e-12)
    test_device_parity(actual, expected; rtol=rtol, atol=atol, label="$T linepass")

    expected_ring = copy(coords0)
    batch_ringpass!(expected_ring, gl_cpu, 3)
    coords_cuda = CUDA.CuArray(coords0)
    CUDA.@sync batch_ringpass!(coords_cuda, gl_cuda, 3)
    test_device_parity(Array(coords_cuda), expected_ring;
                       rtol=3rtol, atol=3atol, label="$T ringpass")

    strengths = CUDA.CuArray(collect(T, range(T(0.4), T(0.8); length=128)))
    qlat = Lattice([Quadrupole(T(0.3), T(0.4); num_int_steps=4), Drift(T(0.7))])
    qgl_cuda = gpu_adapt(qlat, beam, backend; dtype=T)
    sweep_cuda = ParamSweepLattice(qgl_cuda, [(1, 2, strengths)])
    input_cpu = initial_coordinates(T, 128)
    input_cuda = CUDA.CuArray(input_cpu)
    output_cuda = CUDA.zeros(T, 128, 6)
    CUDA.@sync param_sweep_linepass!(output_cuda, input_cuda, sweep_cuda)

    qgl_cpu = GPULattice(qlat, beam; dtype=T)
    sweep_cpu = ParamSweepLattice(qgl_cpu, [(1, 2, Array(strengths))])
    output_cpu = similar(input_cpu)
    param_sweep_linepass!(output_cpu, input_cpu, sweep_cpu)
    test_device_parity(Array(output_cuda), output_cpu;
                       rtol=rtol, atol=atol, label="$T parameter sweep")

    uploaded_sweep = gpu_adapt(sweep_cpu, backend)
    @test uploaded_sweep.fparams isa CUDA.CuArray
end

@testset "CUDA A100 parity" begin
    @test CUDA.capability(CUDA.device()) >= v"8.0"
    run_cuda_case(Float32)
    run_cuda_case(Float64)
end
