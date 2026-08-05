using Test
using TrackPad
using CUDA
using Enzyme

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
    lat = Lattice(supported_gpu_elements(T, beam.energy); periodic=true)
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
    drift_lengths = CUDA.CuArray(collect(T, range(T(0.6), T(0.8); length=128)))
    qlat = Lattice([Quadrupole(T(0.3), T(0.4); num_int_steps=4), Drift(T(0.7))])
    qgl_cuda = gpu_adapt(qlat, beam, backend; dtype=T)
    sweep_cuda = ParamSweepLattice(qgl_cuda, [
        (1, 2, strengths),
        (2, 1, drift_lengths),
    ])
    @test sweep_cuda.fparams === qgl_cuda.fparams
    @test sweep_cuda.variation_index isa CUDA.CuArray
    @test sweep_cuda.varied_elements isa CUDA.CuArray
    @test sweep_cuda.variation_values isa CUDA.CuArray
    @test size(sweep_cuda.variation_values) == (128, 2)
    input_cpu = initial_coordinates(T, 128)
    input_cuda = CUDA.CuArray(input_cpu)
    output_cuda = CUDA.zeros(T, 128, 6)
    CUDA.@sync param_sweep_linepass!(output_cuda, input_cuda, sweep_cuda)

    qgl_cpu = GPULattice(qlat, beam; dtype=T)
    sweep_cpu = ParamSweepLattice(qgl_cpu, [
        (1, 2, Array(strengths)),
        (2, 1, Array(drift_lengths)),
    ])
    output_cpu = similar(input_cpu)
    param_sweep_linepass!(output_cpu, input_cpu, sweep_cpu)
    test_device_parity(Array(output_cuda), output_cpu;
                       rtol=rtol, atol=atol, label="$T parameter sweep")

    grid_strengths = T[0.4, 0.6, 0.8]
    grid_lengths = T[0.5, 0.7]
    grid_cuda = ParamSweepLattice(
        qgl_cuda,
        [(1, 2, grid_strengths), (2, 1, grid_lengths)];
        mode=:cartesian,
    )
    grid_cpu = ParamSweepLattice(
        qgl_cpu,
        [(1, 2, grid_strengths), (2, 1, grid_lengths)];
        mode=:cartesian,
    )
    grid_input_cpu = initial_coordinates(T, 6)
    grid_input_cuda = CUDA.CuArray(grid_input_cpu)
    grid_output_cuda = CUDA.zeros(T, 6, 6)
    grid_output_cpu = similar(grid_input_cpu)
    CUDA.@sync param_sweep_linepass!(grid_output_cuda, grid_input_cuda, grid_cuda)
    param_sweep_linepass!(grid_output_cpu, grid_input_cpu, grid_cpu)
    test_device_parity(Array(grid_output_cuda), grid_output_cpu;
                       rtol=rtol, atol=atol, label="$T Cartesian parameter sweep")

    uploaded_sweep = gpu_adapt(sweep_cpu, backend)
    @test uploaded_sweep.fparams isa CUDA.CuArray
    @test uploaded_sweep.variation_index isa CUDA.CuArray
    @test uploaded_sweep.varied_elements isa CUDA.CuArray
    @test uploaded_sweep.variation_values isa CUDA.CuArray
end

@testset "CUDA A100 parity" begin
    @test CUDA.capability(CUDA.device()) >= v"8.0"
    run_cuda_case(Float32)
    run_cuda_case(Float64)
end

@testset "CUDA A100 Enzyme derivatives" begin
    T = Float64
    n = 64
    beam = Beam(T(3e9))
    lat = Lattice([
        Drift(T(0.4)),
        Sextupole(T(0.2), T(1.1); num_int_steps=4),
        Quadrupole(T(0.3), T(0.7); num_int_steps=4),
        Drift(T(0.6)),
    ])
    coordinates_cpu = initial_coordinates(T, n)
    coordinates_cuda = CUDA.CuArray(coordinates_cpu)
    gl_cpu = GPULattice(lat, beam; dtype=T)
    gl_cuda = gpu_adapt(lat, beam, CUDA.CUDABackend(); dtype=T)

    jacobian_cpu = zeros(T, n, 6, 6)
    jacobian_cuda = CUDA.zeros(T, n, 6, 6)
    batch_jacobian!(jacobian_cpu, coordinates_cpu, gl_cpu)
    CUDA.@sync batch_jacobian!(jacobian_cuda, coordinates_cuda, gl_cuda)
    test_device_parity(
        Array(jacobian_cuda),
        jacobian_cpu;
        rtol=T(3e-12),
        atol=T(3e-13),
        label="Float64 Enzyme Jacobian",
    )

    vectors_cpu = initial_coordinates(T, n)
    vectors_cpu .*= T(500)
    vectors_cuda = CUDA.CuArray(vectors_cpu)
    product_cpu = zeros(T, n, 6, 6)
    product_cuda = CUDA.zeros(T, n, 6, 6)
    derivative_step = T(1e-5)
    batch_hessian_vector_product!(
        product_cpu,
        coordinates_cpu,
        vectors_cpu,
        gl_cpu;
        step=derivative_step,
    )
    CUDA.@sync batch_hessian_vector_product!(
        product_cuda,
        coordinates_cuda,
        vectors_cuda,
        gl_cuda;
        step=derivative_step,
    )
    test_device_parity(
        Array(product_cuda),
        product_cpu;
        rtol=T(2e-8),
        atol=T(2e-9),
        label="Float64 Enzyme Hessian-vector product",
    )

    hessian_cpu = zeros(T, n, 6, 6, 6)
    hessian_cuda = CUDA.zeros(T, n, 6, 6, 6)
    batch_hessian!(hessian_cpu, coordinates_cpu, gl_cpu; step=derivative_step)
    CUDA.@sync batch_hessian!(
        hessian_cuda,
        coordinates_cuda,
        gl_cuda;
        step=derivative_step,
    )
    hessian_actual = Array(hessian_cuda)
    test_device_parity(
        hessian_actual,
        hessian_cpu;
        rtol=T(2e-8),
        atol=T(2e-9),
        label="Float64 Enzyme Hessian",
    )
    @test hessian_actual == permutedims(hessian_actual, (1, 2, 4, 3))
end
