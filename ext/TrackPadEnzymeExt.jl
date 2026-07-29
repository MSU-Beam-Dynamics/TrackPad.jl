"""
    TrackPadEnzymeExt

Enzyme extension for batched derivatives of GPU-compatible tracking.

This module is loaded automatically when `using Enzyme` is called alongside
TrackPad. The APIs work with CPU and GPU arrays supported by
`batch_linepass!`.
"""
module TrackPadEnzymeExt

using TrackPad
using Enzyme

const PHASE_SPACE_DIMENSION = 6

function _batch_track!(coords, gl, nturns)
    TrackPad.batch_linepass!(coords, gl, nturns)
    return nothing
end

function _validate_coordinates(coords)
    ndims(coords) == 2 || throw(DimensionMismatch("coordinates must be a matrix"))
    size(coords, 2) == PHASE_SPACE_DIMENSION ||
        throw(DimensionMismatch("coordinates must have size N×6"))
    return size(coords, 1)
end

function _validate_jacobian(jacobian, n_particles)
    size(jacobian) == (n_particles, PHASE_SPACE_DIMENSION, PHASE_SPACE_DIMENSION) ||
        throw(DimensionMismatch("jacobian must have size N×6×6"))
    return nothing
end

function _validate_hessian(hessian, n_particles)
    expected = (
        n_particles,
        PHASE_SPACE_DIMENSION,
        PHASE_SPACE_DIMENSION,
        PHASE_SPACE_DIMENSION,
    )
    size(hessian) == expected ||
        throw(DimensionMismatch("hessian must have size N×6×6×6"))
    return nothing
end

function _validate_step(step::T) where T
    isfinite(step) && step > zero(T) ||
        throw(ArgumentError("step must be finite and positive"))
    return step
end

_default_second_order_step(::Type{T}) where T = cbrt(eps(T))

function _validate_second_order_method(method::Symbol)
    method === :enzyme_finite_difference ||
        throw(ArgumentError("method must be :enzyme_finite_difference"))
    return method
end

function _batch_hessian_vector_product!(
    result,
    coordinates,
    vectors,
    gl,
    nturns,
    h,
    plus,
    minus,
    jacobian_plus,
    jacobian_minus,
)
    @. plus = coordinates + h * vectors
    @. minus = coordinates - h * vectors
    TrackPad.batch_jacobian!(jacobian_plus, plus, gl; nturns=nturns)
    TrackPad.batch_jacobian!(jacobian_minus, minus, gl; nturns=nturns)

    scale = inv(2h)
    @. result = (jacobian_plus - jacobian_minus) * scale
    return result
end

"""
    batch_jacobian!(jacobian, coordinates, gl; nturns=1)

Compute the tracking Jacobian for every row of `coordinates`.

`coordinates` has shape `N×6` and is not modified. `jacobian` must have shape
`N×6×6`, with indices `(particle, output_coordinate, input_coordinate)`.
Both arrays must use the same scalar type and backend as `gl`.

The implementation uses six Enzyme forward-mode passes, one for each initial
phase-space coordinate.
"""
function TrackPad.batch_jacobian!(
    jacobian::AbstractArray{T,3},
    coordinates::AbstractMatrix{T},
    gl::TrackPad.GPULattice{T};
    nturns::Integer=1,
) where {T<:AbstractFloat}
    nturns >= 0 || throw(ArgumentError("nturns must be nonnegative"))
    n_particles = _validate_coordinates(coordinates)
    _validate_jacobian(jacobian, n_particles)

    primal = similar(coordinates)
    tangent = similar(coordinates)
    mode = Enzyme.Forward

    for input_coordinate in 1:PHASE_SPACE_DIMENSION
        copyto!(primal, coordinates)
        fill!(tangent, zero(T))
        @views tangent[:, input_coordinate] .= one(T)

        Enzyme.autodiff(
            mode,
            Enzyme.Const(_batch_track!),
            Enzyme.Const,
            Enzyme.Duplicated(primal, tangent),
            Enzyme.Const(gl),
            Enzyme.Const(nturns),
        )

        @views jacobian[:, :, input_coordinate] .= tangent
    end
    return jacobian
end

"""
    batch_hessian_vector_product!(
        result, coordinates, vectors, gl;
        nturns=1, step=cbrt(eps(T)), method=:enzyme_finite_difference,
    )

Compute one Hessian-vector product for every particle.

For the six-output tracking map `F`, `result[p, o, i]` is
`sum(∂²F[o]/∂x[i]∂x[j] * vectors[p, j], j=1:6)`. `result` therefore has
shape `N×6×6`, while `coordinates` and `vectors` have shape `N×6`.

The current implementation uses a centered directional difference of exact
Enzyme Jacobians. This avoids the nested-Enzyme CUDA compiler failure in the
current toolchain while retaining GPU parallelism over particles.
"""
function TrackPad.batch_hessian_vector_product!(
    result::AbstractArray{T,3},
    coordinates::AbstractMatrix{T},
    vectors::AbstractMatrix{T},
    gl::TrackPad.GPULattice{T};
    nturns::Integer=1,
    step::Real=_default_second_order_step(T),
    method::Symbol=:enzyme_finite_difference,
) where {T<:AbstractFloat}
    nturns >= 0 || throw(ArgumentError("nturns must be nonnegative"))
    n_particles = _validate_coordinates(coordinates)
    size(vectors) == size(coordinates) ||
        throw(DimensionMismatch("vectors must have the same N×6 shape as coordinates"))
    _validate_jacobian(result, n_particles)
    h = _validate_step(T(step))
    _validate_second_order_method(method)

    plus = similar(coordinates)
    minus = similar(coordinates)
    jacobian_plus = similar(coordinates, T, (n_particles, PHASE_SPACE_DIMENSION, PHASE_SPACE_DIMENSION))
    jacobian_minus = similar(jacobian_plus)
    return _batch_hessian_vector_product!(
        result,
        coordinates,
        vectors,
        gl,
        nturns,
        h,
        plus,
        minus,
        jacobian_plus,
        jacobian_minus,
    )
end

"""
    batch_hessian!(
        hessian, coordinates, gl;
        nturns=1, step=cbrt(eps(T)), method=:enzyme_finite_difference,
        symmetrize=true,
    )

Compute the Hessian of all six final coordinates for every particle.

`hessian` must have shape `N×6×6×6`, indexed as
`(particle, output_coordinate, input_coordinate_1, input_coordinate_2)`.
The implementation evaluates six Hessian-vector products using Cartesian basis
vectors. By default, the final two axes are symmetrized to suppress
finite-difference roundoff.
"""
function TrackPad.batch_hessian!(
    hessian::AbstractArray{T,4},
    coordinates::AbstractMatrix{T},
    gl::TrackPad.GPULattice{T};
    nturns::Integer=1,
    step::Real=_default_second_order_step(T),
    method::Symbol=:enzyme_finite_difference,
    symmetrize::Bool=true,
) where {T<:AbstractFloat}
    nturns >= 0 || throw(ArgumentError("nturns must be nonnegative"))
    n_particles = _validate_coordinates(coordinates)
    _validate_hessian(hessian, n_particles)
    h = _validate_step(T(step))
    _validate_second_order_method(method)

    direction = similar(coordinates)
    product = similar(coordinates, T, (n_particles, PHASE_SPACE_DIMENSION, PHASE_SPACE_DIMENSION))
    plus = similar(coordinates)
    minus = similar(coordinates)
    jacobian_plus = similar(product)
    jacobian_minus = similar(product)
    for input_coordinate in 1:PHASE_SPACE_DIMENSION
        fill!(direction, zero(T))
        @views direction[:, input_coordinate] .= one(T)
        _batch_hessian_vector_product!(
            product,
            coordinates,
            direction,
            gl,
            nturns,
            h,
            plus,
            minus,
            jacobian_plus,
            jacobian_minus,
        )
        @views hessian[:, :, :, input_coordinate] .= product
    end

    if symmetrize
        for i in 1:PHASE_SPACE_DIMENSION
            for j in (i + 1):PHASE_SPACE_DIMENSION
                hij = @view hessian[:, :, i, j]
                hji = @view hessian[:, :, j, i]
                @. hij = (hij + hji) / T(2)
                hji .= hij
            end
        end
    end
    return hessian
end

end # module TrackPadEnzymeExt
