"""
    distributions.jl

Matched Gaussian macroparticle generation and second-moment diagnostics.
Particle matrices use TrackPad's row-major ensemble layout: one particle per
row and canonical coordinates in the columns.
"""

using Random

export gaussian_distribution, matched_gaussian, matched_covariance
export match_moments!, beam_covariance
export projected_emittances, eigenemittances

@inline function _distribution_dimension(matrix::AbstractMatrix)
    size(matrix, 1) == size(matrix, 2) ||
        throw(DimensionMismatch("covariance matrix must be square"))
    dimension = size(matrix, 1)
    dimension in (4, 6) ||
        throw(DimensionMismatch("covariance matrix must be 4x4 or 6x6"))
    return dimension
end

@inline _numeric_type(value::Number) = typeof(value)
@inline _numeric_type(value::AbstractArray) = eltype(value)

function _distribution_type(values...)
    types = Type[]
    for value in values
        isnothing(value) || push!(types, _numeric_type(value))
    end
    isempty(types) && return Float64
    return float(promote_type(types...))
end

function _moment_tolerances(::Type{T}, scale, atol, rtol) where T
    absolute = isnothing(atol) ? zero(T) : T(atol)
    relative = isnothing(rtol) ? T(100) * eps(T) : T(rtol)
    absolute >= zero(T) || throw(ArgumentError("atol must be nonnegative"))
    relative >= zero(T) || throw(ArgumentError("rtol must be nonnegative"))
    return absolute + relative * max(T(scale), eps(T))
end

function _validated_covariance(sigma::AbstractMatrix, ::Type{T};
                               atol=nothing, rtol=nothing,
                               supported_dimension::Bool=true) where T
    dimension = supported_dimension ? _distribution_dimension(sigma) : size(sigma, 1)
    size(sigma, 1) == size(sigma, 2) ||
        throw(DimensionMismatch("covariance matrix must be square"))
    matrix = Matrix{T}(sigma)
    all(isfinite, matrix) || throw(ArgumentError("covariance matrix must be finite"))
    scale = maximum(abs, matrix)
    tolerance = _moment_tolerances(T, scale, atol, rtol)
    norm(matrix - transpose(matrix), Inf) <= tolerance ||
        throw(ArgumentError("covariance matrix must be symmetric"))
    return dimension, Matrix(Symmetric((matrix + transpose(matrix)) / T(2)))
end

function _psd_sqrt(sigma::AbstractMatrix{T}; atol=nothing, rtol=nothing) where T
    factorization = eigen(Symmetric(sigma))
    scale = maximum(abs, factorization.values)
    tolerance = _moment_tolerances(T, scale, atol, rtol)
    minimum(factorization.values) >= -tolerance ||
        throw(ArgumentError("covariance matrix must be positive semidefinite"))
    values = max.(factorization.values, zero(T))
    root = factorization.vectors * Diagonal(sqrt.(values)) * transpose(factorization.vectors)
    return Matrix(Symmetric((root + transpose(root)) / T(2)))
end

function _positive_inverse_sqrt(matrix::AbstractMatrix{T};
                                atol=nothing, rtol=nothing) where T
    factorization = eigen(Symmetric(matrix))
    scale = maximum(abs, factorization.values)
    tolerance = _moment_tolerances(T, scale, atol, rtol)
    minimum(factorization.values) > tolerance || throw(ArgumentError(
        "sample covariance is rank deficient; use more particles or a different random sample",
    ))
    return factorization.vectors * Diagonal(inv.(sqrt.(factorization.values))) *
           transpose(factorization.vectors)
end

function _column_centroid(coordinates::AbstractMatrix{T}) where T
    particles, dimension = size(coordinates)
    center = zeros(T, dimension)
    particles == 0 && return center
    @inbounds for particle in 1:particles, coordinate in 1:dimension
        center[coordinate] += coordinates[particle, coordinate]
    end
    center ./= T(particles)
    return center
end

function _target_centroid(centroid, dimension::Int, ::Type{T}) where T
    if isnothing(centroid)
        return zeros(T, dimension)
    end
    length(centroid) == dimension ||
        throw(DimensionMismatch("centroid must have length $dimension"))
    center = T.(collect(centroid))
    all(isfinite, center) || throw(ArgumentError("centroid must be finite"))
    return center
end

"""
    beam_covariance(coordinates; corrected=false)

Return the centered covariance matrix of a macroparticle ensemble. Rows are
particles and columns are coordinates. The default population normalization is
`1/N`; set `corrected=true` for the sample normalization `1/(N-1)`.
"""
function beam_covariance(coordinates::AbstractMatrix{S};
                         corrected::Bool=false) where {S<:Real}
    particles, dimension = size(coordinates)
    particles > (corrected ? 1 : 0) ||
        throw(ArgumentError("not enough particles for the requested normalization"))
    dimension > 0 || throw(DimensionMismatch("coordinates must have at least one column"))
    T = float(S)
    centered = Matrix{T}(coordinates)
    all(isfinite, centered) || throw(ArgumentError("coordinates must be finite"))
    center = _column_centroid(centered)
    centered .-= transpose(center)
    denominator = T(particles - (corrected ? 1 : 0))
    covariance = transpose(centered) * centered / denominator
    return Matrix(Symmetric((covariance + transpose(covariance)) / T(2)))
end

"""
    match_moments!(coordinates, sigma; centroid=nothing, corrected=false)

Center and linearly transform an existing `N x 4` or `N x 6` ensemble so its
finite-sample covariance equals `sigma` to roundoff. `sigma` may be positive
semidefinite, but the input sample covariance must be positive definite, which
requires more particles than coordinates. Population (`1/N`) moments are used
unless `corrected=true` is requested.
"""
function match_moments!(coordinates::AbstractMatrix{T}, sigma::AbstractMatrix;
                        centroid=nothing, corrected::Bool=false,
                        atol=nothing, rtol=nothing) where {T<:AbstractFloat}
    particles, dimension = size(coordinates)
    dimension == _distribution_dimension(sigma) || throw(DimensionMismatch(
        "coordinate columns and covariance dimension must agree",
    ))
    particles > dimension || throw(ArgumentError(
        "exact moment matching requires more particles than coordinates",
    ))
    all(isfinite, coordinates) || throw(ArgumentError("coordinates must be finite"))

    _, target = _validated_covariance(sigma, T; atol=atol, rtol=rtol)
    target_root = _psd_sqrt(target; atol=atol, rtol=rtol)
    target_center = _target_centroid(centroid, dimension, T)

    sample_center = _column_centroid(coordinates)
    coordinates .-= transpose(sample_center)
    denominator = T(particles - (corrected ? 1 : 0))
    sample_covariance = transpose(coordinates) * coordinates / denominator
    sample_inverse_root = _positive_inverse_sqrt(
        sample_covariance; atol=atol, rtol=rtol,
    )
    coordinates .= coordinates * sample_inverse_root * target_root
    coordinates .+= transpose(target_center)

    # Remove the final matrix-multiplication roundoff from the requested mean.
    residual = target_center - _column_centroid(coordinates)
    coordinates .+= transpose(residual)
    return coordinates
end


"""
    gaussian_distribution([rng], nparticles, sigma;
                          centroid=nothing, exact_moments=true, corrected=false)

Generate a Gaussian `N x 4` or `N x 6` coordinate matrix with target covariance
`sigma`. With `exact_moments=true` (the default), finite-sample mean and
covariance are corrected to the requested values. This constrained ensemble is
not a set of statistically independent samples, although it approaches one as
`nparticles` grows.
"""
function gaussian_distribution(rng::Random.AbstractRNG, nparticles::Integer,
                               sigma::AbstractMatrix;
                               centroid=nothing, exact_moments::Bool=true,
                               corrected::Bool=false, atol=nothing, rtol=nothing)
    nparticles > 0 || throw(ArgumentError("nparticles must be positive"))
    dimension = _distribution_dimension(sigma)
    T = _distribution_type(sigma, centroid)
    T <: AbstractFloat || throw(ArgumentError("distribution scalar type must be floating point"))
    _, target = _validated_covariance(sigma, T; atol=atol, rtol=rtol)
    center = _target_centroid(centroid, dimension, T)
    coordinates = randn(rng, T, Int(nparticles), dimension)
    if exact_moments
        match_moments!(
            coordinates, target; centroid=center, corrected=corrected,
            atol=atol, rtol=rtol,
        )
    else
        coordinates .= coordinates * _psd_sqrt(target; atol=atol, rtol=rtol)
        coordinates .+= transpose(center)
    end
    return coordinates
end

gaussian_distribution(nparticles::Integer, sigma::AbstractMatrix; kwargs...) =
    gaussian_distribution(Random.default_rng(), nparticles, sigma; kwargs...)

function _twiss_covariance(beta::T, alpha::T, emittance::T) where T
    isfinite(beta) && beta > zero(T) ||
        throw(ArgumentError("Twiss beta must be finite and positive"))
    isfinite(alpha) || throw(ArgumentError("Twiss alpha must be finite"))
    isfinite(emittance) && emittance >= zero(T) ||
        throw(ArgumentError("emittance must be finite and nonnegative"))
    gamma = (one(T) + alpha^2) / beta
    return emittance * T[beta -alpha; -alpha gamma]
end

function _response_vector(response, default, name::AbstractString, ::Type{T}) where T
    values = isnothing(response) ? T.(default) : T.(collect(response))
    length(values) == 4 || throw(DimensionMismatch("$name must have length 4"))
    all(isfinite, values) || throw(ArgumentError("$name must be finite"))
    return values
end

"""
    matched_covariance(optics::optics4DUC;
                       emitx, emity, emitz=0, betaz=1, alphaz=0,
                       longitudinal=nothing, dispersion=nothing,
                       crab_dispersion=nothing)

Construct a canonical `6 x 6` covariance matrix from uncoupled transverse
Twiss parameters. `longitudinal` may supply a `2 x 2` covariance in
`(z, delta_E)` and overrides `emitz`, `betaz`, and `alphaz`.

The four-component response vectors act on the transverse coordinates
`(x, px, y, py)`: `dispersion` is the derivative with respect to the stored
coordinate `δE`, and `crab_dispersion` the derivative with respect to `z`. When
omitted, ordinary dispersion comes from the `eta`/`etap` fields of `optics` and
crab dispersion is zero. The covariance lives in the stored coordinates, so it
needs the `δE` response: use the `dx`, `dpx`, `dy`, `dpy` of
`periodic_twiss(lat, beam; wrt=:deltae)`, or divide a `δP` dispersion (MAD-X,
or `periodic_twiss`'s default) by the reference `β0`.
"""
function matched_covariance(optics::optics4DUC;
                            emitx, emity, emitz=nothing,
                            betaz=nothing, alphaz=nothing,
                            longitudinal=nothing,
                            dispersion=nothing, crab_dispersion=nothing,
                            atol=nothing, rtol=nothing)
    ox = optics.optics_x
    oy = optics.optics_y
    T = _distribution_type(
        ox.beta, ox.alpha, ox.eta, ox.etap,
        oy.beta, oy.alpha, oy.eta, oy.etap,
        emitx, emity, emitz, betaz, alphaz,
        longitudinal, dispersion, crab_dispersion,
    )
    longitudinal_emittance = isnothing(emitz) ? zero(T) : T(emitz)
    longitudinal_beta = isnothing(betaz) ? one(T) : T(betaz)
    longitudinal_alpha = isnothing(alphaz) ? zero(T) : T(alphaz)
    sigma0 = zeros(T, 6, 6)
    sigma0[1:2, 1:2] .= _twiss_covariance(T(ox.beta), T(ox.alpha), T(emitx))
    sigma0[3:4, 3:4] .= _twiss_covariance(T(oy.beta), T(oy.alpha), T(emity))
    if isnothing(longitudinal)
        sigma0[5:6, 5:6] .=
            _twiss_covariance(
                longitudinal_beta, longitudinal_alpha, longitudinal_emittance,
            )
    else
        size(longitudinal) == (2, 2) ||
            throw(DimensionMismatch("longitudinal covariance must be 2x2"))
        _, longitudinal_matrix = _validated_covariance(
            longitudinal, T; atol=atol, rtol=rtol, supported_dimension=false,
        )
        _psd_sqrt(longitudinal_matrix; atol=atol, rtol=rtol)
        sigma0[5:6, 5:6] .= longitudinal_matrix
    end

    default_dispersion = (ox.eta, ox.etap, oy.eta, oy.etap)
    ordinary = _response_vector(dispersion, default_dispersion, "dispersion", T)
    crab = _response_vector(crab_dispersion, ntuple(_ -> zero(T), 4),
                            "crab_dispersion", T)
    response = Matrix{T}(I, 6, 6)
    response[1:4, 5] .= crab
    response[1:4, 6] .= ordinary
    sigma = response * sigma0 * transpose(response)
    return Matrix(Symmetric((sigma + transpose(sigma)) / T(2)))
end

"""
    matched_gaussian([rng], nparticles, optics::optics4DUC; kwargs...)

Generate an `N x 6` Gaussian ensemble matched to transverse optics, emittances,
longitudinal covariance, and optional ordinary/crab dispersion. Covariance
keywords are forwarded to [`matched_covariance`](@ref); `centroid`,
`exact_moments`, and `corrected` control [`gaussian_distribution`](@ref).
"""
function matched_gaussian(rng::Random.AbstractRNG, nparticles::Integer,
                          optics::optics4DUC;
                          centroid=nothing, exact_moments::Bool=true,
                          corrected::Bool=false, atol=nothing, rtol=nothing,
                          kwargs...)
    sigma = matched_covariance(optics; atol=atol, rtol=rtol, kwargs...)
    return gaussian_distribution(
        rng, nparticles, sigma; centroid=centroid,
        exact_moments=exact_moments, corrected=corrected,
        atol=atol, rtol=rtol,
    )
end

matched_gaussian(nparticles::Integer, optics::optics4DUC; kwargs...) =
    matched_gaussian(Random.default_rng(), nparticles, optics; kwargs...)

function matched_gaussian(rng::Random.AbstractRNG, nparticles::Integer;
                          betax, alphax=nothing, betay, alphay=nothing, kwargs...)
    T = _distribution_type(betax, alphax, betay, alphay)
    ax = isnothing(alphax) ? zero(T) : T(alphax)
    ay = isnothing(alphay) ? zero(T) : T(alphay)
    optics = optics4DUC(T(betax), ax, T(betay), ay)
    return matched_gaussian(rng, nparticles, optics; kwargs...)
end

matched_gaussian(nparticles::Integer; kwargs...) =
    matched_gaussian(Random.default_rng(), nparticles; kwargs...)

"""
    projected_emittances(sigma)

Return `sqrt(det(sigma_plane))` for each canonical `2 x 2` diagonal block of a
`4 x 4` or `6 x 6` covariance matrix. These projected emittances generally
differ from normal-mode emittances when coupling or dispersion is present.
"""
function projected_emittances(sigma::AbstractMatrix; atol=nothing, rtol=nothing)
    dimension = _distribution_dimension(sigma)
    T = _distribution_type(sigma)
    _, matrix = _validated_covariance(sigma, T; atol=atol, rtol=rtol)
    _psd_sqrt(matrix; atol=atol, rtol=rtol)
    emittances = zeros(T, div(dimension, 2))
    scale = maximum(abs, matrix)
    tolerance = _moment_tolerances(T, scale^2, atol, rtol)
    for plane in eachindex(emittances)
        i = 2 * plane - 1
        determinant = matrix[i, i] * matrix[i + 1, i + 1] -
                      matrix[i, i + 1] * matrix[i + 1, i]
        determinant >= -tolerance ||
            throw(ArgumentError("projected covariance block is not positive semidefinite"))
        emittances[plane] = sqrt(max(determinant, zero(T)))
    end
    return emittances
end

"""
    eigenemittances(sigma)

Return the sorted canonical normal-mode emittances of a `4 x 4` or `6 x 6`
covariance matrix. They are the paired singular values of
`sqrt(sigma) * J * sqrt(sigma)`, where `J` is TrackPad's canonical symplectic
form. Unlike [`projected_emittances`](@ref), these values are invariant under
linear symplectic coordinate transformations.
"""
function eigenemittances(sigma::AbstractMatrix; atol=nothing, rtol=nothing)
    dimension = _distribution_dimension(sigma)
    T = _distribution_type(sigma)
    _, matrix = _validated_covariance(sigma, T; atol=atol, rtol=rtol)
    root = _psd_sqrt(matrix; atol=atol, rtol=rtol)
    symplectic_form = zeros(T, dimension, dimension)
    for i in 1:2:dimension
        symplectic_form[i, i + 1] = one(T)
        symplectic_form[i + 1, i] = -one(T)
    end
    values = sort(svdvals(root * symplectic_form * root))
    return T[(values[2i - 1] + values[2i]) / T(2) for i in 1:div(dimension, 2)]
end
