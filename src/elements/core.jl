abstract type AbstractElement end
abstract type AbstractMagnet <: AbstractElement end
abstract type AbstractDrift <: AbstractElement end
abstract type AbstractCavity <: AbstractElement end
abstract type AbstractTransferMap <: AbstractElement end
abstract type AbstractTransverseMap <: AbstractTransferMap end
abstract type AbstractLongitudinalRFMap <: AbstractTransferMap end

function _promote_element_type(L, args...)
    T = typeof(L)
    for arg in args
        if !isnothing(arg)
            T = promote_type(T, eltype(arg))
        end
    end
    return T
end

_default_vec(val, ::Type{T}, ::Val{N}) where {T, N} = isnothing(val) ? zero(SVector{N, T}) : val
_default_mat(val, ::Type{T}, ::Val{N}) where {T, N} = isnothing(val) ? zero(SMatrix{N, N, T, N * N}) : val
