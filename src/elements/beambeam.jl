"""
    LorentzBoost{T, N}

Lorentz boost element for crossing-angle transformations.
"""
struct LorentzBoost{T, N} <: AbstractElement
    name::N
    angle::T
    cosang::T
    tanang::T
    mode::Int
end

function LorentzBoost(angle;
                      name::Union{Symbol, String} = :LORENTZBOOST,
                      mode::Int = 0)
    T = _promote_element_type(angle)
    LorentzBoost{T, Symbol}(Symbol(name), T(angle), cos(T(angle)), tan(T(angle)), mode)
end

function Adapt.adapt_structure(to, x::LorentzBoost)
    LorentzBoost(adapt(to, x.angle); name = x.name, mode = x.mode)
end

"""
    InvLorentzBoost{T, N}

Inverse Lorentz boost element for crossing-angle transformations.
"""
struct InvLorentzBoost{T, N} <: AbstractElement
    name::N
    angle::T
    sinang::T
    cosang::T
    mode::Int
end

function InvLorentzBoost(angle;
                         name::Union{Symbol, String} = :INVLORENTZBOOST,
                         mode::Int = 0)
    T = _promote_element_type(angle)
    InvLorentzBoost{T, Symbol}(Symbol(name), T(angle), sin(T(angle)), cos(T(angle)), mode)
end

function Adapt.adapt_structure(to, x::InvLorentzBoost)
    InvLorentzBoost(adapt(to, x.angle); name = x.name, mode = x.mode)
end

"""
    StrongThinGaussianBeam{T, N}

Thin-lens strong beam-beam kick model with Gaussian transverse profile.
"""
struct StrongThinGaussianBeam{T, N} <: AbstractElement
    name::N
    amplitude::T
    rmssizex::T
    rmssizey::T
    zloc::T
    xoffset::T
    yoffset::T
end

function StrongThinGaussianBeam(amplitude, rmssizex, rmssizey;
                                name::Union{Symbol, String} = :STRONGTHINGAUSSIANBEAM,
                                zloc = 0.0, xoffset = 0.0, yoffset = 0.0)
    T = _promote_element_type(amplitude, rmssizex, rmssizey, zloc, xoffset, yoffset)
    StrongThinGaussianBeam{T, Symbol}(Symbol(name), T(amplitude), T(rmssizex), T(rmssizey), T(zloc), T(xoffset), T(yoffset))
end

function Adapt.adapt_structure(to, x::StrongThinGaussianBeam)
    StrongThinGaussianBeam(
        adapt(to, x.amplitude), adapt(to, x.rmssizex), adapt(to, x.rmssizey);
        name = x.name, zloc = adapt(to, x.zloc), xoffset = adapt(to, x.xoffset), yoffset = adapt(to, x.yoffset),
    )
end

"""
    StrongGaussianBeam{T, N}

Multi-slice strong beam-beam model metadata.
"""
struct StrongGaussianBeam{T, N, V<:AbstractVector{T}} <: AbstractElement
    name::N
    charge::T
    mass::T
    atomnum::T
    num_particle::Int
    total_energy::T
    momentum::T
    gamma::T
    beta::T
    beamsize::SVector{2, T}
    nzslice::Int
    zslice_center::V
    zslice_npar::V
    xoffsets::V
    yoffsets::V
end

function StrongGaussianBeam(charge, mass, atomnum, num_particle::Int, total_energy, beamsize;
                            name::Union{Symbol, String} = :STRONGGAUSSIANBEAM,
                            nzslice::Int = 1, zslice_center = nothing, zslice_npar = nothing,
                            xoffsets = nothing, yoffsets = nothing)
    T = _promote_element_type(charge, mass, atomnum, total_energy, beamsize)
    mom2 = max(T(total_energy)^2 - T(mass)^2, zero(T))
    momentum = sqrt(mom2)
    gamma = T(total_energy) / T(mass)
    beta = ifelse(iszero(total_energy), zero(T), momentum / T(total_energy))
    bs = SVector{2, T}(T(beamsize[1]), T(beamsize[2]))
    zc = isnothing(zslice_center) ? collect(range(-one(T), one(T), length = nzslice)) : T.(collect(zslice_center))
    zn = isnothing(zslice_npar) ? fill(one(T) / T(max(nzslice, 1)), nzslice) : T.(collect(zslice_npar))
    xo = isnothing(xoffsets) ? zeros(T, nzslice) : T.(collect(xoffsets))
    yo = isnothing(yoffsets) ? zeros(T, nzslice) : T.(collect(yoffsets))
    StrongGaussianBeam{T, Symbol, Vector{T}}(
        Symbol(name), T(charge), T(mass), T(atomnum), num_particle, T(total_energy), momentum, gamma, beta,
        bs, nzslice, zc, zn, xo, yo,
    )
end

function Adapt.adapt_structure(to, x::StrongGaussianBeam)
    StrongGaussianBeam(
        adapt(to, x.charge), adapt(to, x.mass), adapt(to, x.atomnum), x.num_particle, adapt(to, x.total_energy), x.beamsize;
        name = x.name, nzslice = x.nzslice,
        zslice_center = adapt(to, x.zslice_center), zslice_npar = adapt(to, x.zslice_npar),
        xoffsets = adapt(to, x.xoffsets), yoffsets = adapt(to, x.yoffsets),
    )
end
