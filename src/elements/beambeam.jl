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
    CLASSICAL_RADIUS_EV_M

`e^2 / (4 π ε0)` in eV·m. The classical radius of a particle with rest-mass
energy `mc2` (eV) and charge `q` (units of `e`) is `q^2 * CLASSICAL_RADIUS_EV_M / mc2`.
"""
const CLASSICAL_RADIUS_EV_M = 1.439964547e-9

"""
    classical_radius(mass_ev, charge) -> r0 [m]

Classical radius `q^2 e^2 / (4 π ε0 m c^2)` for rest-mass energy `mass_ev` in eV
and charge `charge` in units of `e` (2.818e-15 m for the electron).
"""
classical_radius(mass_ev::Real, charge::Real) = charge^2 * CLASSICAL_RADIUS_EV_M / mass_ev

"""
    StrongThinGaussianBeam{T, N}

Thin strong beam-beam kick from a bi-Gaussian strong beam (Bassetti-Erskine
field; closed round-beam form when `rmssizex == rmssizey`).

    Δpx = amplitude * Ex(x - xoffset, y - yoffset)
    Δpy = amplitude * Ey(x - xoffset, y - yoffset)

with `(Ex, Ey) = gaussian_beam_field(...)`, normalized so that the linear kick
is `Δpx ≈ amplitude * x / rmssizex^2` for a round beam. Physically

    amplitude = N * r0_w * q_w * q_s / γ_w

for `N` strong-beam particles, the classical radius `r0_w` and Lorentz factor
`γ_w` of the *weak* (tracked) beam, and the signed charges `q_w`, `q_s`;
use [`beambeam_amplitude`](@ref) to build it from a `Beam`. Positive
`amplitude` (like charges) defocuses. The linear beam-beam parameter is
`ξ = β* amplitude / (4π σ^2)`. `zloc` is metadata only; the thin kick does not
depend on it.
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
    StrongGaussianBeam{T, N, V}

Longitudinally sliced strong beam for the synchro-beam mapping. The fields
`charge`, `mass`, `atomnum`, `num_particle`, `total_energy`, `momentum`,
`gamma`, `beta` describe the *strong* beam. `zslice_npar` holds the number of
strong particles in each slice (absolute counts, default `num_particle/nzslice`),
`zslice_center` their longitudinal positions (positive toward the head of the
strong bunch), `xoffsets`/`yoffsets` per-slice transverse offsets.

`kick_scale = r0_w * q_w * q_s / γ_w` couples the strong beam to the weak
(tracked) beam: the slice kick is `kick_scale * zslice_npar[i] *
gaussian_beam_field(...)`. Pass `weak_beam = Beam(...)` to the constructor to
compute it, or give `kick_scale` directly. See `pass!(::StrongGaussianBeam, ...)`
for the mapping.
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
    kick_scale::T
    beamsize::SVector{2, T}
    nzslice::Int
    zslice_center::V
    zslice_npar::V
    xoffsets::V
    yoffsets::V
end

function StrongGaussianBeam(charge, mass, atomnum, num_particle::Int, total_energy, beamsize;
                            name::Union{Symbol, String} = :STRONGGAUSSIANBEAM,
                            kick_scale = nothing, weak_beam = nothing,
                            nzslice::Int = 1, zslice_center = nothing, zslice_npar = nothing,
                            xoffsets = nothing, yoffsets = nothing)
    T = _promote_element_type(charge, mass, atomnum, total_energy, beamsize)
    nzslice >= 1 || throw(ArgumentError("nzslice must be positive"))
    if kick_scale === nothing
        weak_beam === nothing && throw(ArgumentError(
            "StrongGaussianBeam needs the weak-beam coupling: pass `weak_beam = Beam(...)` " *
            "or `kick_scale = r0_weak * q_weak * q_strong / gamma_weak`"))
        kick_scale = classical_radius(weak_beam.mass, weak_beam.charge) *
                     weak_beam.charge * charge / weak_beam.gamma
    end
    mom2 = max(T(total_energy)^2 - T(mass)^2, zero(T))
    momentum = sqrt(mom2)
    gamma = T(total_energy) / T(mass)
    beta = ifelse(iszero(total_energy), zero(T), momentum / T(total_energy))
    bs = SVector{2, T}(T(beamsize[1]), T(beamsize[2]))
    zc = isnothing(zslice_center) ? zeros(T, nzslice) : T.(collect(zslice_center))
    zn = isnothing(zslice_npar) ? fill(T(num_particle) / T(nzslice), nzslice) : T.(collect(zslice_npar))
    xo = isnothing(xoffsets) ? zeros(T, nzslice) : T.(collect(xoffsets))
    yo = isnothing(yoffsets) ? zeros(T, nzslice) : T.(collect(yoffsets))
    length(zc) == nzslice && length(zn) == nzslice && length(xo) == nzslice && length(yo) == nzslice ||
        throw(ArgumentError("zslice_center, zslice_npar, xoffsets and yoffsets must all have length nzslice"))
    StrongGaussianBeam{T, Symbol, Vector{T}}(
        Symbol(name), T(charge), T(mass), T(atomnum), num_particle, T(total_energy), momentum, gamma, beta,
        T(kick_scale), bs, nzslice, zc, zn, xo, yo,
    )
end

function Adapt.adapt_structure(to, x::StrongGaussianBeam)
    StrongGaussianBeam(
        adapt(to, x.charge), adapt(to, x.mass), adapt(to, x.atomnum), x.num_particle, adapt(to, x.total_energy), x.beamsize;
        name = x.name, kick_scale = adapt(to, x.kick_scale), nzslice = x.nzslice,
        zslice_center = adapt(to, x.zslice_center), zslice_npar = adapt(to, x.zslice_npar),
        xoffsets = adapt(to, x.xoffsets), yoffsets = adapt(to, x.yoffsets),
    )
end
