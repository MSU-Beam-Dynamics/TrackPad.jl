"""
    Wiggler{T, N}

Canonical wiggler element metadata. Harmonics are stored in six-integer
blocks. Horizontal-field (`Bx`) blocks require nonzero `kx` and `kz` entries;
vertical-field (`By`) blocks require nonzero `ky` and `kz` entries.
"""
struct Wiggler{T, N, V<:AbstractVector{Int}} <: AbstractMagnet
    name::N
    L::T
    lw::T
    Bmax::T
    Nsteps::Int
    By::V
    Bx::V
    energy::T
    NHharm::Int
    NVharm::Int
    rad::Int
    t1::SVector{6, T}
    t2::SVector{6, T}
    r1::SMatrix{6, 6, T, 36}
    r2::SMatrix{6, 6, T, 36}
end

function Wiggler(L;
                 name::Union{Symbol, String} = :WIGGLER,
                 lw = 0.0, Bmax = 0.0, Nsteps::Int = 10,
                 By = [1, 1, 0, 1, 1, 0], Bx = Int[],
                 energy = 1.0e9, rad::Int = 0,
                 t1 = nothing, t2 = nothing,
                 r1 = nothing, r2 = nothing)
    T = _promote_element_type(L, lw, Bmax, energy, t1, t2, r1, r2)
    byv = Int[By...]
    bxv = Int[Bx...]
    isfinite(L) && L >= zero(T) || throw(ArgumentError("L must be finite and nonnegative"))
    isfinite(lw) && lw > zero(T) || throw(ArgumentError("lw must be finite and positive"))
    isfinite(Bmax) || throw(ArgumentError("Bmax must be finite"))
    isfinite(energy) && energy > T(M_ELECTRON) ||
        throw(ArgumentError("energy must be finite and greater than the electron rest-mass energy"))
    Nsteps > 0 || throw(ArgumentError("Nsteps must be positive"))
    length(byv) % 6 == 0 || throw(ArgumentError("By must contain complete six-value harmonic blocks"))
    length(bxv) % 6 == 0 || throw(ArgumentError("Bx must contain complete six-value harmonic blocks"))
    for base in 0:6:length(byv)-1
        iszero(byv[base + 4]) && throw(ArgumentError("By harmonic ky entries must be nonzero"))
        iszero(byv[base + 5]) && throw(ArgumentError("By harmonic kz entries must be nonzero"))
    end
    for base in 0:6:length(bxv)-1
        iszero(bxv[base + 3]) && throw(ArgumentError("Bx harmonic kx entries must be nonzero"))
        iszero(bxv[base + 5]) && throw(ArgumentError("Bx harmonic kz entries must be nonzero"))
    end
    nh = length(byv) ÷ 6
    nv = length(bxv) ÷ 6
    Wiggler{T, Symbol, Vector{Int}}(
        Symbol(name), T(L), T(lw), T(Bmax), Nsteps,
        byv, bxv, T(energy), nh, nv, rad,
        SVector{6, T}(_default_vec(t1, T, Val(6))),
        SVector{6, T}(_default_vec(t2, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r1, T, Val(6))),
        SMatrix{6, 6, T}(_default_mat(r2, T, Val(6))),
    )
end

function Adapt.adapt_structure(to, x::Wiggler)
    Wiggler(
        adapt(to, x.L);
        name = x.name, lw = adapt(to, x.lw), Bmax = adapt(to, x.Bmax), Nsteps = x.Nsteps,
        By = adapt(to, x.By), Bx = adapt(to, x.Bx), energy = adapt(to, x.energy), rad = x.rad,
        t1 = adapt(to, x.t1), t2 = adapt(to, x.t2), r1 = adapt(to, x.r1), r2 = adapt(to, x.r2),
    )
end
