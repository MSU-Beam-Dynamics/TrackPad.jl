using Test

function include_isolated(name::Symbol, filename::AbstractString)
    test_module = Module(name)
    Base.include(test_module, joinpath(@__DIR__, filename))
    return nothing
end

@testset "TrackPad Tests" begin
    @testset "Physics Verification" begin
        include_isolated(:VerifyFODO, "verify_fodo.jl")
        include_isolated(:VerifyDBA, "verify_dba.jl")
        include_isolated(:VerifyBend, "verify_bend.jl")
        include_isolated(:VerifyExactBend, "verify_exact_bend.jl")
        include_isolated(:VerifyOptics, "verify_optics.jl")
        include_isolated(:VerifyClosedOrbit, "verify_closed_orbit.jl")
        include_isolated(:VerifyTimeDependence, "verify_time_dependence.jl")
        include_isolated(:VerifyEnzymeCompatibility, "verify_enzyme_compat.jl")
        include_isolated(:VerifyElements, "verify_elements.jl")
        include_isolated(:VerifyGPU, "verify_gpu.jl")
    end
    @testset "TPSA" begin
        include_isolated(:VerifyTPSA, "verify_tpsa.jl")
        include_isolated(:VerifyPolySeries, "verify_polyseries.jl")
    end
end
