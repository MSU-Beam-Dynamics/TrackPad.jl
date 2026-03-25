using Test

@testset "TrackPad Tests" begin
    @testset "Physics Verification" begin
        include("verify_fodo.jl")
        include("verify_dba.jl")
        include("verify_bend.jl")
        include("verify_optics.jl")
        include("verify_closed_orbit.jl")
        include("verify_time_dependence.jl")
        include("verify_enzyme_compat.jl")
        include("verify_elements.jl")
        include("verify_polyseries.jl")
    end
end
