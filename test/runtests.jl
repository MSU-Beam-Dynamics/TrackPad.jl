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
        include_isolated(:VerifyRingOptics, "verify_ring_optics.jl")
        include_isolated(:VerifyLatticePlot, "verify_lattice_plot.jl")
        include_isolated(:VerifyDistributions, "verify_distributions.jl")
        include_isolated(:VerifyClosedOrbit, "verify_closed_orbit.jl")
        include_isolated(:VerifyTimeDependence, "verify_time_dependence.jl")
        include_isolated(:VerifyEnzymeCompatibility, "verify_enzyme_compat.jl")
        include_isolated(:VerifyElements, "verify_elements.jl")
        include_isolated(:VerifyTracking, "verify_tracking.jl")
        include_isolated(:VerifyReviewFixes, "verify_review_fixes.jl")
        include_isolated(:VerifyBeamBeam, "verify_beambeam.jl")
        include_isolated(:VerifyGPU, "verify_gpu.jl")
        include_isolated(:VerifyIO, "verify_io.jl")
        include_isolated(:VerifyExamples, "verify_examples.jl")
    end
    @testset "TPSA" begin
        include_isolated(:VerifyTPSA, "verify_tpsa.jl")
        include_isolated(:VerifyPolySeries, "verify_polyseries.jl")
    end
end
