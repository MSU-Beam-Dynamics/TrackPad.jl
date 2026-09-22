using Test
using TrackPad
using StaticArrays
using LinearAlgebra
Base.include(@__MODULE__, joinpath(@__DIR__, "convention_helpers.jl"))

# A simple stable ring for closed-orbit checks
D = Drift(1.0; name="D")
QF = Quadrupole(0.5, 0.6; name="QF", num_int_steps=8)
QD = Quadrupole(0.5, -0.6; name="QD", num_int_steps=8)
ring = Lattice([D, QF, D, QD]; periodic=true)
beam = Beam(3.0e9)

# JuTrack reference ring: DRIFT(1.0), SBEND(len=0.5, angle=0, PolynomB=[0,0.6,0,0]),
# DRIFT(1.0), SBEND(len=0.5, angle=0, PolynomB=[0,-0.6,0,0]), at 3 GeV.
x6_jt = jutrack_reference("closed_orbit/6d")
x4_jt = jutrack_reference("closed_orbit/4d")

@testset "TrackPad Closed Orbit API" begin
    x6 = find_closed_orbit_6d(ring, beam; x0=@SVector [1e-3, -2e-4, 8e-4, 3e-4, 2e-3, 1e-3])
    r6_out = linepass(ring, x6, beam)
    @test norm(r6_out - x6) < 1e-8
    @test isapprox(collect(x6[1:4]), x6_jt[1:4]; atol=1e-15)
    @test isfinite(x6[5])
    @test isfinite(x6[6])

    x4 = find_closed_orbit_4d(ring, beam; dp=0.0, x0=@SVector [1e-3, -2e-4, 8e-4, 3e-4])
    r6 = linepass(ring, SVector{6,Float64}(x4[1], x4[2], x4[3], x4[4], 0.0, 0.0), beam)
    @test norm(SVector{4,Float64}(r6[1], r6[2], r6[3], r6[4]) - x4) < 1e-8
    @test isapprox(collect(x4), x4_jt; atol=1e-15)
end
