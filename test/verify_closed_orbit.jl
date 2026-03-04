using Test
using TrackPad
import JuTrack
using StaticArrays
using LinearAlgebra

# A simple stable ring for closed-orbit checks
D = Drift(1.0; name="D")
QF = Quadrupole(0.5, 0.6; name="QF", num_int_steps=8)
QD = Quadrupole(0.5, -0.6; name="QD", num_int_steps=8)
ring = Lattice([D, QF, D, QD])
beam = Beam(3.0e9)

ring_jt = [
    JuTrack.DRIFT(len=1.0),
    JuTrack.SBEND(len=0.5, angle=0.0, PolynomB=[0.0, 0.6, 0.0, 0.0]),
    JuTrack.DRIFT(len=1.0),
    JuTrack.SBEND(len=0.5, angle=0.0, PolynomB=[0.0, -0.6, 0.0, 0.0]),
]

old_exact_beti = JuTrack.use_exact_beti
JuTrack.use_exact_beti = 1
try
    @testset "TrackPad Closed Orbit API" begin
        x6 = find_closed_orbit_6d(ring, beam; x0=@SVector [1e-3, -2e-4, 8e-4, 3e-4, 2e-3, 1e-3])
        r6_out = linepass(ring, x6, beam)
        @test norm(r6_out - x6) < 1e-8
        x6_jt = JuTrack.find_closed_orbit_6d(ring_jt; energy=3.0e9, mass=JuTrack.m_e)
        @test isapprox(collect(x6[1:4]), x6_jt[1:4]; atol=1e-15)
        @test isfinite(x6[5])
        @test isfinite(x6[6])

        x4 = find_closed_orbit_4d(ring, beam; dp=0.0, x0=@SVector [1e-3, -2e-4, 8e-4, 3e-4])
        r6 = linepass(ring, SVector{6,Float64}(x4[1], x4[2], x4[3], x4[4], 0.0, 0.0), beam)
        @test norm(SVector{4,Float64}(r6[1], r6[2], r6[3], r6[4]) - x4) < 1e-8
        x4_jt = JuTrack.find_closed_orbit_4d(ring_jt; dp=0.0, energy=3.0e9, mass=JuTrack.m_e)
        @test isapprox(collect(x4), x4_jt; atol=1e-15)
    end
finally
    JuTrack.use_exact_beti = old_exact_beti
end

x4 = find_closed_orbit_4d(ring, beam; dp=0.0, x0=@SVector [1e-3, -2e-4, 8e-4, 3e-4])
r6 = linepass(ring, SVector{6,Float64}(x4[1], x4[2], x4[3], x4[4], 0.0, 0.0), beam)
       
x4_jt = JuTrack.find_closed_orbit_4d(ring_jt; dp=0.0, energy=3.0e9, mass=JuTrack.m_e)