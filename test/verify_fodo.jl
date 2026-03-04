using Test
using TrackPad
using LinearAlgebra
import JuTrack

# Deterministic initial coordinates shared across all regression tests.
particles_initial = [
    1.0e-4   2.0e-4   3.0e-4  -1.0e-4   5.0e-5   2.0e-4
   -2.2e-4  1.7e-4  -1.1e-4   2.3e-4  -7.0e-5   1.0e-4
    3.5e-4  -2.1e-4  8.0e-5  -1.8e-4   1.4e-4  -2.6e-4
   -4.0e-4  2.9e-4   1.6e-4   9.0e-5  -1.2e-4   3.1e-4
    5.2e-4  -3.3e-4 -2.4e-4   1.1e-4   2.6e-4  -3.7e-4
   -6.1e-4  4.4e-4   2.7e-4  -2.5e-4  -3.0e-4   4.5e-4
    7.0e-4  -5.2e-4 -3.1e-4   3.4e-4   3.3e-4  -5.0e-4
   -8.0e-4  6.0e-4   3.9e-4  -4.2e-4  -3.8e-4   5.8e-4
    9.1e-4  -6.7e-4 -4.6e-4   5.1e-4   4.4e-4  -6.3e-4
   -9.8e-4  7.3e-4   5.3e-4  -5.9e-4  -4.9e-4   7.1e-4
]
energy_val = 3.5e9
old_exact_beti = JuTrack.use_exact_beti
JuTrack.use_exact_beti = 1

# Reconstruct Lattice in TrackPad
# JuTrack: LINE = [D1, Q1, D2, B1, D3, Q2, D4]
# D1=D2=D3=D4 = DRIFT(len=1.0)
# Q1 = KQUAD(len=1.0, k1=-0.9)
# Q2 = KQUAD(len=1.0, k1=0.3)
# B1 = SBEND(len=0.6, angle=pi/15.0)

D1 = Drift(1.0; name="D1")
# JuTrack SBEND uses 10 steps by default. TrackPad should match.
Q1 = Quadrupole(1.0, -0.9; name="Q1", num_int_steps=10)
D2 = Drift(1.0; name="D2")
B1 = SBend(0.6, pi/15.0; name="B1", num_int_steps=10)
D3 = Drift(1.0; name="D3")
Q2 = Quadrupole(1.0, 0.3; name="Q2", num_int_steps=10)
D4 = Drift(1.0; name="D4")

line = Lattice([D1, Q1, D2, B1, D3, Q2, D4])

# JuTrack reference with matching symplectic-equivalent definitions.
line_jt = [
    JuTrack.DRIFT(name="D1", len=1.0),
    JuTrack.SBEND(name="Q1", len=1.0, angle=0.0, PolynomB=[0.0, -0.9, 0.0, 0.0]),
    JuTrack.DRIFT(name="D2", len=1.0),
    JuTrack.SBEND(name="B1", len=0.6, angle=pi / 15.0),
    JuTrack.DRIFT(name="D3", len=1.0),
    JuTrack.SBEND(name="Q2", len=1.0, angle=0.0, PolynomB=[0.0, 0.3, 0.0, 0.0]),
    JuTrack.DRIFT(name="D4", len=1.0),
]
try
    beam_jt = JuTrack.Beam(copy(particles_initial), energy=energy_val, mass=JuTrack.m_e)
    JuTrack.linepass!(line_jt, beam_jt)
    particles_final_jutrack = beam_jt.r

    # Prepare coordinates
    coords = deepcopy(particles_initial)
    nparticles = size(coords, 1)
    lost_flags = zeros(Int, nparticles)

    # Tracking
    beam = Beam(energy_val)
    linepass!(coords, line, beam, lost_flags)

    # Verification
    diff = norm(coords - particles_final_jutrack)

    @testset "JuTrack vs TrackPad FODO Verification" begin
        @test diff < 1e-15
        @test isapprox(coords, particles_final_jutrack, atol=1e-15)
    end
finally
    JuTrack.use_exact_beti = old_exact_beti
end
