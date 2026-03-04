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

# Reconstruct Bend Lattice in TrackPad
# B_DBA = SBEND(len=1.0, angle=pi/9, e1=pi/18, e2=pi/18)
angle_val = pi/4
B_DBA = SBend(1.0, angle_val, angle_val/2, angle_val/2; name="B_DBA")

line_bend = Lattice([B_DBA])

# JuTrack reference.
line_bend_jt = [JuTrack.SBEND(len=1.0, angle=angle_val, e1=angle_val / 2, e2=angle_val / 2)]
try
    beam_jt = JuTrack.Beam(copy(particles_initial), energy=energy_val, mass=JuTrack.m_e)
    JuTrack.linepass!(line_bend_jt, beam_jt)
    particles_final_jutrack = beam_jt.r

    # Prepare coordinates
    coords = deepcopy(particles_initial)
    nparticles = size(coords, 1)
    lost_flags = zeros(Int, nparticles)

    # Tracking
    beam = Beam(energy_val)
    linepass!(coords, line_bend, beam, lost_flags)
    # Verification
    diff = norm(coords - particles_final_jutrack)

    @testset "JuTrack vs TrackPad Single Bend Verification" begin
        @test diff < 1e-15
        @test isapprox(coords, particles_final_jutrack, atol=1e-15)
    end
finally
    JuTrack.use_exact_beti = old_exact_beti
end
