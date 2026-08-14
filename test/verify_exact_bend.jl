using Test
using TrackPad
using LinearAlgebra
import JuTrack
Base.include(@__MODULE__, joinpath(@__DIR__, "convention_helpers.jl"))

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
# JuTrack's exact bend uses `1/beta0 + delta_E` as a momentum magnitude.
# Compare only the shared ultrarelativistic limit; finite-beta canonical
# behavior is covered in verify_elements.jl.
energy_val = 1.0e15
old_exact_beti = JuTrack.use_exact_beti
JuTrack.use_exact_beti = 1

# ExactSBend
angle_val = pi/9
# JuTrack ESBEND defaults: NumIntSteps=10
B_EXACT = ExactSBend(1.0, angle_val, angle_val/2, angle_val/2; name="B_EXACT", num_int_steps=10)

line_exact = Lattice([B_EXACT])

# JuTrack reference.
line_exact_jt = [JuTrack.ESBEND(len=1.0, angle=angle_val, e1=angle_val / 2, e2=angle_val / 2)]
try
    beam_jt = JuTrack.Beam(flip_longitudinal_coordinate(particles_initial), energy=energy_val, mass=JuTrack.m_e)
    JuTrack.linepass!(line_exact_jt, beam_jt)
    particles_final_jutrack = flip_longitudinal_coordinate(beam_jt.r)

    beam = Beam(energy_val)
    coords = deepcopy(particles_initial)
    nparticles = size(coords, 1)
    lost_flags = zeros(Int, nparticles)

    # Tracking
    linepass!(coords, line_exact, beam, lost_flags)

    # Verification
    diff = norm(coords - particles_final_jutrack)

    @testset "JuTrack vs TrackPad ExactSBend Verification" begin
        @test diff < 1e-15
        @test isapprox(coords, particles_final_jutrack, atol=1e-15)
    end
finally
    JuTrack.use_exact_beti = old_exact_beti
end
