using Test
using TrackPad
using LinearAlgebra
Base.include(@__MODULE__, joinpath(@__DIR__, "convention_helpers.jl"))

# Deterministic initial coordinates shared across all regression tests.
const particles_initial = PARITY_PARTICLES
# JuTrack's bend kick mixes delta_P and delta_E at finite beta. The frozen
# reference was taken in the shared ultrarelativistic limit; finite-beta
# behavior is tested directly in verify_elements.jl.
energy_val = 1.0e15

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

# JuTrack reference with matching symplectic-equivalent definitions:
# DRIFT(1.0), SBEND(len=1.0, angle=0, PolynomB=[0,-0.9,0,0]), DRIFT(1.0),
# SBEND(len=0.6, angle=pi/15), DRIFT(1.0),
# SBEND(len=1.0, angle=0, PolynomB=[0,0.3,0,0]), DRIFT(1.0).
particles_final_jutrack = jutrack_reference("line/fodo")

# Prepare coordinates
coords = deepcopy(particles_initial)
nparticles = size(coords, 1)
lost_flags = zeros(Int, nparticles)

# Tracking
beam = Beam(energy_val)
linepass!(coords, line, beam, lost_flags)

# Verification
diff = norm(coords - particles_final_jutrack)

# TrackPad's drift evaluates the z update without subtracting two O(L)
# numbers; JuTrack's has ~ε·L of roundoff per drift, so the agreement is to
# JuTrack's roundoff, not to the bit.
@testset "JuTrack vs TrackPad FODO Verification" begin
    @test diff < 1e-14
    @test isapprox(coords, particles_final_jutrack, atol=1e-14)
end
