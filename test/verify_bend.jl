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

# Reconstruct Bend Lattice in TrackPad
# B_DBA = SBEND(len=1.0, angle=pi/9, e1=pi/18, e2=pi/18)
angle_val = pi/4
# JuTrack SBEND uses 10 integration steps by default; TrackPad now defaults
# to 4, so the parity model is pinned explicitly.
B_DBA = SBend(1.0, angle_val, angle_val/2, angle_val/2; name="B_DBA", num_int_steps=10)

line_bend = Lattice([B_DBA])

# JuTrack reference: [JuTrack.SBEND(len=1.0, angle=pi/4, e1=pi/8, e2=pi/8)]
particles_final_jutrack = jutrack_reference("line/single_bend")

# Prepare coordinates
coords = deepcopy(particles_initial)
nparticles = size(coords, 1)
lost_flags = zeros(Int, nparticles)

# Tracking
beam = jutrack_beam(energy_val)
linepass!(coords, line_bend, beam, lost_flags)
# Verification
diff = norm(coords - particles_final_jutrack)

@testset "JuTrack vs TrackPad Single Bend Verification" begin
    @test diff < 1e-15
    @test isapprox(coords, particles_final_jutrack, atol=1e-15)
end
