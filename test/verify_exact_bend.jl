using Test
using TrackPad
using LinearAlgebra
Base.include(@__MODULE__, joinpath(@__DIR__, "convention_helpers.jl"))

# Deterministic initial coordinates shared across all regression tests.
const particles_initial = PARITY_PARTICLES
# JuTrack's exact bend uses `1/beta0 + delta_E` as a momentum magnitude. The
# frozen reference was taken in the shared ultrarelativistic limit; finite-beta
# canonical behavior is covered in verify_elements.jl.
energy_val = 1.0e15

# ExactSBend
angle_val = pi/9
# JuTrack ESBEND defaults: NumIntSteps=10
B_EXACT = ExactSBend(1.0, angle_val, angle_val/2, angle_val/2; name="B_EXACT", num_int_steps=10)

line_exact = Lattice([B_EXACT])

# JuTrack reference: [JuTrack.ESBEND(len=1.0, angle=pi/9, e1=pi/18, e2=pi/18)]
particles_final_jutrack = jutrack_reference("line/exact_bend")

beam = Beam(energy_val)
coords = deepcopy(particles_initial)
nparticles = size(coords, 1)
lost_flags = zeros(Int, nparticles)

# Tracking
linepass!(coords, line_exact, beam, lost_flags)

# Verification
diff = norm(coords - particles_final_jutrack)

# JuTrack's exact-bend body forms x_new = (pz_new − pzmx cos + px sin − 1)/h
# directly and loses ~ε/h ≈ 1e-15 m per step; TrackPad's `exact_bend_body`
# is the same algebra without the O(1) cancellation, so the codes agree to
# JuTrack's roundoff.
@testset "JuTrack vs TrackPad ExactSBend Verification" begin
    @test diff < 1e-13
    @test isapprox(coords, particles_final_jutrack, atol=1e-13)
end
