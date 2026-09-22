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

# Reconstruct DBA Lattice in TrackPad
# [D_L, B_DBA, D_S, Q_F, D_S, Q_D, D_S, Q_F, D_S, B_DBA, D_L]
# D_L = DRIFT(len=2.0)
# D_S = DRIFT(len=0.5)
# B_DBA = SBEND(len=1.0, angle=pi/9, e1=pi/18, e2=pi/18)
# Q_F = KQUAD(len=0.4, k1=1.5)
# Q_D = KQUAD(len=0.4, k1=-1.2)

D_L = Drift(2.0; name="D_L")
D_S = Drift(0.5; name="D_S")

angle_val = pi/9
# Note: JuTrack RBEND sets e1=e2=angle/2. We do the same here.
# JuTrack SBEND uses 10 integration steps by default; TrackPad now defaults
# to 4, so the parity model is pinned explicitly.
B_DBA = SBend(1.0, angle_val, angle_val/2, angle_val/2; name="B_DBA", num_int_steps=10)

# TrackPad Quadrupole and JuTrack KQUAD both use k1 directly.
# k1 = 1.5 for QF, -1.2 for QD.
# JuTrack KQUAD uses 10 integration steps by default. TrackPad should match.
Q_F = Quadrupole(0.4, 1.5; name="Q_F", num_int_steps=10)
Q_D = Quadrupole(0.4, -1.2; name="Q_D", num_int_steps=10)

line_dba = Lattice([D_L, B_DBA, D_S, Q_F, D_S, Q_D, D_S, Q_F, D_S, B_DBA, D_L])

# JuTrack reference with matching symplectic-equivalent definitions (the same
# eleven elements, with NumIntSteps=10 on the quadrupoles).
particles_final_jutrack = jutrack_reference("line/dba")

# Prepare coordinates
coords = deepcopy(particles_initial)
nparticles = size(coords, 1)
lost_flags = zeros(Int, nparticles)

# Tracking
beam = jutrack_beam(energy_val)
linepass!(coords, line_dba, beam, lost_flags)

# Verification
diff = norm(coords - particles_final_jutrack)

# Agreement is to JuTrack's roundoff (its drift z update subtracts two
# O(L) numbers; TrackPad's is cancellation-free), not to the bit.
@testset "JuTrack vs TrackPad DBA Verification" begin
    @test diff < 1e-14
    @test isapprox(coords, particles_final_jutrack, atol=1e-14)
end
