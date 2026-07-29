using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "JuTrack.jl")))
Pkg.instantiate()
using JuTrack
using Serialization
using Random

# Deterministic random seed
Random.seed!(1234)

println("Generating ground truth data using JuTrack...")

# Lattice Definition (from README)
D1 = DRIFT(name="D1", len=1.0)
D2 = DRIFT(name="D2", len=1.0)
D3 = DRIFT(name="D3", len=1.0)
D4 = DRIFT(name="D4", len=1.0)
# Use SBEND with angle=0 for Quadrupoles to enforce symplectic tracking in JuTrack
# PolynomB indices: 1->Dipole, 2->Quad, 3->Sext, 4->Oct
Q1 = SBEND(name="Q1", len=1.0, angle=0.0, PolynomB=[0.0, -0.9, 0.0, 0.0])
Q2 = SBEND(name="Q2", len=1.0, angle=0.0, PolynomB=[0.0, 0.3, 0.0, 0.0])
B1 = SBEND(name="B1", len=0.6, angle=pi/15.0)

println("Q1 type: ", typeof(Q1))
println("Q1 PolynomB: ", Q1.PolynomB)

LINE = [D1, Q1, D2, B1, D3, Q2, D4]

# Beam Definition
# 10 particles, 6 dimensions
particles = rand(10, 6) / 1000
initial_particles = copy(particles) # Keep a copy of initial state
energy_val = 3.5e9
beam = Beam(particles, energy=energy_val)

println("Initial beam:")
println(beam.r[1,:])

# Tracking FODO
linepass!(LINE, beam)

println("Final beam (FODO):")
println(beam.r[1,:])

# Save FODO data
data = Dict(
    "particles_initial" => initial_particles,
    "energy" => energy_val,
    "particles_final" => beam.r,
)
serialize("fodo_golden.ser", data)
println("Saved FODO ground truth to fodo_golden.ser")

# ==========================================
# DBA Generation
# ==========================================
println("\nGenerating DBA ground truth...")

# Define DBA Lattice
# [Drift, Bend, Drift, Quad, Drift, Quad, Drift, Bend, Drift]
D_L = DRIFT(len=2.0)
D_S = DRIFT(len=0.5)
# RBend equivalent (sector bend with e1=e2=angle/2)
angle_val = pi/9
B_DBA = SBEND(len=1.0, angle=angle_val, e1=angle_val/2, e2=angle_val/2)
# Symplectic Quads
Q_F = SBEND(len=0.4, angle=0.0, PolynomB=[0.0, 1.5, 0.0, 0.0])
Q_D = SBEND(len=0.4, angle=0.0, PolynomB=[0.0, -1.2, 0.0, 0.0])

LINE_DBA = [D_L, B_DBA, D_S, Q_F, D_S, Q_D, D_S, Q_F, D_S, B_DBA, D_L]

# Reset beam - Use COPY to avoid modifying initial_particles if Beam aliases it
beam_dba = Beam(copy(initial_particles), energy=energy_val)

# Tracking DBA
linepass!(LINE_DBA, beam_dba)

println("Final beam (DBA):")
println(beam_dba.r[1,:])

# Save DBA data
data_dba = Dict(
    "particles_initial" => initial_particles, # This should be the original random state
    "energy" => energy_val,
    "particles_final" => beam_dba.r,
)
serialize("dba_golden.ser", data_dba)
println("Saved DBA ground truth to dba_golden.ser")

# ==========================================
# Single Bend Generation
# ==========================================
println("\nGenerating Single Bend ground truth...")
LINE_BEND = [B_DBA]
# Use COPY
beam_bend = Beam(copy(initial_particles), energy=energy_val)
linepass!(LINE_BEND, beam_bend)
println("Final beam (Single Bend):")
println(beam_bend.r[1,:])
println("B_DBA details: angle=", B_DBA.angle, ", len=", B_DBA.len, ", e1=", B_DBA.e1, ", e2=", B_DBA.e2)

data_bend = Dict(
    "particles_initial" => initial_particles,
    "energy" => energy_val,
    "particles_final" => beam_bend.r,
)
serialize("bend_golden.ser", data_bend)
println("Saved Single Bend ground truth to bend_golden.ser")

# ==========================================
# Exact Bend Generation (ESBEND)
# ==========================================
println("\nGenerating Exact Bend (ESBEND) ground truth...")
# JuTrack ESBEND
B_ESBEND = ESBEND(len=1.0, angle=angle_val, e1=angle_val/2, e2=angle_val/2)
LINE_EXACT = [B_ESBEND]
beam_exact = Beam(copy(initial_particles), energy=energy_val)
linepass!(LINE_EXACT, beam_exact)
println("Final beam (Exact Bend):")
println(beam_exact.r[1,:])

data_exact = Dict(
    "particles_initial" => initial_particles,
    "energy" => energy_val,
    "particles_final" => beam_exact.r,
)
serialize("exact_bend_golden.ser", data_exact)
println("Saved Exact Bend ground truth to exact_bend_golden.ser")

# Save DBA data
data_dba = Dict(
    "particles_initial" => initial_particles,
    "energy" => energy_val,
    "particles_final" => beam_dba.r,
)
serialize("dba_golden.ser", data_dba)
println("Saved DBA ground truth to dba_golden.ser")
