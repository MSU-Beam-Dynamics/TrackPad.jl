using Pkg
Pkg.activate(".")
Pkg.develop(path="../JuTrack.jl")
using TrackPad
using JuTrack
using StaticArrays
using LinearAlgebra
using Test

# Helper to check equality
function check_diff(name, v1, v2; tol=1e-14)
    diff = norm(v1 - v2)
    if diff > tol
        println("❌ $name mismatch: diff = $diff")
        println("  v1: $v1")
        println("  v2: $v2")
    else
        println("✅ $name match (diff = $diff)")
    end
end

println("=== Physics Primitive Debugging ===")

# 1. Drift
println("\n1. Drift6")
r0 = [0.001, 0.001, -0.001, 0.002, 0.0, 0.001]
L = 1.0
beti = 1.0

# JuTrack
r_ju = copy(r0)
JuTrack.drift6!(r_ju, L, beti)

# TrackPad
r_tp_s = SVector{6,Float64}(r0...)
r_tp = TrackPad.drift6(r_tp_s, L, beti)

check_diff("drift6", r_ju, Vector(r_tp))

# 2. Thin Bend Kick
println("\n2. Bend Thin Kick")
r0 = [0.001, 0.001, -0.001, 0.002, 0.0, 0.001]
L = 0.1
angle = 0.05
irho = angle / L
max_order = 0
PolynomA = zeros(4)
PolynomB = zeros(4)
PolynomB[2] = 1.0 # Quad
PolynomB[3] = 10.0 # Sext

# JuTrack
r_ju = copy(r0)
try
    JuTrack.bndthinkick!(r_ju, PolynomA, PolynomB, L, irho, 2, beti)
catch e
    println("JuTrack.bndthinkick! not accessible: $e")
    # Use local definition
    function bndthinkick_local!(r, A, B, L, irho, max_order, beti)
        ReSum = B[max_order + 1]
        ImSum = A[max_order + 1]
        ReSumTemp = 0.0
        for i in max_order-1:-1:0
            ReSumTemp = ReSum * r[1] - ImSum * r[3] + B[i+1]
            ImSum = ImSum * r[1] + ReSum * r[3] + A[i+1]
            ReSum = ReSumTemp
        end
        r[2] -= L * (ReSum - (r[6] - r[1] * irho) * irho)
        r[4] += L * ImSum
        r[5] += L * irho * r[1] * beti
    end
    bndthinkick_local!(r_ju, PolynomA, PolynomB, L, irho, 2, beti)
end

# TrackPad
r_tp_s = SVector{6,Float64}(r0...)
poly_a_s = SVector{4,Float64}(PolynomA...)
poly_b_s = SVector{4,Float64}(PolynomB...)
r_tp = TrackPad.bndthinkick(r_tp_s, poly_a_s, poly_b_s, L, irho, 2, beti)

check_diff("bndthinkick (Order 2)", r_ju, Vector(r_tp))

# 3. Edge Fringe Entrance
println("\n3. Edge Fringe Entrance")
r0 = [0.001, 0.001, -0.001, 0.002, 0.0, 0.001]
edge_angle = 0.1
fint = 0.5
gap = 0.02
method = 1

# JuTrack
r_ju = copy(r0)
try
    JuTrack.edge_fringe_entrance!(r_ju, irho, edge_angle, fint, gap, method)
catch e
    println("JuTrack.edge_fringe_entrance! not accessible: $e")
end

# TrackPad
r_tp_s = SVector{6,Float64}(r0...)
r_tp = TrackPad.edge_fringe_entrance(r_tp_s, irho, edge_angle, fint, gap, method)

if isdefined(Main, :r_ju)
    check_diff("edge_fringe_entrance", r_ju, Vector(r_tp))
end

# 4. Full SBend Pass (Full Element)
println("\n4. SBend Pass (Full Element)")
# Use verify_bend.jl parameters
len = 1.0
angle = 0.3490658503988659
e1 = angle/2
e2 = angle/2
num_steps = 10
energy = 3.5e9

# Initial particle from verify_bend failure
r0 = [-0.002919849600831203, -0.0009466154073801037, 0.0015117003899169953, 5.102556798674624e-7, 0.00023481000332187994, 0.0002164667906754636]

println("Testing with angle = $angle (20 deg)")

# JuTrack
bend_ju = JuTrack.SBEND(len=len, angle=angle, e1=e1, e2=e2, NumIntSteps=num_steps)
r_ju_mat = reshape(copy(r0), 1, 6)
beam_ju = JuTrack.Beam(zeros(1,6), energy=energy)
JuTrack.pass!(bend_ju, r_ju_mat, 1, beam_ju)
r_ju_final = r_ju_mat[1,:]

# TrackPad
bend_tp = TrackPad.SBend(len, angle, e1, e2; num_int_steps=num_steps)
r_tp_s = SVector{6,Float64}(r0...)
beam_tp = TrackPad.Beam(energy)
beti = 1.0 / beam_tp.beta
r_tp_final = TrackPad.pass!(bend_tp, r_tp_s, beti)

check_diff("SBend Pass", r_ju_final, Vector(r_tp_final); tol=1e-6)
