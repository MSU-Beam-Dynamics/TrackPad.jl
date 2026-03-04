using Test
using TrackPad
import JuTrack
using StaticArrays
using LinearAlgebra

function jutrack_linepass(r::SVector{6,Float64}, k1::Float64; num_steps::Int)
    line = [
        JuTrack.DRIFT(len=1.0),
        JuTrack.SBEND(len=0.4, angle=0.0, PolynomB=[0.0, k1, 0.0, 0.0], NumIntSteps=num_steps),
        JuTrack.DRIFT(len=1.0),
    ]
    beam = JuTrack.Beam(reshape(collect(r), 1, 6), energy=3.0e9, mass=JuTrack.m_e)
    JuTrack.linepass!(line, beam)
    return SVector{6,Float64}(vec(beam.r[1, :])...)
end

old_exact_beti = JuTrack.use_exact_beti
JuTrack.use_exact_beti = 1
try
    @testset "TrackPad Time-Dependent Parameters" begin
        t = Time()
        nturn = Turn()
        p = 2.0 + 0.5 * sin(t) + 0.1 * nturn
        @test isapprox(p(0.0), 2.0; atol=1e-15)
        @test isapprox(p(TimeContext(pi / 2; turn=3)), 2.8; atol=1e-15)

        q0 = Quadrupole(0.4, 0.6; name="QTD", num_int_steps=10)
        qtd = timed(q0; k1=0.6 + 0.2 * sin(t) + 0.01 * nturn,
                    num_int_steps=(ctx -> ((ctx isa TimeContext ? ctx.real_time : ctx) < 0.5 ? 8 : 12)))

        q_at_0 = materialize(qtd, 0.0)
        q_at_1 = materialize(qtd, TimeContext(1.0; turn=5))
        @test isapprox(q_at_0.k1, 0.6; atol=1e-15)
        @test isapprox(q_at_1.k1, 0.6 + 0.2 * sin(1.0) + 0.05; atol=1e-15)
        @test q_at_0.num_int_steps == 8
        @test q_at_1.num_int_steps == 12

        d = Drift(1.0; name="D")
        lat_static = Lattice([d, q0, d])
        lat_td = Lattice([d, qtd, d])
        beam = Beam(3.0e9)
        r0 = @SVector [1e-3, 2e-4, 5e-4, -1e-4, 0.0, 0.0]

        r_static = linepass(lat_static, r0, beam)
        r_t0 = linepass(lat_td, r0, beam; time=0.0, turn=0)
        r_t1 = linepass(lat_td, r0, beam; time=1.0, turn=5)
        r_jt_static = jutrack_linepass(r0, 0.6; num_steps=10)
        r_jt_t1 = jutrack_linepass(r0, 0.6 + 0.2 * sin(1.0) + 0.01 * 5; num_steps=12)

        # Remove turn-dependence to compare with original static lattice.
        lat_td_match = materialize_lattice(lat_td; time=0.0, turn=0)
        r_match = linepass(lat_td_match, r0, beam)
        @test isapprox(r_match, r_static; atol=1e-10)
        @test isapprox(r_static, r_jt_static; atol=1e-15)
        @test isapprox(r_t1, r_jt_t1; atol=1e-15)
        @test norm(r_t1 - r_t0) > 1e-12

        lat_turn0 = materialize_lattice(lat_td; time=0.2, turn=0)
        lat_turn7 = materialize_lattice(lat_td; time=0.2, turn=7)
        @test lat_turn0[2] isa Quadrupole
        @test lat_turn7[2] isa Quadrupole
        @test !isapprox(lat_turn0[2].k1, lat_turn7[2].k1; atol=1e-12)

        r_ring = ringpass(lat_td, r0, beam, 3; time=0.0, dt_turn=0.1, turn=4)
        r_manual = r0
        r_jt = r0
        for k in 0:2
            r_manual = linepass(lat_td, r_manual, beam; time=0.1 * k, turn=4 + k)
            k1 = 0.6 + 0.2 * sin(0.1 * k) + 0.01 * (4 + k)
            r_jt = jutrack_linepass(r_jt, k1; num_steps=8)
        end
        @test isapprox(r_ring, r_manual; atol=1e-12)
        @test isapprox(r_ring, r_jt; atol=1e-15)
    end
finally
    JuTrack.use_exact_beti = old_exact_beti
end
