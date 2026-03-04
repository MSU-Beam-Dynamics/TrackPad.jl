using Test
using TrackPad
import JuTrack
using LinearAlgebra

# Simple periodic cell to validate optics APIs
D = Drift(1.0; name="D")
QF = Quadrupole(0.5, 0.6; name="QF", num_int_steps=8)
QD = Quadrupole(0.5, -0.6; name="QD", num_int_steps=8)

ring = Lattice([D, QF, D, QD])
beam = Beam(3.0e9)

ring_jt = [
    JuTrack.DRIFT(len=1.0),
    JuTrack.KQUAD(len=0.5, k1=0.6, NumIntSteps=8),
    JuTrack.DRIFT(len=1.0),
    JuTrack.KQUAD(len=0.5, k1=-0.6, NumIntSteps=8),
]
tune_jt = JuTrack.gettune(ring_jt; energy=3.0e9, mass=JuTrack.m_e)
chrom_jt = JuTrack.getchrom(ring_jt; energy=3.0e9, mass=JuTrack.m_e)

@testset "TrackPad Optics API" begin
    qx, qy = gettune(ring, beam)
    @test isfinite(qx)
    @test isfinite(qy)
    @test 0.0 <= qx <= 1.0
    @test 0.0 <= qy <= 1.0
    @test isapprox(qx, tune_jt[1]; atol=1e-15)
    @test isapprox(qy, tune_jt[2]; atol=1e-15)

    ξx, ξy = getchrom(ring, beam; dp=0.0)
    @test isfinite(ξx)
    @test isfinite(ξy)
    @test isapprox(ξx, chrom_jt[1]; atol=1e-8)
    @test isapprox(ξy, chrom_jt[2]; atol=1e-8)

    tw = twissline(ring, beam)
    @test length(tw.s) == length(ring) + 1
    @test length(tw.betax) == length(ring) + 1
    @test length(tw.betay) == length(ring) + 1
    @test all(tw.betax .> 0)
    @test all(tw.betay .> 0)
    @test isapprox(tw.tunex, qx; atol=1e-12)
    @test isapprox(tw.tuney, qy; atol=1e-12)
end

@testset "TrackPad JuTrack-Compatible Optics/Map Interface" begin
    m66 = fastfindm66(ring, 0.0; E0=3.0e9, m0=M_ELECTRON)
    m66_ref = one_turn_map(ring, beam)
    @test isapprox(m66, m66_ref; atol=1e-7)

    m66_ord0 = findm66(ring, 0.0, 0; E0=3.0e9, m0=M_ELECTRON)
    @test isapprox(m66_ord0, m66; atol=1e-12)

    m66_jt = JuTrack.fastfindm66(ring_jt, 0.0; E0=3.0e9, m0=JuTrack.m_e)
    @test isapprox(m66, m66_jt; atol=1e-7)

    refpts = [2, 4]
    mref = fastfindm66_refpts(ring, 0.0, refpts; E0=3.0e9, m0=M_ELECTRON)
    @test size(mref) == (6, 6, 2)
    @test isapprox(mref[:, :, 2] * mref[:, :, 1], m66; atol=5e-7)

    op = periodicEdwardsTengTwiss(ring, 0.0, 0; E0=3.0e9, m0=M_ELECTRON)
    @test op isa optics4DUC
    @test op.optics_x.beta > 0
    @test op.optics_y.beta > 0

    op_jt = JuTrack.periodicEdwardsTengTwiss(ring_jt, 0.0, 0; E0=3.0e9, m0=JuTrack.m_e)
    @test isapprox(op.optics_x.beta, op_jt.betax; atol=1e-13)
    @test isapprox(op.optics_x.alpha, op_jt.alphax; atol=1e-13)
    @test isapprox(op.optics_y.beta, op_jt.betay; atol=1e-13)
    @test isapprox(op.optics_y.alpha, op_jt.alphay; atol=1e-13)

    twr = twissring(ring, 0.0, 0; E0=3.0e9, m0=M_ELECTRON)
    @test twr isa TwissLineResult
    @test length(twr.s) == length(ring) + 1
    @test isapprox(twr.tunex, tune_jt[1]; atol=5e-7)
    @test isapprox(twr.tuney, tune_jt[2]; atol=5e-7)

    twr_ref = twissring(ring, 0.0, 0, refpts; E0=3.0e9, m0=M_ELECTRON)
    @test length(twr_ref) == length(refpts)
    @test twr_ref[1] isa optics4DUC

    tin = optics4DUC(op.optics_x, op.optics_y)
    tout = twissline(tin, ring, 0.0, 0, 4; E0=3.0e9, m0=M_ELECTRON)
    @test tout isa optics4DUC
    @test isapprox(tout.optics_x.beta, op.optics_x.beta; atol=1e-13)
    @test isapprox(tout.optics_x.alpha, op.optics_x.alpha; atol=1e-13)
    @test isapprox(tout.optics_y.beta, op.optics_y.beta; atol=1e-13)
    @test isapprox(tout.optics_y.alpha, op.optics_y.alpha; atol=1e-13)

    twline_ref = twissline(tin, ring, 0.0, 0, refpts; E0=3.0e9, m0=M_ELECTRON)
    @test length(twline_ref) == length(refpts)
    @test twline_ref[end] isa optics4DUC
    twj_ref = JuTrack.twissring(ring_jt, 0.0, 0, refpts; E0=3.0e9, m0=JuTrack.m_e)
    for i in eachindex(refpts)
        @test isapprox(twline_ref[i].optics_x.beta, twj_ref[i].betax; atol=1e-13)
        @test isapprox(twline_ref[i].optics_x.alpha, twj_ref[i].alphax; atol=1e-13)
        @test isapprox(twline_ref[i].optics_y.beta, twj_ref[i].betay; atol=1e-13)
        @test isapprox(twline_ref[i].optics_y.alpha, twj_ref[i].alphay; atol=1e-13)
    end
end
