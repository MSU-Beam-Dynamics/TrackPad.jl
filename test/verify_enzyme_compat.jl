using Test
using TrackPad
import JuTrack
using StaticArrays
using Enzyme

@testset "TrackPad Enzyme Compatibility" begin
    # Enzyme currently handles the linearized branch more robustly than the
    # exact-Hamiltonian sqrt-domain path.
    old_exact = TrackPad.USE_EXACT_HAMILTONIAN
    TrackPad.USE_EXACT_HAMILTONIAN = false

    function enzyme_fwd_derivative(f::Function, x::Float64)
        mode = Enzyme.set_runtime_activity(Enzyme.Forward)
        out = Enzyme.autodiff(
            mode,
            Enzyme.Const(f),
            Enzyme.Duplicated,
            Enzyme.Duplicated(x, 1.0),
        )
        d = out isa Tuple ? out[1] : out
        while !(d isa Number)
            if d isa NamedTuple
                d = first(values(d))
            elseif d isa Tuple
                d = d[1]
            else
                error("Unexpected Enzyme derivative container: $(typeof(d))")
            end
        end
        return d
    end

    try
        beam = Beam(3.0e9)
        d = Drift(1.0; name="D")
        r0 = @SVector [1e-3, 2e-4, 5e-4, -1e-4, 0.0, 0.0]

        function loss_static(k::Float64)
            q = Quadrupole(0.4, k; name="Q", num_int_steps=8)
            lat = Lattice([d, q, d])
            r = linepass(lat, r0, beam)
            return r[1]
        end
        function loss_static_jt(k::Float64)
            line = [
                JuTrack.DRIFT(len=1.0),
                JuTrack.SBEND(len=0.4, angle=0.0, PolynomB=[0.0, k, 0.0, 0.0]),
                JuTrack.DRIFT(len=1.0),
            ]
            beam_jt = JuTrack.Beam(reshape(collect(r0), 1, 6), energy=3.0e9, mass=JuTrack.m_e)
            JuTrack.linepass!(line, beam_jt)
            return beam_jt.r[1, 1]
        end

        k0 = 0.6
        h = 1e-6
        fd_static = (loss_static(k0 + h) - loss_static(k0 - h)) / (2h)
        fd_static_jt = (loss_static_jt(k0 + h) - loss_static_jt(k0 - h)) / (2h)
        grad_static = enzyme_fwd_derivative(loss_static, k0)
        @test isapprox(fd_static, fd_static_jt; atol=1e-9)
        @test isfinite(grad_static)
        @test isapprox(grad_static, fd_static_jt; rtol=1e-5, atol=1e-7)

        t = Time()
        q0 = Quadrupole(0.4, 0.6; name="QTD", num_int_steps=8)
        qtd = timed(q0; k1=0.6 + 0.15 * sin(t))
        lat_td = Lattice([d, qtd, d])

        function loss_time(real_time::Float64)
            lat_now = materialize_lattice(lat_td; time=real_time, turn=3)
            r = linepass(lat_now, r0, beam)
            return r[1]
        end
        function loss_time_jt(real_time::Float64)
            k = 0.6 + 0.15 * sin(real_time)
            line = [
                JuTrack.DRIFT(len=1.0),
                JuTrack.SBEND(len=0.4, angle=0.0, PolynomB=[0.0, k, 0.0, 0.0]),
                JuTrack.DRIFT(len=1.0),
            ]
            beam_jt = JuTrack.Beam(reshape(collect(r0), 1, 6), energy=3.0e9, mass=JuTrack.m_e)
            JuTrack.linepass!(line, beam_jt)
            return beam_jt.r[1, 1]
        end

        t0 = 0.3
        fd_time = (loss_time(t0 + h) - loss_time(t0 - h)) / (2h)
        fd_time_jt = (loss_time_jt(t0 + h) - loss_time_jt(t0 - h)) / (2h)
        grad_time = enzyme_fwd_derivative(loss_time, t0)
        @test isapprox(fd_time, fd_time_jt; atol=1e-9)
        @test isfinite(grad_time)
        @test abs(grad_time) > 1e-10
        @test sign(grad_time) == sign(fd_time_jt)
        ratio = abs(grad_time / fd_time_jt)
        @test 0.2 <= ratio <= 2.0
    finally
        TrackPad.USE_EXACT_HAMILTONIAN = old_exact
    end
end
